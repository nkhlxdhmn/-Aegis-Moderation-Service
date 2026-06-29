"""Video post moderation pipeline for Aegis.

Workflow:
  1. Validate video file.
  2. Extract one frame per second with ffmpeg (capped at MAX_FRAMES).
  3. Moderate all frames concurrently via batch_analyzer.analyze_batch(),
     which reuses the full image pipeline (NSFW, YOLO, SigLIP2, OCR, BLIP,
     Qwen2.5-VL, Llama).
  4. Aggregate per-frame scores: max score per key + temporal metrics.
  5. Extract audio with ffmpeg â†’ 16 kHz mono WAV.
  6. Transcribe audio with Whisper-large-v3.
  7. Moderate transcript with text_moderation.moderate_text().
  8. Merge frame scores + transcript scores + video temporal metrics.
  9. Feed final scores dict to decision_engine.decide_with_reason_code().

Video-specific score keys (read by Tier 0-D in decision_engine.py):
  video_unsafe_frame_ratio        â€” fraction of frames classified as unsafe
  video_consecutive_unsafe_frames â€” longest run of consecutive unsafe frames
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

WHISPER_MODEL_ID = "openai/whisper-large-v3"
WHISPER_DEVICE = "cuda:1"          # shares GPU with Llama / BLIP

MAX_FRAMES = 120                   # cap at 2 minutes @ 1 fps to prevent OOM
FRAME_FPS = 1                      # frames to extract per second
FRAME_JPEG_QUALITY = 2             # ffmpeg -q:v (1=best, 31=worst); 2 = high quality

# A frame is "unsafe" when any key danger score exceeds this threshold.
_FRAME_UNSAFE_THRESHOLD = 0.55

# â”€â”€ Whisper singleton â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class _WhisperState:
    pipe: object


_whisper_state: _WhisperState | None = None
_whisper_lock = threading.Lock()


def _get_whisper() -> _WhisperState | None:
    global _whisper_state
    if _whisper_state is not None:
        return _whisper_state
    with _whisper_lock:
        if _whisper_state is not None:
            return _whisper_state
        try:
            from transformers import pipeline as hf_pipeline

            logger.info("Loading Whisper-large-v3 on %s", WHISPER_DEVICE)
            pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL_ID,
                chunk_length_s=30,
                device=WHISPER_DEVICE,
            )
            _whisper_state = _WhisperState(pipe=pipe)
            logger.info("Whisper-large-v3 loaded")
        except ImportError:
            logger.warning("transformers not available â€” audio transcription disabled")
        except Exception:
            logger.exception("Whisper failed to load â€” audio transcription disabled")
    return _whisper_state


# â”€â”€ Result dataclass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass(frozen=True)
class VideoModerationResult:
    scores: dict[str, float]             # decision_engine-compatible merged scores
    frame_scores: list[dict[str, float]] # per-frame score snapshots (subset of keys)
    transcript: str                      # Whisper transcript (empty if unavailable)
    text_scores: dict[str, float]        # text_moderation scores for transcript
    frame_count: int
    unsafe_frame_count: int
    max_consecutive_unsafe: int
    pipeline_error: bool = False
    error_reason: str | None = None


# â”€â”€ ffmpeg helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _run_ffmpeg(args: list[str], description: str) -> bool:
    """Run an ffmpeg command. Returns True on success."""
    try:
        result = subprocess.run(
            ["ffmpeg", *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:500]
            logger.error("ffmpeg %s failed (rc=%d): %s", description, result.returncode, stderr)
            return False
        return True
    except FileNotFoundError:
        logger.error("ffmpeg not found â€” install ffmpeg and ensure it is on PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg %s timed out", description)
        return False
    except Exception:
        logger.exception("ffmpeg %s raised an exception", description)
        return False


def _extract_frames(video_path: str, output_dir: str) -> list[str]:
    """Extract 1 frame/second as JPEG files. Returns sorted list of paths."""
    pattern = os.path.join(output_dir, "frame_%06d.jpg")
    ok = _run_ffmpeg(
        [
            "-i", video_path,
            "-vf", f"fps={FRAME_FPS}",
            "-q:v", str(FRAME_JPEG_QUALITY),
            "-frames:v", str(MAX_FRAMES),
            pattern,
            "-y",
        ],
        "frame extraction",
    )
    if not ok:
        return []
    frames = sorted(
        str(p) for p in Path(output_dir).glob("frame_*.jpg")
    )
    logger.info("Extracted %d frames from %s", len(frames), video_path)
    return frames


def _extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract 16 kHz mono PCM WAV for Whisper."""
    return _run_ffmpeg(
        [
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path,
            "-y",
        ],
        "audio extraction",
    )


# â”€â”€ Transcription â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _transcribe_audio(audio_path: str) -> str:
    """Return Whisper transcript, or empty string on failure."""
    state = _get_whisper()
    if state is None:
        return ""
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
        logger.info("Audio file too small or missing â€” skipping transcription")
        return ""
    try:
        logger.info("Whisper transcription started")
        result = state.pipe(audio_path, return_timestamps=False)
        transcript = (result if isinstance(result, str) else result.get("text", ""))
        transcript = str(transcript).strip()
        logger.info("Whisper transcription completed (%d chars)", len(transcript))
        return transcript
    except Exception:
        logger.exception("Whisper transcription failed")
        return ""


# â”€â”€ Per-frame risk helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _frame_risk(scores: dict[str, float]) -> float:
    """Compute a single risk value for one frame."""
    return max(
        scores.get("adult_score", 0.0),
        scores.get("violence_self_harm_score", 0.0),
        scores.get("weapon_score", 0.0),
        scores.get("blood_score", 0.0),
        scores.get("child_safety_score", 0.0),
    )


def _max_consecutive_unsafe(flags: list[bool]) -> int:
    max_run = current_run = 0
    for flag in flags:
        if flag:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


_FRAME_SNAPSHOT_KEYS = (
    "adult_score",
    "violence_self_harm_score",
    "weapon_score",
    "blood_score",
    "child_safety_score",
    "heritage_score",
    "ensemble_risk_score",
)


def _snapshot(scores: dict[str, float]) -> dict[str, float]:
    """Extract a lightweight snapshot of a frame's key scores."""
    return {k: round(scores.get(k, 0.0), 4) for k in _FRAME_SNAPSHOT_KEYS}


# â”€â”€ Score aggregation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _aggregate_frame_scores(
    frame_results: list,  # list[ModerationPipelineResult]
) -> tuple[dict[str, float], list[bool], list[dict[str, float]]]:
    """Return (max_scores, unsafe_flags, snapshots)."""
    max_scores: dict[str, float] = {}
    flags: list[bool] = []
    snapshots: list[dict[str, float]] = []

    for result in frame_results:
        scores = result.scores
        for key, val in scores.items():
            if isinstance(val, (int, float)):
                max_scores[key] = max(max_scores.get(key, 0.0), float(val))
        risk = _frame_risk(scores)
        flags.append(risk >= _FRAME_UNSAFE_THRESHOLD)
        snapshots.append(_snapshot(scores))

    return max_scores, flags, snapshots


def _default_video_scores() -> dict[str, float]:
    """Zero-filled fallback compatible with decision_engine."""
    from pipeline.safety_flags import _default_scores
    base = _default_scores()
    base["video_unsafe_frame_ratio"] = 0.0
    base["video_consecutive_unsafe_frames"] = 0.0
    base["nsfw_score"] = 0.0
    base["visual_explicit_indicator"] = 0.0
    base["marketing_keyword_count"] = 0.0
    base["course_promotion_score"] = 0.0
    base["political_score"] = 0.0
    base["political_campaign_score"] = 0.0
    base["animal_cruelty_text_score"] = 0.0
    base["human_killing_text_score"] = 0.0
    return base


def _merge_scores(
    frame_max: dict[str, float],
    text_scores: dict[str, float],
    unsafe_ratio: float,
    consecutive: int,
    frame_count: int,
) -> dict[str, float]:
    """Produce the final decision_engine-compatible scores dict."""
    final = _default_video_scores()

    # Frame max scores override defaults (take max with existing defaults)
    for key, val in frame_max.items():
        if key in final and isinstance(val, (int, float)):
            final[key] = max(final[key], float(val))

    # Merge text/transcript scores (take max per key)
    for key, val in text_scores.items():
        if key in final and isinstance(val, (int, float)):
            final[key] = max(final[key], float(val))

    # Video temporal metrics
    final["video_unsafe_frame_ratio"] = unsafe_ratio
    final["video_consecutive_unsafe_frames"] = float(consecutive)

    # Propagate nsfw_score from adult_score
    final["nsfw_score"] = final["adult_score"]

    # Boost ensemble risk with video-specific signals
    video_temporal_risk = min(1.0, unsafe_ratio * 1.5)
    final["ensemble_risk_score"] = max(
        final.get("ensemble_risk_score", 0.0),
        video_temporal_risk,
    )

    return final


# â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def moderate_video(
    video_path: str,
    metadata: dict | None = None,
) -> VideoModerationResult:
    """Moderate a video post end-to-end.

    Args:
        video_path: Absolute path to the video file.
        metadata:   Optional dict with caption, user context, etc.

    Returns:
        VideoModerationResult whose .scores dict is compatible with
        decision_engine.decide_with_reason_code().
    """
    if not os.path.isfile(video_path):
        return VideoModerationResult(
            scores=_default_video_scores(),
            frame_scores=[],
            transcript="",
            text_scores={},
            frame_count=0,
            unsafe_frame_count=0,
            max_consecutive_unsafe=0,
            pipeline_error=True,
            error_reason=f"Video file not found: {video_path}",
        )

    logger.info("Video moderation pipeline started: %s", video_path)
    caption: str = (metadata or {}).get("caption", "") or ""

    try:
        with tempfile.TemporaryDirectory(prefix="vidmod_") as tmp:
            frames_dir = os.path.join(tmp, "frames")
            audio_path = os.path.join(tmp, "audio.wav")
            os.makedirs(frames_dir, exist_ok=True)

            # â”€â”€ Stage 1: Frame extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            frame_paths = _extract_frames(video_path, frames_dir)
            if not frame_paths:
                return VideoModerationResult(
                    scores=_default_video_scores(),
                    frame_scores=[],
                    transcript="",
                    text_scores={},
                    frame_count=0,
                    unsafe_frame_count=0,
                    max_consecutive_unsafe=0,
                    pipeline_error=True,
                    error_reason="Frame extraction failed â€” check ffmpeg installation.",
                )

            frame_count = len(frame_paths)

            # â”€â”€ Stage 2: Concurrent frame moderation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            from pipeline.batch_analyzer import analyze_batch

            items: list[tuple[str, str | None]] = [
                (fp, caption or None) for fp in frame_paths
            ]
            frame_results = analyze_batch(items, queue_depth=frame_count)
            logger.info("Frame moderation completed (%d frames)", len(frame_results))

            # â”€â”€ Stage 3: Aggregate frame scores â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            frame_max, unsafe_flags, snapshots = _aggregate_frame_scores(frame_results)
            unsafe_count = sum(unsafe_flags)
            consecutive = _max_consecutive_unsafe(unsafe_flags)
            unsafe_ratio = unsafe_count / frame_count if frame_count > 0 else 0.0

            logger.info(
                "Frame metrics: %d/%d unsafe (%.1f%%), max consecutive=%d",
                unsafe_count, frame_count, unsafe_ratio * 100, consecutive,
            )

            # â”€â”€ Stage 4: Audio extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            has_audio = _extract_audio(video_path, audio_path)

            # â”€â”€ Stage 5: Whisper transcription â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            transcript = ""
            if has_audio:
                transcript = _transcribe_audio(audio_path)

            # â”€â”€ Stage 6: Text moderation on transcript â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            text_scores: dict[str, float] = {}
            if transcript:
                from pipeline.text_moderation import moderate_text
                text_result = moderate_text(transcript, metadata=metadata)
                text_scores = text_result.scores
                logger.info(
                    "Transcript moderation completed (ensemble=%.3f)",
                    text_scores.get("ensemble_risk_score", 0.0),
                )

            # â”€â”€ Stage 7: Merge all signals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            final_scores = _merge_scores(
                frame_max=frame_max,
                text_scores=text_scores,
                unsafe_ratio=unsafe_ratio,
                consecutive=consecutive,
                frame_count=frame_count,
            )

            logger.info(
                "Video moderation completed (unsafe_ratio=%.3f, consecutive=%d, "
                "ensemble=%.3f)",
                unsafe_ratio, consecutive, final_scores["ensemble_risk_score"],
            )

            return VideoModerationResult(
                scores=final_scores,
                frame_scores=snapshots,
                transcript=transcript,
                text_scores=text_scores,
                frame_count=frame_count,
                unsafe_frame_count=unsafe_count,
                max_consecutive_unsafe=consecutive,
            )

    except Exception as exc:
        logger.exception("Video moderation pipeline failed")
        return VideoModerationResult(
            scores=_default_video_scores(),
            frame_scores=[],
            transcript="",
            text_scores={},
            frame_count=0,
            unsafe_frame_count=0,
            max_consecutive_unsafe=0,
            pipeline_error=True,
            error_reason=str(exc),
        )

