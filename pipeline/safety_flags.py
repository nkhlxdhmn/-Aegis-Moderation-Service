"""Central orchestration for Aegis image moderation signals.

Pipeline stages (Phase 5 â€” maximum accuracy):
  Stage 0   â€” Image quality gate
  Stage 0b  â€” Hash dedup cache
  Stage 1   â€” Parallel NSFW + YOLO (GPU0)
  Stage 2   â€” Smart OCR (GPU0) â€” text_detector gates OCR on text density / YOLO class
  Stage 3   â€” SigLIP2 (GPU0) â€” category / heritage / safety embeddings
  Stage 3b  â€” Embedding similarity search (FAISS + SigLIP) â€” cache hit short-circuits GPU1
  Stage 4   â€” Signal fusion + cultural protection (child safety dominance)
  Stage 5   â€” Text / PII rules
  Stage 6   â€” BLIP multi-caption (GPU1, 3 captions)
  Stage 7   â€” Qwen2.5-VL description (GPU1)
  Stage 7b  â€” Qwen secondary verification for borderline content (0.60â€“0.85 pre-LLM risk)
  Stage 8   â€” Llama-3.1-8B AWQ reasoning with all evidence (GPU1, always runs)
  Stage 9   â€” Uncertainty estimation
  Stage 10  â€” Confidence-weighted ensemble
  Stage 11  â€” Hash store + Embedding store

Phase 5 changes from Phase 4:
  - Smart OCR: text_detector gates OCR (text_density/entropy/YOLO class)
  - Qwen2.5-VL-7B for detailed scene description (GPU1)
  - Qwen secondary verification pass for 0.60 < pre_llm_risk < 0.85
  - Confidence-weighted Llama: effective_llama = llama_risk * llama_confidence
  - Uncertainty estimation: 4-signal composite, routes HIGH_UNCERTAINTY to review
  - FAISS embedding cache: similarity > 0.93 reuses prior moderation decision
  - Child safety dominance: heritage never reduces child safety scores below raw
  - Extended evidence: qwen_description, qwen_confidence, uncertainty_score, similar_images
"""

from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline import (
    child_safety,
    circuit_breaker as cb,
    clip_engine,
    embedding_cache,
    hard_block,
    hash_cache,
    image_quality,
    language_detector,
    metrics as _metrics,
    ml_toxicity,
    nsfw,
    object_detector,
    ocr,
    pii_detector,
    promotion_detector,
    qr_detector,
    qwen_vl,
    safety_detector,
    text_classifier,
    text_detector,
    text_safety,
    timeout_utils,
    uncertainty as uncertainty_module,
    vlm_engine,
)
from pipeline.calibration import calibrate_yolo

logger = logging.getLogger(__name__)

# Heritage keywords for cultural-protection text-based detection.
HERITAGE_LABELS: frozenset[str] = frozenset({
    "temple", "idol", "shiva", "krishna", "hanuman", "ramayana", "mahabharata",
    "festival", "rangoli", "diya", "heritage", "pilgrimage",
})

_HERITAGE_PROTECT_THRESHOLD = 0.60

# Pre-LLM risk range that triggers Qwen secondary verification.
_SECONDARY_VERIFY_LOW  = 0.60
_SECONDARY_VERIFY_HIGH = 0.85

MODEL_VERSIONS: dict[str, str] = {
    "nsfw":    "Falconsai/nsfw_image_detection",
    "siglip":  "google/siglip2-large-patch16-384",
    "yolo":    "yolo11x",
    "ocr":     "Surya+EasyOCR-hybrid",
    "blip":    "Salesforce/blip-image-captioning-large",
    "llama":   "Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "qwen":    "Qwen/Qwen2.5-VL-7B-Instruct",
}


@dataclass(frozen=True)
class ModerationPipelineResult:
    """All model outputs + status for one moderated image."""

    scores: dict[str, float]
    category_scores: dict[str, float]
    ocr_text: str
    pipeline_error: bool = False
    error_reason: str | None = None
    llama_result: dict | None = None
    # Evidence fields
    detected_objects: list[str] = field(default_factory=list)
    generated_caption: str = ""
    generated_captions: list[str] = field(default_factory=list)
    model_versions: dict[str, str] = field(default_factory=dict)
    image_hash: str | None = None
    # Phase 5 evidence
    qwen_description: str = ""
    qwen_confidence: float = 0.5
    uncertainty_score: float = 0.0
    similar_images: list[dict] = field(default_factory=list)


def _default_scores(**overrides: float) -> dict[str, float]:
    scores: dict[str, float] = {
        "adult_score": 0.0,
        "heritage_score": 0.0,
        "content_quality_score": 0.0,
        "child_safety_score": 0.0,
        "child_presence_score": 0.0,
        "violence_self_harm_score": 0.0,
        "weapon_score": 0.0,
        "blood_score": 0.0,
        "self_harm_score": 0.0,
        "promotion_score": 0.0,
        "advertising_score": 0.0,
        "affiliate_score": 0.0,
        "social_media_score": 0.0,
        "marketing_keyword_count": 0.0,
        "course_promotion_score": 0.0,
        "phone_number_score": 0.0,
        "social_handle_score": 0.0,
        "url_score": 0.0,
        "qr_code_score": 0.0,
        "terrorism_score": 0.0,
        "fraud_score": 0.0,
        "hate_speech_score": 0.0,
        "harassment_score": 0.0,
        "misinformation_score": 0.0,
        "self_harm_text_score": 0.0,
        "political_score": 0.0,
        "political_campaign_score": 0.0,
        "ml_toxicity_score": 0.0,
        "ml_hate_score": 0.0,
        "text_classifier_score": 0.0,
        "pii_score": 0.0,
        "llama_risk_score": 0.0,
        "llama_approves": 0.0,
        "ensemble_risk_score": 0.0,
        "uncertainty_score": 0.0,
    }
    scores.update(overrides)
    return scores


def _yolo_risk(detections: list[dict]) -> float:
    RISK_WEIGHTS = {"weapon": 0.90, "blood": 0.80, "fire": 0.40, "gun": 0.90, "knife": 0.70}
    score = 0.0
    for det in detections:
        conf = calibrate_yolo(float(det.get("confidence", 0.0)))
        w = RISK_WEIGHTS.get(str(det.get("class", "")).lower(), 0.0)
        score = max(score, w * conf)
    return min(1.0, score)


def _llama_to_risk(llama_result: dict | None) -> tuple[float, float, float]:
    """Return (llama_risk, llama_approves, llama_confidence)."""
    if not llama_result:
        return 0.5, 0.0, 0.5
    decision = str(llama_result.get("decision", "UNDER_REVIEW")).upper()
    confidence = max(0.0, min(1.0, float(llama_result.get("confidence", 0.5))))
    if decision == "REJECTED":
        return confidence, 0.0, confidence
    if decision == "APPROVED":
        return 1.0 - confidence, confidence, confidence
    return 0.5, 0.0, confidence


def _compute_pre_llm_risk(
    adult: float,
    child: float,
    violence: float,
    fraud: float,
    weapon: float,
    heritage: float,
) -> float:
    """Estimate ensemble risk before Llama runs, to gate secondary verification."""
    heritage_factor = max(0.0, min(1.0, heritage))
    return max(0.0, min(1.0,
        adult    * 0.30
        + child    * 0.25
        + violence * 0.18
        + fraud    * 0.12
        + weapon   * 0.10
        + (1.0 - heritage_factor) * 0.05
    ))


def _compute_ensemble(
    adult_score: float,
    child_score: float,
    violence_score: float,
    fraud_score: float,
    weapon_score: float,
    heritage_score: float,
    llama_result: dict | None = None,
) -> float:
    """Pure signal ensemble risk â€” Llama removed, weights redistributed to vision models."""
    heritage_factor = max(0.0, min(1.0, heritage_score))
    return max(0.0, min(1.0,
        adult_score    * 0.30
        + child_score    * 0.24
        + violence_score * 0.18
        + fraud_score    * 0.12
        + weapon_score   * 0.12
        + (1.0 - heritage_factor) * 0.04
    ))


def _apply_cultural_protection(
    child_scores: dict[str, float],
    violence_scores: dict[str, float],
    heritage_score: float,
    blip_captions: list[str],
    ocr_text: str,
    caption: str | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Reduce weapon/violence scores when cultural heritage is strongly indicated.

    Phase 5 child safety dominance: child scores are NEVER reduced â€” heritage
    context does not diminish child safety signals.

    Weapon/blood/violence: Ã— 0.70 when heritage_score > threshold or heritage
    keywords detected in combined text.
    """
    if heritage_score > _HERITAGE_PROTECT_THRESHOLD:
        is_heritage = True
    else:
        combined = " ".join([ocr_text or "", caption or "", *blip_captions]).lower()
        is_heritage = any(label in combined for label in HERITAGE_LABELS)

    if not is_heritage:
        return child_scores, violence_scores

    logger.info("Cultural protection applied (heritage_score=%.3f)", heritage_score)

    # Phase 5: Child safety dominance â€” pass through raw child scores unchanged.
    protected_child = dict(child_scores)

    protected_violence = {**violence_scores}
    protected_violence["weapon_score"] = violence_scores.get("weapon_score", 0.0) * 0.70
    protected_violence["blood_score"] = violence_scores.get("blood_score", 0.0) * 0.70
    protected_violence["violence_self_harm_score"] = (
        violence_scores.get("violence_self_harm_score", 0.0) * 0.70
    )
    protected_violence["self_harm_score"] = violence_scores.get("self_harm_score", 0.0) * 0.70

    return protected_child, protected_violence


def _safe_call(fn, breaker, timeout: float, model_name: str, fallback):
    """Run fn() through circuit breaker + timeout, returning fallback on any failure."""
    if breaker.is_open:
        logger.warning("Circuit open for '%s' â€” using fallback", model_name)
        return fallback
    try:
        return breaker.call(
            lambda: timeout_utils.timeout_call(fn, timeout=timeout, model_name=model_name)
        )
    except Exception:
        logger.exception("'%s' stage failed â€” using fallback", model_name)
        _metrics.model_errors_total.labels(model=model_name).inc()
        return fallback


def _detect_objects(image_path: str) -> list[dict]:
    """Thin wrapper for backward-compatibility with test mocks."""
    return object_detector.detect_objects(image_path)


def _check_gpu_memory() -> None:
    """Flush GPU caches if either GPU is over 90% utilization."""
    try:
        import torch
        for i in range(2):
            if torch.cuda.is_available() and i < torch.cuda.device_count():
                total = torch.cuda.get_device_properties(i).total_memory
                used = torch.cuda.memory_allocated(i)
                if total > 0 and (used / total) > 0.90:
                    logger.warning("GPU%d high memory (%.1f%%) â€” flushing cache", i, used / total * 100)
                    torch.cuda.empty_cache()
                    gc.collect()
    except Exception:
        pass


def _parallel_nsfw_yolo(image_path: str) -> tuple[float | None, list[dict]]:
    """Run NSFW and YOLO concurrently on GPU0, collecting results from both."""
    results: dict[str, Any] = {}

    def run_nsfw() -> None:
        results["nsfw"] = _safe_call(
            lambda: nsfw.get_adult_score(image_path),
            cb.nsfw_breaker, timeout_utils.TIMEOUTS["nsfw"], "nsfw", fallback=None,
        )

    def run_yolo() -> None:
        results["yolo"] = _safe_call(
            lambda: _detect_objects(image_path),
            cb.yolo_breaker, timeout_utils.TIMEOUTS["yolo"], "yolo", fallback=[],
        )

    t_nsfw = threading.Thread(target=run_nsfw, daemon=True)
    t_yolo = threading.Thread(target=run_yolo, daemon=True)
    t_nsfw.start()
    t_yolo.start()
    t_nsfw.join(timeout=timeout_utils.TIMEOUTS["nsfw"] + 2)
    t_yolo.join(timeout=timeout_utils.TIMEOUTS["yolo"] + 2)

    return results.get("nsfw"), results.get("yolo", [])


def analyze_image(
    image_path: str,
    caption: str | None = None,
) -> ModerationPipelineResult:
    """Run the full moderation pipeline. Never raises â€” errors become pipeline_error=True."""

    _check_gpu_memory()

    if not Path(image_path).is_file():
        return ModerationPipelineResult(
            scores=_default_scores(), category_scores={}, ocr_text="",
            pipeline_error=True, error_reason="Input image file does not exist.",
        )

    # â”€â”€ Stage 0: Image quality gate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ok, quality_reason = image_quality.check_image_quality(image_path)
    if not ok:
        return ModerationPipelineResult(
            scores=_default_scores(), category_scores={}, ocr_text="",
            pipeline_error=True,
            error_reason=f"Image quality check failed: {quality_reason}",
        )

    # â”€â”€ Stage 0b: Hash dedup cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cached = hash_cache.lookup(image_path)
    if cached:
        _metrics.hash_cache_hits_total.inc()
        logger.info("Returning cached moderation decision (hash=%s)", cached.get("image_hash"))
        return ModerationPipelineResult(
            scores=_default_scores(),
            category_scores={},
            ocr_text="",
            llama_result={
                "decision": cached["decision"],
                "reason": cached["reason"],
                "confidence": 1.0,
                "category": "Cached",
            },
            image_hash=cached.get("image_hash"),
            model_versions=MODEL_VERSIONS,
        )

    logger.info("Moderation pipeline started")

    # â”€â”€ Stage 1: Parallel NSFW + YOLO (GPU0) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    raw_adult, yolo_detections = _parallel_nsfw_yolo(image_path)

    if raw_adult is None:
        return ModerationPipelineResult(
            scores=_default_scores(), category_scores={}, ocr_text="",
            pipeline_error=True, error_reason="Critical NSFW model failed.",
        )
    adult_score = raw_adult
    yolo_risk = _yolo_risk(yolo_detections)
    detected_objects = [str(d.get("class", "")) for d in yolo_detections]

    # â”€â”€ Stage 2: Smart OCR (GPU0) â€” Surya primary + EasyOCR fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€
    should_run_ocr, _ocr_metrics = text_detector.detect_text_regions(
        image_path, yolo_detections
    )
    if should_run_ocr:
        ocr_text: str = _safe_call(
            lambda: ocr.extract_ocr_text(image_path),
            cb.ocr_breaker, timeout_utils.TIMEOUTS["ocr"], "ocr", fallback="",
        )
    else:
        logger.debug("Smart OCR skipped (no text signal detected)")
        ocr_text = ""
    content_quality_score: float = ocr.get_text_quality_score(ocr_text, caption)

    # â”€â”€ Stage 2b: Hard Block â€” zero-tolerance keyword scan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Runs before all remaining GPU inference to save compute.
    _hb_text = f"{ocr_text or ''} {caption or ''}".strip()
    if _hb_text:
        _hb_blocked, _hb_reason = hard_block.check(_hb_text)
        if _hb_blocked:
            logger.warning("Hard-block triggered: %s", _hb_reason)
            _metrics.moderation_decisions_total.labels(decision="REJECTED").inc()
            return ModerationPipelineResult(
                scores=_default_scores(content_quality_score=content_quality_score),
                category_scores={},
                ocr_text=ocr_text,
                llama_result={
                    "decision": "REJECTED",
                    "reason": f"Zero-tolerance content detected: {_hb_reason}",
                    "confidence": 1.0,
                    "category": "HARD_BLOCK",
                },
                model_versions=MODEL_VERSIONS,
            )

    # â”€â”€ Stage 2c: QR code detection + language detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    qr_result = qr_detector.analyze_qr(image_path)
    detected_language = language_detector.detect(ocr_text)
    logger.info(
        "QR detected=%s language=%s",
        qr_result["qr_code_detected"], detected_language,
    )

    # â”€â”€ Stage 2d: Text abuse classification (optional â€” MuRIL hook) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Runs only when text_classifier weights are present in models/muril_abuse_final/.
    # When disabled (weights absent) _tc_result.disabled=True and abuse_score=0.0
    # so this stage is a pure no-op that adds zero overhead when not configured.
    _tc_text = (ocr_text or "").strip()
    if _tc_text and text_classifier.is_available():
        _tc_result: dict = _safe_call(
            lambda: text_classifier.classify_text(_tc_text),
            cb.text_classifier_breaker,
            timeout_utils.TIMEOUTS["text_classifier"],
            "text_classifier",
            fallback=text_classifier._DISABLED_RESULT,
        )
    else:
        _tc_result = text_classifier._DISABLED_RESULT
    text_classifier_score: float = float(_tc_result.get("abuse_score", 0.0))
    if text_classifier_score > 0.0 and not _tc_result.get("disabled"):
        logger.info(
            "Text classifier: label=%s abuse_score=%.4f",
            _tc_result.get("label"), text_classifier_score,
        )

    # â”€â”€ Stage 3: SigLIP2 (GPU0) â€” uses OCR context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _run_siglip() -> clip_engine.ClipAnalysisResult:
        return clip_engine.analyze_content(image_path, caption, ocr_text)

    clip_result: clip_engine.ClipAnalysisResult | None = _safe_call(
        _run_siglip, cb.siglip_breaker, timeout_utils.TIMEOUTS["siglip"], "siglip", fallback=None,
    )
    if clip_result is None:
        return ModerationPipelineResult(
            scores=_default_scores(adult_score=adult_score, content_quality_score=content_quality_score),
            category_scores={}, ocr_text=ocr_text,
            pipeline_error=True, error_reason="Critical SigLIP2 model failed.",
        )
    heritage_score = clip_result.heritage_score
    category_scores = clip_result.category_scores

    # â”€â”€ Stage 3b: Embedding similarity search (FAISS) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    image_embedding = _safe_call(
        lambda: clip_engine.get_image_embedding(image_path),
        cb.embedding_breaker, timeout_utils.TIMEOUTS["embedding"], "embedding", fallback=None,
    )
    embedding_hit: dict | None = embedding_cache.search_similar_image(
        image_path, embedding=image_embedding
    )
    similar_images: list[dict] = [embedding_hit] if embedding_hit else []

    if embedding_hit:
        logger.info(
            "Embedding cache hit (similarity=%.4f) â€” reusing cached decision",
            embedding_hit.get("similarity", 0.0),
        )
        cached_scores = dict(embedding_hit.get("scores", {}))
        cached_scores.setdefault("uncertainty_score", 0.0)
        return ModerationPipelineResult(
            scores=_default_scores(**{k: float(v) for k, v in cached_scores.items() if k in _default_scores()}),
            category_scores=category_scores,
            ocr_text=ocr_text,
            llama_result={
                "decision": embedding_hit.get("decision", "UNDER_REVIEW"),
                "reason": "Decision reused from semantically similar image.",
                "confidence": float(embedding_hit.get("similarity", 0.93)),
                "category": "EmbeddingCache",
            },
            detected_objects=detected_objects,
            model_versions=MODEL_VERSIONS,
            similar_images=similar_images,
        )

    # â”€â”€ Stage 4: Signal fusion + cultural protection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    raw_child_scores = child_safety.analyze_child_safety(yolo_detections, clip_result.child_scores)
    raw_violence_scores = safety_detector.analyze_safety(yolo_detections, clip_result.safety_scores)

    child_scores, violence_scores = _apply_cultural_protection(
        raw_child_scores, raw_violence_scores,
        heritage_score, blip_captions=[], ocr_text=ocr_text, caption=caption,
    )

    # â”€â”€ Stage 5: Text / PII rules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    promotion_scores = promotion_detector.analyze_promotion(
        ocr_text, caption, clip_result.promotion_scores, yolo_detections,
        qr_decoded_text=qr_result.get("qr_decoded_text", ""),
    )
    text_scores = text_safety.analyze_text_safety(ocr_text, caption)
    pii_scores  = pii_detector.analyze_pii(ocr_text, caption)

    # â”€â”€ Stage 5b: ML Toxicity / Hate Speech (XLM-RoBERTa, GPU) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _tox_text = f"{ocr_text or ''} {caption or ''}".strip()
    ml_tox_scores: dict = _safe_call(
        lambda: ml_toxicity.analyze(_tox_text, None),
        cb.toxicity_breaker, 45.0, "ml_toxicity",
        fallback={"ml_toxicity_score": 0.0, "ml_hate_score": 0.0},
    )

    _LLAMA_FALLBACK: dict = {
        "decision": "UNDER_REVIEW",
        "reason": "Reasoning unavailable.",
        "confidence": 0.5,
        "category": "Uncategorized",
    }

    # â”€â”€ Stage 6: BLIP multi-caption (GPU1) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    blip_captions: list[str] = _safe_call(
        lambda: vlm_engine.generate_captions(image_path),
        cb.blip_breaker, timeout_utils.TIMEOUTS["blip"], "blip", fallback=[],
    )

    # Re-apply cultural protection now captions are available.
    if blip_captions:
        child_scores, violence_scores = _apply_cultural_protection(
            raw_child_scores, raw_violence_scores,
            heritage_score, blip_captions, ocr_text, caption,
        )

    # â”€â”€ Stage 7: Qwen2.5-VL description (GPU1) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _run_qwen_describe() -> dict:
        return qwen_vl.describe_image(image_path)

    qwen_result: dict = _safe_call(
        _run_qwen_describe,
        cb.qwen_breaker, timeout_utils.TIMEOUTS["qwen"], "qwen",
        fallback={"description": "", "confidence": 0.5},
    )
    qwen_description: str = qwen_result.get("description", "")
    qwen_confidence: float = float(qwen_result.get("confidence", 0.5))

    # â”€â”€ Stage 7b: Secondary verification for borderline content â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    pre_llm_risk = _compute_pre_llm_risk(
        adult=adult_score,
        child=child_scores["child_safety_score"],
        violence=violence_scores["violence_self_harm_score"],
        fraud=text_scores["fraud_score"],
        weapon=violence_scores["weapon_score"],
        heritage=heritage_score,
    )

    qwen_verification_reason: str = ""
    if _SECONDARY_VERIFY_LOW < pre_llm_risk < _SECONDARY_VERIFY_HIGH:
        def _run_qwen_verify() -> dict:
            return qwen_vl.verify_borderline(image_path, pre_llm_risk)

        verify_result: dict = _safe_call(
            _run_qwen_verify,
            cb.qwen_breaker, timeout_utils.TIMEOUTS["qwen"], "qwen",
            fallback={"verification_reason": "", "confidence": 0.5},
        )
        qwen_verification_reason = verify_result.get("verification_reason", "")
        logger.info(
            "Qwen secondary verification triggered (pre_llm_risk=%.3f)", pre_llm_risk
        )

    # â”€â”€ Stage 8: Llama reasoning (GPU1) â€” always runs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _run_llama() -> dict:
        return vlm_engine.reason_moderation(
            nsfw_score=adult_score,
            objects_detected=detected_objects,
            ocr_text=ocr_text,
            caption=caption or "",
            blip_captions=blip_captions,
            heritage_score=heritage_score,
            child_safety_score=child_scores["child_safety_score"],
            violence_score=violence_scores["violence_self_harm_score"],
            weapon_score=violence_scores["weapon_score"],
            qwen_description=qwen_description,
            qwen_verification=qwen_verification_reason,
        )

    llama_result: dict = _safe_call(
        _run_llama, cb.llama_breaker, timeout_utils.TIMEOUTS["llama"], "llama",
        fallback=_LLAMA_FALLBACK,
    )

    # â”€â”€ Stage 9: Uncertainty estimation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    uncertainty_score: float = uncertainty_module.compute_uncertainty(
        scores={
            "adult_score": adult_score,
            "child_safety_score": child_scores["child_safety_score"],
            "violence_self_harm_score": violence_scores["violence_self_harm_score"],
            "weapon_score": violence_scores["weapon_score"],
            "fraud_score": text_scores["fraud_score"],
            "terrorism_score": text_scores["terrorism_score"],
        },
        captions=blip_captions,
        llama_result=llama_result,
        qwen_description=qwen_description,
    )

    # â”€â”€ Stage 10: Ensemble (confidence-weighted Llama) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ensemble_risk = _compute_ensemble(
        adult_score=adult_score,
        child_score=child_scores["child_safety_score"],
        violence_score=violence_scores["violence_self_harm_score"],
        fraud_score=text_scores["fraud_score"],
        weapon_score=violence_scores["weapon_score"],
        heritage_score=heritage_score,
        llama_result=llama_result,
    )
    llama_risk, llama_approves, _ = _llama_to_risk(llama_result)

    # â”€â”€ Stage 11: Hash store + Embedding store â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    llama_decision = llama_result.get("decision", "UNDER_REVIEW")
    image_hash = hash_cache.store(
        image_path, llama_decision, llama_result.get("reason", ""),
        extra={"ensemble_risk": ensemble_risk},
    )

    final_scores = _default_scores(
        adult_score=adult_score,
        heritage_score=heritage_score,
        content_quality_score=content_quality_score,
        child_safety_score=child_scores["child_safety_score"],
        child_presence_score=child_scores["child_presence_score"],
        violence_self_harm_score=violence_scores["violence_self_harm_score"],
        weapon_score=violence_scores["weapon_score"],
        blood_score=violence_scores["blood_score"],
        self_harm_score=violence_scores["self_harm_score"],
        promotion_score=promotion_scores["promotion_score"],
        advertising_score=promotion_scores["advertising_score"],
        affiliate_score=promotion_scores["affiliate_score"],
        social_media_score=promotion_scores["social_media_score"],
        marketing_keyword_count=promotion_scores.get("marketing_keyword_count", 0.0),
        course_promotion_score=promotion_scores.get("course_promotion_score", 0.0),
        phone_number_score=promotion_scores.get("phone_number_score", 0.0),
        social_handle_score=promotion_scores.get("social_handle_score", 0.0),
        url_score=promotion_scores.get("url_score", 0.0),
        qr_code_score=qr_result.get("qr_code_score", 0.0),
        terrorism_score=text_scores["terrorism_score"],
        fraud_score=text_scores["fraud_score"],
        hate_speech_score=text_scores["hate_speech_score"],
        harassment_score=text_scores["harassment_score"],
        misinformation_score=text_scores["misinformation_score"],
        self_harm_text_score=text_scores["self_harm_text_score"],
        political_score=text_scores.get("political_score", 0.0),
        political_campaign_score=text_scores.get("political_campaign_score", 0.0),
        ml_toxicity_score=ml_tox_scores.get("ml_toxicity_score", 0.0),
        ml_hate_score=ml_tox_scores.get("ml_hate_score", 0.0),
        text_classifier_score=text_classifier_score,
        pii_score=pii_scores["pii_score"],
        llama_risk_score=llama_risk,
        llama_approves=llama_approves,
        ensemble_risk_score=ensemble_risk,
        uncertainty_score=uncertainty_score,
    )

    embedding_cache.store_image(
        image_path=image_path,
        embedding=image_embedding,
        decision=llama_decision,
        scores=final_scores,
        evidence={
            "qwen_description": qwen_description,
            "qwen_confidence": qwen_confidence,
        },
    )

    generated_caption = blip_captions[0] if blip_captions else ""
    logger.info(
        "Moderation pipeline completed (ensemble_risk=%.3f, uncertainty=%.3f)",
        ensemble_risk, uncertainty_score,
    )

    return ModerationPipelineResult(
        scores=final_scores,
        category_scores=category_scores,
        ocr_text=ocr_text,
        llama_result=llama_result,
        detected_objects=detected_objects,
        generated_caption=generated_caption,
        generated_captions=blip_captions,
        model_versions=MODEL_VERSIONS,
        image_hash=image_hash,
        qwen_description=qwen_description,
        qwen_confidence=qwen_confidence,
        uncertainty_score=uncertainty_score,
        similar_images=similar_images,
    )


def get_mock_scores() -> dict[str, float]:
    return _default_scores(adult_score=0.05, heritage_score=0.85, content_quality_score=0.1)


def get_mock_category_scores() -> dict[str, float]:
    return {
        "Religious & Spiritual Heritage": 0.9,
        "Historical Monuments & Events": 0.05,
        "Arts, Culture & Festivals": 0.02,
        "Education & Documentation": 0.01,
        "Community & Personal Stories": 0.01,
    }


def get_mock_ocr_text() -> str:
    return ""

