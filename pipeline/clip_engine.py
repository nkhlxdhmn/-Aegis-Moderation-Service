"""SigLIP2 Large heritage understanding and safety scoring.

Replaces the previous OpenCLIP ViT-B/32 engine.

Key improvements over CLIP ViT-B/32:
  - SigLIP uses sigmoid scoring (each image-text pair is independent).
    Unlike CLIP softmax, this produces calibrated probabilities instead of
    relative rankings, which eliminates the (cosine+1)/2 normalization bug.
  - ViT-L/16 at 384px → 16px patch size gives 4× better spatial detail than B/32.
  - Contextualized prompts replace single-word triggers to prevent false positives
    on heritage imagery (temple spires ≠ "knife", carved figures ≠ "child").
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

MODEL_ID = "google/siglip2-large-patch16-384"
SIGLIP_MODEL_ID = MODEL_ID   # alias for external callers / test discovery
DEVICE = "cuda:0"

# ── Category prompts ──────────────────────────────────────────────────────────
CATEGORY_PROMPTS: dict[str, tuple[str, ...]] = {
    "Religious & Spiritual Heritage": (
        "a Hindu temple with gopuram tower and sculptures",
        "deity murti or idol inside a shrine",
        "religious puja ceremony with offerings and lamps",
        "ancient stone temple architecture in India",
        "Buddhist stupa or Jain temple interior",
    ),
    "Historical Monuments & Events": (
        "a historical fort or palace in India",
        "Mughal or colonial era heritage architecture",
        "archaeological ruins or ancient site in India",
        "a historical battle or military event depicted in art",
        "a heritage building or monument with plaques",
    ),
    "Arts, Culture & Festivals": (
        "Indian classical or folk dance performance",
        "Diwali or Holi festival celebration with lights and colors",
        "traditional Indian art painting or folk craft",
        "a cultural procession or community celebration",
        "rangoli pattern or kolam artwork",
    ),
    "Education & Documentation": (
        "a Sanskrit or palm leaf manuscript",
        "historical inscription or stone carving with text",
        "an educational exhibit or museum display about India",
        "a documentary photograph of cultural practices",
        "archival document or ancient script",
    ),
    "Community & Personal Stories": (
        "a family gathering for a traditional ceremony",
        "village community event or local festival",
        "personal photograph at a pilgrimage site",
        "oral history or storytelling scene",
        "a wedding or cultural rite of passage",
    ),
}

# Heritage detection prompts — broad coverage of Indian cultural contexts
HERITAGE_PROMPTS: tuple[str, ...] = (
    "a Hindu temple with gopuram and sculptural decoration",
    "ancient stone temple architecture in India",
    "deity sculpture or murti at a shrine or museum",
    "religious puja ceremony with diya lamps and flowers",
    "Buddhist stupa or monastery in India",
    "Jain temple ornate carved marble interior",
    "mosque or dargah with Islamic architecture in India",
    "historical fort walls and palace in India",
    "archaeological ruins at an Indian heritage site",
    "Sanskrit manuscript or palm leaf inscription",
    "Indian classical Bharatanatyam or Kathak dance performance",
    "traditional Indian festival celebration with diyas or rangoli",
    "tribal or folk art painting from India",
    "Mughal miniature painting or historical artwork",
)

# Safety prompts — contextualized to avoid single-word false positives
SAFETY_PROMPTS: tuple[str, ...] = (
    "a person being physically attacked or assaulted",
    "a gun or firearm being used to threaten or shoot someone",
    "a knife or sharp weapon being used to harm a person",
    "graphic blood and gore from a real injury or violence",
    "a real dead body or human remains from a crime or accident",
    "explicit real-world violence or murder",
    "a person cutting themselves or engaging in self-harm",
    "content depicting a suicide attempt",
    "explicit pornographic sexual content between adults",
    "adult sexual nudity in an explicit sexual context",
)

# Child safety prompts — requires harmful intent context, not mere child presence
CHILD_PROMPTS: tuple[str, ...] = (
    "child sexual exploitation or abuse material",
    "a child in a dangerous, harmful, or threatening situation",
    "inappropriate sexual content involving a minor",
    "a child being abused or harmed by an adult",
)

# Promotion prompts
PROMOTION_PROMPTS: tuple[str, ...] = (
    "sponsored advertisement or brand promotion post",
    "affiliate marketing or referral code promotion",
    "social media growth scheme like subscribe for subscribe",
    "telegram or whatsapp group spam promotion",
    "online course selling or recruitment advertisement",
    "influencer paid partnership or brand deal disclosure",
    "discount sale or promotional offer advertisement",
)


class ModelInferenceError(RuntimeError):
    """Raised when SigLIP2 cannot process an image."""


@dataclass
class _SigLIPState:
    model: Any
    processor: Any
    torch: Any
    device: str
    # Pre-encoded text feature tensors for reuse across requests
    category_inputs: dict[str, Any]       # {category: processor_output}
    heritage_inputs: Any
    safety_inputs: Any
    child_inputs: Any
    promotion_inputs: Any


@dataclass(frozen=True)
class ClipAnalysisResult:
    """SigLIP2 category and heritage signals for one moderation image."""

    category_scores: dict[str, float]
    heritage_score: float
    safety_scores: dict[str, float] = field(default_factory=dict)
    child_scores: dict[str, float] = field(default_factory=dict)
    promotion_scores: dict[str, float] = field(default_factory=dict)


_state: _SigLIPState | None = None
_state_lock = threading.Lock()


def _encode_texts(prompts: tuple[str, ...], processor: Any, device: str) -> Any:
    """Pre-encode text prompts; returned tensor stays on CPU to save VRAM."""
    return processor(
        text=list(prompts),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


def _get_state() -> _SigLIPState:
    global _state
    if _state is not None:
        return _state

    with _state_lock:
        if _state is not None:
            return _state

        logger.info("Loading SigLIP2 Large on %s", DEVICE)
        try:
            import torch
            from transformers import AutoProcessor, AutoModel

            processor = AutoProcessor.from_pretrained(MODEL_ID)
            model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(DEVICE)
            model.eval()
            torch.backends.cudnn.benchmark = True

            # Pre-encode all static prompt sets (kept on CPU, moved to GPU per request)
            category_inputs = {
                cat: _encode_texts(prompts, processor, DEVICE)
                for cat, prompts in CATEGORY_PROMPTS.items()
            }
            heritage_inputs = _encode_texts(HERITAGE_PROMPTS, processor, DEVICE)
            safety_inputs = _encode_texts(SAFETY_PROMPTS, processor, DEVICE)
            child_inputs = _encode_texts(CHILD_PROMPTS, processor, DEVICE)
            promotion_inputs = _encode_texts(PROMOTION_PROMPTS, processor, DEVICE)

            local = _SigLIPState(
                model=model,
                processor=processor,
                torch=torch,
                device=DEVICE,
                category_inputs=category_inputs,
                heritage_inputs=heritage_inputs,
                safety_inputs=safety_inputs,
                child_inputs=child_inputs,
                promotion_inputs=promotion_inputs,
            )
            _state = local  # publish only after full init (double-checked locking)
        except Exception as exc:
            logger.exception("Failed to load SigLIP2")
            raise ModelInferenceError("SigLIP2 failed to load") from exc

        logger.info("SigLIP2 Large loaded on %s", DEVICE)
    return _state


def _encode_image(image_path: str, state: _SigLIPState) -> Any:
    """Return pixel_values tensor for one image."""
    from PIL import Image

    with Image.open(image_path) as img:
        pixel_values = state.processor(
            images=img.convert("RGB"),
            return_tensors="pt",
        ).pixel_values
    return pixel_values.to(state.device, dtype=state.torch.float16)


def _sigmoid_scores(
    pixel_values: Any,
    text_inputs: Any,
    state: _SigLIPState,
    prompts: tuple[str, ...],
) -> dict[str, float]:
    """Score image against every prompt using SigLIP2 sigmoid (not softmax).

    SigLIP sigmoid gives an independent probability for each (image, text) pair.
    Scores near 0 = unrelated; near 1 = semantically matched.
    Typical noise floor for unrelated pairs: 0.01–0.08.
    """
    torch = state.torch
    # Move text tensors to GPU for this call
    t_inputs = {k: v.to(state.device) for k, v in text_inputs.items()}

    with torch.inference_mode():
        outputs = state.model(pixel_values=pixel_values, **t_inputs)
        # logits_per_image shape: [1, n_texts] when pixel_values is a single image
        # squeeze to [n_texts] then apply sigmoid for per-pair probabilities
        logits = outputs.logits_per_image.squeeze(0)  # [n_texts]
        probs = torch.sigmoid(logits).cpu().float().tolist()

    if isinstance(probs, float):
        probs = [probs]

    return {prompt: max(0.0, min(1.0, float(p))) for prompt, p in zip(prompts, probs)}


def _max_category_score(scores_per_prompt: dict[str, float]) -> float:
    return max(scores_per_prompt.values(), default=0.0)


def _fuse_category_scores(
    image_scores: dict[str, float],
    ocr_scores: dict[str, float],
    caption_scores: dict[str, float],
) -> dict[str, float]:
    return {
        cat: max(0.0, min(1.0,
            image_scores[cat] * 0.50
            + ocr_scores[cat] * 0.25
            + caption_scores[cat] * 0.25,
        ))
        for cat in CATEGORY_PROMPTS
    }


def _empty_category_scores() -> dict[str, float]:
    return {cat: 0.0 for cat in CATEGORY_PROMPTS}


def _encode_text_query(text: str, state: _SigLIPState) -> Any | None:
    text = text.strip()
    if not text:
        return None
    return state.processor(
        text=[text],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


def _category_scores_from_pv(pixel_values: Any, state: _SigLIPState) -> dict[str, float]:
    return {
        cat: _max_category_score(
            _sigmoid_scores(pixel_values, text_inputs, state, CATEGORY_PROMPTS[cat])
        )
        for cat, text_inputs in state.category_inputs.items()
    }


def _category_scores_from_text(text_inputs: Any, pixel_values: Any, state: _SigLIPState) -> dict[str, float]:
    return {
        cat: _max_category_score(
            _sigmoid_scores(pixel_values, text_inputs_cat, state, CATEGORY_PROMPTS[cat])
        )
        for cat, text_inputs_cat in state.category_inputs.items()
    }


def analyze_content(
    image_path: str,
    caption: str | None,
    ocr_text: str,
) -> ClipAnalysisResult:
    """Return SigLIP2 category, heritage, and safety signals for an image."""

    logger.info("SigLIP2 analysis started")
    try:
        state = _get_state()
        pixel_values = _encode_image(image_path, state)

        # Image → category scores
        image_cat_scores = _category_scores_from_pv(pixel_values, state)

        # OCR text → category scores
        ocr_inputs = _encode_text_query(ocr_text or "", state)
        if ocr_inputs is not None:
            ocr_cat_scores = _category_scores_from_text(ocr_inputs, pixel_values, state)
        else:
            ocr_cat_scores = _empty_category_scores()

        # Caption → category scores
        cap_inputs = _encode_text_query(caption or "", state)
        if cap_inputs is not None:
            cap_cat_scores = _category_scores_from_text(cap_inputs, pixel_values, state)
        else:
            cap_cat_scores = _empty_category_scores()

        category_scores = _fuse_category_scores(image_cat_scores, ocr_cat_scores, cap_cat_scores)

        # Heritage score: max over heritage prompts + best category
        heritage_prompt_scores = _sigmoid_scores(
            pixel_values, state.heritage_inputs, state, HERITAGE_PROMPTS
        )
        heritage_score = max(
            max(heritage_prompt_scores.values(), default=0.0),
            max(category_scores.values(), default=0.0),
        )

        # Safety scores (contextualized — very low noise floor with SigLIP sigmoid)
        safety_scores = _sigmoid_scores(
            pixel_values, state.safety_inputs, state, SAFETY_PROMPTS
        )

        # Child safety scores
        child_scores = _sigmoid_scores(
            pixel_values, state.child_inputs, state, CHILD_PROMPTS
        )

        # Promotion scores
        promotion_scores = _sigmoid_scores(
            pixel_values, state.promotion_inputs, state, PROMOTION_PROMPTS
        )

        if state.torch.cuda.is_available():
            state.torch.cuda.empty_cache()

    except Exception as exc:
        logger.exception("SigLIP2 analysis failed")
        if isinstance(exc, ModelInferenceError):
            raise
        raise ModelInferenceError("SigLIP2 analysis failed") from exc

    logger.info("SigLIP2 analysis completed")
    return ClipAnalysisResult(
        category_scores=category_scores,
        heritage_score=max(0.0, min(1.0, heritage_score)),
        safety_scores=safety_scores,
        child_scores=child_scores,
        promotion_scores=promotion_scores,
    )


def get_category_scores(
    image_path: str,
    caption: str | None,
    ocr_text: str,
) -> dict[str, float]:
    return analyze_content(image_path, caption, ocr_text).category_scores


def get_heritage_score(
    image_path: str,
    caption: str | None,
    ocr_text: str,
) -> float:
    return max(0.0, min(1.0, analyze_content(image_path, caption, ocr_text).heritage_score))


def get_image_embedding(image_path: str) -> "Any | None":
    """Return a normalised float32 SigLIP2 vision embedding for FAISS search.

    Uses the already-loaded SigLIP2 model singleton.  The vision encoder's
    pooler output is extracted and L2-normalised so cosine similarity can be
    computed as an inner product.

    Returns None on any failure (model not loaded, image unreadable, etc.) so
    callers can treat a missing embedding as a cache miss and continue normally.
    """
    try:
        import numpy as np

        state = _get_state()
        pixel_values = _encode_image(image_path, state)

        with state.torch.inference_mode():
            vision_out = state.model.vision_model(pixel_values=pixel_values)
            emb = vision_out.pooler_output  # [1, hidden_dim]

        emb_np = emb.squeeze(0).float().cpu().numpy().astype("float32")
        norm = float(np.linalg.norm(emb_np))
        if norm > 1e-8:
            emb_np = emb_np / norm
        return emb_np
    except Exception:
        logger.debug("get_image_embedding failed for '%s'", image_path)
        return None
