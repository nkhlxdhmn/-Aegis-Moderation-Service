# Integration Guide: Adding MuRIL to Your Existing OCR Pipeline

This shows how to add the MuRIL abuse classifier to your current Surya OCR pipeline in ocr.py

## Step 1: Update ocr.py Imports

Add these imports at the top of your existing `pipeline/ocr.py`:

```python
from pathlib import Path
from transformers import pipeline as hf_pipeline
import torch

# MuRIL classifier globals (load once, reuse)
MURIL_MODEL_PATH = None
MURIL_CLASSIFIER = None

def _init_muril_classifier():
    """Lazy-load MuRIL classifier on first use."""
    global MURIL_CLASSIFIER, MURIL_MODEL_PATH
    
    if MURIL_CLASSIFIER is not None:
        return  # Already loaded
    
    MURIL_MODEL_PATH = Path(__file__).parent.parent / "models" / "muril_abuse_final"
    
    if not MURIL_MODEL_PATH.exists():
        raise RuntimeError(
            f"MuRIL model not found at {MURIL_MODEL_PATH}\n"
            "Train first: python moderation-service/muril_local_train.py"
        )
    
    MURIL_CLASSIFIER = hf_pipeline(
        task="text-classification",
        model=str(MURIL_MODEL_PATH),
        tokenizer=str(MURIL_MODEL_PATH),
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=128,
    )
    talker.info("🚀 MuRIL classifier initialized")
```

## Step 2: Add Abuse Detection Function

Add this function to your ocr.py after the existing OCR functions:

```python
def detect_abusive_content(ocr_lines: list[str], classifier) -> dict:
    """
    Detect abusive/harmful content in OCR-extracted text lines.
    
    Args:
        ocr_lines: List of text strings from OCR output
        classifier: MuRIL pipeline instance
    
    Returns:
        {
          "flagged": bool,
          "abuse_count": int,
          "abuse_ratio": float,
          "total_lines": int,
          "details": [
            {"text": str, "label": str, "score": float, "flagged": bool}
          ]
        }
    """
    if not ocr_lines:
        return {
            "flagged": False,
            "abuse_count": 0,
            "abuse_ratio": 0.0,
            "total_lines": 0,
            "details": [],
        }
    
    # Filter empty lines
    valid_lines = [line.strip() for line in ocr_lines if line.strip() and len(line.strip()) > 3]
    
    if not valid_lines:
        return {
            "flagged": False,
            "abuse_count": 0,
            "abuse_ratio": 0.0,
            "total_lines": 0,
            "details": [],
        }
    
    try:
        # Batch inference (32 = good balance for GPU)
        predictions = classifier(valid_lines, batch_size=32)
        
        details = []
        abuse_count = 0
        
        for text, pred in zip(valid_lines, predictions):
            # Threshold: confidence must be >75% to flag as abusive
            is_flagged = (pred["label"] == "abusive" and pred["score"] >= 0.75)
            
            if is_flagged:
                abuse_count += 1
            
            details.append({
                "text": text,
                "label": pred["label"],
                "score": round(pred["score"], 4),
                "flagged": is_flagged,
            })
        
        abuse_ratio = abuse_count / len(valid_lines) if valid_lines else 0
        
        # Flag image if >20% of lines are abusive
        image_flagged = abuse_ratio >= 0.20
        
        return {
            "flagged": image_flagged,
            "abuse_count": abuse_count,
            "abuse_ratio": round(abuse_ratio, 3),
            "total_lines": len(valid_lines),
            "details": details,
        }
        
    except Exception as e:
        talker.error(f"MuRIL classification error: {e}")
        return {
            "flagged": False,
            "abuse_count": 0,
            "abuse_ratio": 0.0,
            "total_lines": len(valid_lines),
            "details": [],
            "error": str(e),
        }
```

## Step 3: Modify Existing OCR Processing Function

Find your current function that processes Surya OCR output. Update it to include abuse detection:

**Before:**
```python
async def process_image_ocr(image_path: str) -> dict:
    """Process image with Surya OCR."""
    
    # ... existing Surya OCR code ...
    ocr_result = reader.ocr(image)
    
    extracted_text = "\n".join([line.text for line in ocr_result.lines])
    
    return {
        "status": "success",
        "text": extracted_text,
        "lines": ocr_result.lines,
    }
```

**After:**
```python
async def process_image_ocr(image_path: str) -> dict:
    """Process image with Surya OCR + abuse detection."""
    
    # Initialize classifier if needed
    _init_muril_classifier()
    
    # ... existing Surya OCR code ...
    ocr_result = reader.ocr(image)
    
    extracted_text = "\n".join([line.text for line in ocr_result.lines])
    ocr_lines = [line.text for line in ocr_result.lines]
    
    # NEW: Detect abuse in extracted text
    abuse_result = detect_abusive_content(ocr_lines, MURIL_CLASSIFIER)
    
    return {
        "status": "success",
        "text": extracted_text,
        "lines": ocr_result.lines,
        "abuse_detection": abuse_result,           # NEW
        "is_abusive": abuse_result["flagged"],     # NEW
        "abuse_ratio": abuse_result["abuse_ratio"], # NEW
    }
```

## Step 4: Update Your Moderation Decision Logic

Modify your content moderation decision based on abuse detection:

```python
def get_moderation_decision(ocr_result: dict) -> dict:
    """
    Make moderation decision based on OCR + abuse detection.
    
    Decision:
      REJECT  : >30% abusive lines
      REVIEW  : 15-30% abusive lines (manual review)
      APPROVE : <15% abusive lines
    """
    abuse_ratio = ocr_result.get("abuse_ratio", 0)
    abuse_details = ocr_result.get("abuse_detection", {})
    
    # Decision thresholds
    if abuse_ratio > 0.30:
        decision = "REJECT"
        reason = f"High abuse content ({abuse_ratio:.0%} of lines)"
    elif abuse_ratio > 0.15:
        decision = "REVIEW"
        reason = f"Potential abuse content ({abuse_ratio:.0%} of lines)"
    else:
        decision = "APPROVE"
        reason = "Content appears safe"
    
    return {
        "decision": decision,
        "reason": reason,
        "abuse_ratio": abuse_ratio,
        "abusive_lines": [
            d for d in abuse_details.get("details", []) if d["flagged"]
        ],
        "confidence": abuse_details.get("details", []),
    }
```

## Step 5: Add to Your API Endpoints

If you expose OCR via API, add abuse detection flag:

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/api/v1/ocr")
async def ocr_endpoint(image_url: str):
    """OCR endpoint with abuse detection."""
    
    try:
        # Process image
        ocr_result = await process_image_ocr(image_url)
        
        # Get moderation decision
        decision = get_moderation_decision(ocr_result)
        
        return JSONResponse({
            "status": "success",
            "ocr": {
                "text": ocr_result["text"],
                "line_count": len(ocr_result.get("lines", [])),
            },
            "moderation": {
                "decision": decision["decision"],
                "reason": decision["reason"],
                "abuse_ratio": decision["abuse_ratio"],
                "abusive_lines_count": len(decision["abusive_lines"]),
            },
            "details": {
                "abuse_detection": ocr_result.get("abuse_detection"),
            },
        })
    
    except Exception as e:
        return JSONResponse(
            {"error": str(e), "status": "error"},
            status_code=400,
        )
```

## Step 6: Error Handling & Fallback

Add robust error handling:

```python
def process_image_with_fallback(image_path: str) -> dict:
    """
    Process image with OCR + abuse detection, with graceful fallback.
    
    If MuRIL fails, continues with OCR results only.
    """
    try:
        result = process_image_ocr(image_path)
    except Exception as e:
        talker.error(f"OCR processing failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "ocr": None,
            "abuse_detection": None,
        }
    
    # If abuse detection failed, still return OCR
    if result.get("abuse_detection", {}).get("error"):
        talker.warning("Abuse detection failed, proceeding without it")
        result["is_abusive"] = None  # Unknown
        result["abuse_ratio"] = None
    
    return result
```

## Quick Checklist

- [ ] Install MuRIL training dependencies: `pip install -r requirements-muril-training.txt`
- [ ] Train model: `python moderation-service/muril_local_train.py`
- [ ] Update `pipeline/ocr.py` with imports and functions above
- [ ] Test with `examples_muril_integration.py`
- [ ] Deploy trained model to production
- [ ] Monitor abuse detection accuracy in production

## Performance Notes

**Expected Performance:**
- OCR only: ~100-200 ms per image (Surya)
- + Abuse detection: ~500-800 ms per image (adds MuRIL inference)
- Batch processing: ~50 ms per image (amortized)

**Optimization Tips:**
1. **Batch processing**: Process multiple images together
2. **Async/threading**: Run OCR and abuse detection in parallel
3. **Caching**: Cache results for identical images
4. **GPU**: Ensure model runs on GPU (check CUDA_VISIBLE_DEVICES)

## Testing Integration

```python
# Quick test before deployment
from examples_muril_integration import load_muril_classifier, classify_text_line

classifier = load_muril_classifier()

test_cases = [
    ("aaj mausam accha hai", False),           # Non-abusive
    ("tujhe toh main dekh lunga", True),       # Abusive
    ("nice movie rating", False),              # Non-abusive
    ("teri ma ki aankh", True),               # Abusive
]

for text, expected_abusive in test_cases:
    result = classify_text_line(text, classifier)
    is_correct = result["flagged"] == expected_abusive
    status = "✅" if is_correct else "❌"
    print(f"{status} {text:<40} → {result['label']:<12} ({result['score']:.0%})")
```

## Rollback Plan

If abuse detection causes issues in production:

1. Comment out the abuse detection call in `process_image_ocr()`
2. Keep OCR functionality running
3. Investigate issue in separate environment
4. Deploy fix

```python
# Temporary disable (add these lines)
# abuse_result = detect_abusive_content(ocr_lines, MURIL_CLASSIFIER)
# Set to empty result instead
abuse_result = {
    "flagged": False,
    "abuse_count": 0,
    "abuse_ratio": 0.0,
    "total_lines": 0,
    "details": [],
}
```

---

Need help? Check:
- [MURIL_QUICK_REFERENCE.md](MURIL_QUICK_REFERENCE.md)
- [MURIL_LOCAL_TRAINING.md](MURIL_LOCAL_TRAINING.md)
- [examples_muril_integration.py](examples_muril_integration.py)
