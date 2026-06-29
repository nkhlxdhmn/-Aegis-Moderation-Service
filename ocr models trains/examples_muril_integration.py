# Example: Using MuRIL Classifier with Surya OCR Pipeline
# This shows how to integrate the trained classifier into your moderation pipeline

from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, pipeline

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: Load Pre-trained MuRIL Classifier
# ─────────────────────────────────────────────────────────────────────────────

def load_muril_classifier():
    """Load fine-tuned MuRIL model for abuse detection."""
    model_path = Path("./models/muril_abuse_final")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            "Train first: python muril_local_train.py"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    classifier = pipeline(
        task="text-classification",
        model=str(model_path),
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=128,
    )
    
    return classifier


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: Moderate Single Line of Text
# ─────────────────────────────────────────────────────────────────────────────

def classify_text_line(text: str, classifier, threshold: float = 0.75) -> Dict[str, Any]:
    """
    Classify a single line of text.
    
    Returns:
        {
          "text": str,
          "label": "abusive" | "non_abusive",
          "score": float,
          "flagged": bool (True if abusive AND confidence > threshold)
        }
    """
    result = classifier(text)[0]
    
    is_flagged = (result["label"] == "abusive" and result["score"] >= threshold)
    
    return {
        "text": text,
        "label": result["label"],
        "score": round(result["score"], 4),
        "flagged": is_flagged,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3: Moderate Multiple OCR Lines (Batch)
# ─────────────────────────────────────────────────────────────────────────────

def moderate_ocr_lines(
    ocr_lines: List[str],
    classifier,
    abuse_threshold: float = 0.75,
    image_flag_ratio: float = 0.20,
) -> Dict[str, Any]:
    """
    Classify multiple OCR-extracted text lines.
    
    Flags image if percentage of abusive lines exceeds image_flag_ratio.
    
    Args:
        ocr_lines: List of text strings from OCR (one per line)
        classifier: Loaded MuRIL pipeline
        abuse_threshold: Confidence needed to flag a line as abusive
        image_flag_ratio: Fraction of abusive lines to flag entire image
    
    Returns:
        {
          "image_flagged": bool,
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
            "image_flagged": False,
            "abuse_count": 0,
            "abuse_ratio": 0.0,
            "total_lines": 0,
            "details": [],
        }
    
    # Filter empty lines
    valid_lines = [line.strip() for line in ocr_lines if line.strip()]
    if not valid_lines:
        return {
            "image_flagged": False,
            "abuse_count": 0,
            "abuse_ratio": 0.0,
            "total_lines": 0,
            "details": [],
        }
    
    # Batch classify
    predictions = classifier(valid_lines, batch_size=32)
    
    details = []
    abuse_count = 0
    
    for text, pred in zip(valid_lines, predictions):
        is_flagged = (pred["label"] == "abusive" and pred["score"] >= abuse_threshold)
        
        if is_flagged:
            abuse_count += 1
        
        details.append({
            "text": text,
            "label": pred["label"],
            "score": round(pred["score"], 4),
            "flagged": is_flagged,
        })
    
    abuse_ratio = abuse_count / len(valid_lines)
    image_flagged = abuse_ratio >= image_flag_ratio
    
    return {
        "image_flagged": image_flagged,
        "abuse_count": abuse_count,
        "abuse_ratio": round(abuse_ratio, 3),
        "total_lines": len(valid_lines),
        "details": details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 4: Integration with Surya OCR Results
# ─────────────────────────────────────────────────────────────────────────────

def process_surya_ocr_with_abuse_detection(
    surya_result: Dict[str, Any],
    classifier,
) -> Dict[str, Any]:
    """
    Take Surya OCR output and add abuse classification.
    
    Assumes surya_result structure:
        {
          "text": str (full concatenated text),
          "lines": [
            {"text": str, "bbox": [...], "confidence": float}
          ]
        }
    """
    # Extract text lines from Surya output
    ocr_lines = [line.get("text", "") for line in surya_result.get("lines", [])]
    
    # Run abuse detection
    abuse_detection = moderate_ocr_lines(ocr_lines, classifier)
    
    # Combine results
    return {
        "original_text": surya_result.get("text", ""),
        "ocr_lines": ocr_lines,
        "abuse_detection": abuse_detection,
        "is_flagged": abuse_detection["image_flagged"],
        "confidence_scores": [d["score"] for d in abuse_detection["details"]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 5: Full Pipeline (Surya OCR → MuRIL Classification)
# ─────────────────────────────────────────────────────────────────────────────

def moderate_image_from_surya(
    surya_ocr_result: Dict[str, Any],
    classifier,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Complete moderation pipeline: OCR result → abuse classification.
    
    Returns decision: APPROVE, REVIEW, or REJECT
    """
    result = process_surya_ocr_with_abuse_detection(surya_ocr_result, classifier)
    
    abuse_ratio = result["abuse_detection"]["abuse_ratio"]
    
    # Decision logic
    if abuse_ratio > 0.3:
        decision = "REJECT"  # >30% abusive
    elif abuse_ratio > 0.15:
        decision = "REVIEW"  # 15-30% (manual review)
    else:
        decision = "APPROVE" # <15% (safe)
    
    result["moderation_decision"] = decision
    
    if verbose:
        print(f"\n🔍 Moderation Result:")
        print(f"   Decision: {decision}")
        print(f"   Abuse ratio: {abuse_ratio:.1%}")
        print(f"   Lines flagged: {result['abuse_detection']['abuse_count']} / {result['abuse_detection']['total_lines']}")
        print(f"\n   Details:")
        for detail in result["abuse_detection"]["details"]:
            flag = "🚨" if detail["flagged"] else "✅"
            print(f"   {flag} [{detail['label']:<12} {detail['score']:.0%}] {detail['text'][:60]}")
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TEST EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*80)
    print("MuRIL Abuse Classifier - Integration Examples")
    print("="*80)
    
    # Load classifier
    print("\n📥 Loading classifier...")
    try:
        classifier = load_muril_classifier()
        print("✅ Classifier loaded")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        exit(1)
    
    # Example 1: Single line
    print("\n" + "─"*80)
    print("Example 1: Single Line Classification")
    print("─"*80)
    
    test_line = "aaj mausam bahut accha hai"
    result = classify_text_line(test_line, classifier)
    print(f"Text: '{test_line}'")
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['score']:.0%}")
    print(f"Flagged: {result['flagged']}")
    
    # Example 2: Batch classification
    print("\n" + "─"*80)
    print("Example 2: Batch Classification (Multiple Lines)")
    print("─"*80)
    
    test_lines = [
        "aaj mausam accha hai",
        "nice movie, bahut achi thi",
        "tujhe toh main dekh lunga",
        "teri ma ki aankh",
        "coffee peena chahta hoon kya?",
    ]
    
    result = moderate_ocr_lines(test_lines, classifier)
    print(f"Total lines: {result['total_lines']}")
    print(f"Abusive lines: {result['abuse_count']}")
    print(f"Abuse ratio: {result['abuse_ratio']:.1%}")
    print(f"Image flagged: {result['image_flagged']}")
    print(f"\nPer-line results:")
    for detail in result["details"]:
        flag = "🚨" if detail["flagged"] else "✅"
        print(f"  {flag} [{detail['label']:<12} {detail['score']:.0%}] {detail['text']}")
    
    # Example 3: Surya integration
    print("\n" + "─"*80)
    print("Example 3: Surya OCR Integration")
    print("─"*80)
    
    # Simulated Surya OCR output
    mock_surya_output = {
        "text": "aaj bahut accha tha, tujhe toh main dekh lunga, mausam acha hai",
        "lines": [
            {"text": "aaj bahut accha tha", "bbox": [0, 0, 100, 50], "confidence": 0.95},
            {"text": "tujhe toh main dekh lunga", "bbox": [0, 50, 100, 100], "confidence": 0.92},
            {"text": "mausam acha hai", "bbox": [0, 100, 100, 150], "confidence": 0.97},
        ]
    }
    
    result = moderate_image_from_surya(mock_surya_output, classifier, verbose=True)
    
    # Example 4: High abuse content
    print("\n" + "─"*80)
    print("Example 4: High Abuse Content")
    print("─"*80)
    
    high_abuse_lines = [
        "tujhe toh main dekh lunga aukaat mein reh",
        "teri ma ki aankh nikal yahan se",
        "bakwaas band kar warna thappad padega",
        "saala kamine kuch kaam nahi hai tujhe",
        "ch*tiye apna muh band rakh",
    ]
    
    result = moderate_ocr_lines(high_abuse_lines, classifier)
    print(f"\nAbuse ratio: {result['abuse_ratio']:.0%}")
    print(f"Image flagged: {'🚨 YES' if result['image_flagged'] else '✅ NO'}")
    
    print("\n" + "="*80)
    print("Examples complete!")
    print("="*80)
