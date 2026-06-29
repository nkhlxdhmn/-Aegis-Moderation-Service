import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config import Config

class PromotionalDetector:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = str(Config.BEST_MODEL_DIR)
            
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} not found. Please train and export a model first.")
            
        print(f"Loading promotional detector from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=self.device,
            return_all_scores=True
        )

    def _keyword_boost(self, text: str, confidence: float) -> float:
        """Boost confidence if specific promotional keywords are found."""
        text_lower = text.lower()
        has_kw = any(kw in text_lower for kw in Config.PROMO_KEYWORDS)
        
        if has_kw:
            # Boost confidence towards 1.0
            boosted = confidence + Config.KEYWORD_BOOST_WEIGHT * (1.0 - confidence)
            return min(1.0, boosted)
        return confidence

    def predict(self, text: str) -> dict:
        """
        Predict whether a text is promotional.
        
        Args:
            text (str): The text message to classify.
            
        Returns:
            dict: Classification result containing label and confidence.
        """
        # Clean text
        from dataset_analysis import clean_text
        cleaned_text = clean_text(text)
        
        if not cleaned_text.strip():
            return {"label": Config.ID_TO_LABEL[0], "confidence": 1.0}
            
        results = self.classifier(cleaned_text, truncation=True, max_length=Config.MAX_LENGTH)[0]
        
        # Results is a list of dicts: [{'label': 'NON_PROMOTIONAL', 'score': 0.1}, {'label': 'PROMOTIONAL', 'score': 0.9}]
        promo_score = next((r['score'] for r in results if r['label'] == Config.ID_TO_LABEL[1]), 0.0)
        
        # Apply keyword boosting
        boosted_score = self._keyword_boost(cleaned_text, promo_score)
        
        if boosted_score >= 0.5:
            return {
                "label": Config.ID_TO_LABEL[1],
                "confidence": round(boosted_score, 4)
            }
        else:
            return {
                "label": Config.ID_TO_LABEL[0],
                "confidence": round(1.0 - boosted_score, 4)
            }

# Singleton instance
_detector = None

def predict(text: str) -> dict:
    """Convenience function for inference."""
    global _detector
    if _detector is None:
        _detector = PromotionalDetector()
    return _detector.predict(text)

if __name__ == "__main__":
    # Test cases
    test_texts = [
        "Join our free AI course today",
        "Your OTP for login is 123456",
        "Limited offer! Get 50% OFF on all items",
        "Your account balance is Rs 500"
    ]
    
    print("Testing Inference Module:")
    try:
        detector = PromotionalDetector()
        for t in test_texts:
            result = detector.predict(t)
            print(f"\nText: {t}")
            print(f"Result: {result}")
    except FileNotFoundError as e:
        print(f"Setup incomplete: {e}")
