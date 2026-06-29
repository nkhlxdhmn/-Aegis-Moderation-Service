import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "FINAL_MERGED_SMS_DATASET.csv"
    MODELS_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports"
    BEST_MODEL_DIR = MODELS_DIR / "best_model"

    # Dataset settings
    TEXT_COLUMN = "message"
    LABEL_COLUMN = "category"
    LABEL_MAP = {"service": 0, "promotional": 1}
    ID_TO_LABEL = {0: "NON_PROMOTIONAL", 1: "PROMOTIONAL"}

    # Training Data Split
    TEST_SIZE = 0.1
    VAL_SIZE = 0.1 # 10% of remaining 90% -> ~9% overall, or we split temp 50/50
    RANDOM_STATE = 42

    # Models to evaluate
    MODELS = {
        "minilm": "microsoft/MiniLM-L12-H384-uncased",
        "distilbert": "distilbert-base-uncased",
        "muril": "google/muril-base-cased",
        "xlm-roberta": "xlm-roberta-base"
    }

    # Default Training Hyperparameters
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1

    # Keyword boosting list
    PROMO_KEYWORDS = [
        "free", "offer", "discount", "sale", "admission", "registration",
        "telegram", "whatsapp", "follow us", "subscribe", "election",
        "vote", "campaign", "coaching", "buy now", "limited offer", "contact us"
    ]
    
    KEYWORD_BOOST_WEIGHT = 0.2 # How much to boost probability if keyword found

    @classmethod
    def setup_dirs(cls):
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
