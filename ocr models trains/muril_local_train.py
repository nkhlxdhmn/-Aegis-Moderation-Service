# =============================================================================
# 🛡️  MURIL ABUSIVE vs NON-ABUSIVE CLASSIFIER — LOCAL GPU TRAINING SCRIPT
# =============================================================================
# Base Model : google/muril-base-cased (236M params, Apache 2.0)
# Task       : Binary classification — 0 = Non-Abusive, 1 = Abusive
# Languages  : Hinglish (Roman Hindi), Hindi (Devanagari), Multilingual Indic
# GPU Target : 80GB Local GPU (FP16 mixed-precision + gradient accumulation)
# Framework  : Hugging Face Transformers + Trainer API
# Output     : Fine-tuned weights saved locally for offline inference
# =============================================================================

import os
import sys
import re
import json
import warnings
import random
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold

import numpy as np
import pandas as pd
import torch
from torch import nn

# Hugging Face core imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from datasets import Dataset, DatasetDict
import evaluate

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """Centralized configuration for local training."""
    
    # ── Reproducibility ───────────────────────────────────────────────────
    SEED = 42
    
    # ── Model & tokenizer ─────────────────────────────────────────────────
    MODEL_NAME   = "google/muril-base-cased"
    MAX_LENGTH   = 256
    EPOCHS       = 10

    # ── Training hyperparameters (single RTX 4000 Ada, 20 GB VRAM) ──
    BATCH_SIZE            = 16           # Safe for 20 GB VRAM on one GPU
    GRADIENT_ACCUMULATION = 4            # Effective batch = 64
    LEARNING_RATE         = 3e-5
    WARMUP_RATIO          = 0.06
    WEIGHT_DECAY          = 0.01

    # ── Advanced optimizations ─────────────────────────────────────────
    USE_FP16              = True          # Mixed precision
    GRADIENT_CHECKPOINTING = False       # OFF: FP16 + grad checkpointing hurts convergence
    USE_FLASH_ATTENTION   = False        # Disable: not installed in this env
    NUM_WORKERS           = 2            # Parallel data loading
    
    # ── Label mapping ─────────────────────────────────────────────────────
    LABEL2ID = {"non_abusive": 0, "abusive": 1}
    ID2LABEL = {0: "non_abusive", 1: "abusive"}
    NUM_LABELS = 2
    
    # ── Paths (local filesystem) ──────────────────────────────────────────
    PROJECT_ROOT      = Path(__file__).parent          # muril_test/
    DATA_DIR          = PROJECT_ROOT                   # CSV lives here
    OUTPUT_DIR        = PROJECT_ROOT / "models" / "muril_abuse_model"
    FINAL_MODEL_DIR   = PROJECT_ROOT / "models" / "muril_abuse_final"
    LOGS_DIR          = PROJECT_ROOT / "logs"
    
    def __init__(self):
        # Create directories
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SETUP & REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def setup_reproducibility(seed: int = 42):
    """Set seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_device_info():
    """Print GPU/device information."""
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if device_obj.type == "cuda" else "cpu"
    print(f"\n✅  Device: {device_str}")
    if device_obj.type == "cuda":
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1e9
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)} | {vram_gb:.1f} GB VRAM")
        try:
            cur = torch.cuda.current_device()
            print(f"    Current GPU: {torch.cuda.get_device_name(cur)} (index {cur})")
        except Exception:
            print(f"    Current GPU index: unknown")
    return device_obj


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lightweight text cleaning for social-media / OCR text.
    Preserves Devanagari and other Unicode scripts.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", " ", text)      # Remove URLs
    text = re.sub(r"<[^>]+>", " ", text)             # Remove HTML tags
    text = re.sub(r"([!?.,])\1{2,}", r"\1", text)    # Fix repetition
    text = re.sub(r"\s+", " ", text).strip()         # Normalize whitespace
    return text


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_hasoc(filepath: str) -> pd.DataFrame:
    """Load HASOC 2019/2020 dataset (Hindi/English hate speech)."""
    df = pd.read_csv(filepath, sep="\t") if str(filepath).endswith(".tsv") else pd.read_csv(filepath)
    
    text_col  = next((c for c in df.columns if "text" in c.lower() or "tweet" in c.lower()), df.columns[1])
    label_col = next((c for c in df.columns if "task_1" in c.lower() or "label" in c.lower()), df.columns[2])
    
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "raw_label"})
    df["label"] = df["raw_label"].apply(lambda x: 1 if str(x).strip().upper() in {"HOF", "ABUSIVE", "1", "TRUE"} else 0)
    df["text"]  = df["text"].apply(clean_text)
    return df[["text", "label"]].dropna()


def load_constraint2021(filepath: str) -> pd.DataFrame:
    """Load CONSTRAINT 2021 Hindi Hostility dataset."""
    df = pd.read_csv(filepath)
    
    text_col  = next((c for c in df.columns if "post" in c.lower() or "text" in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if "label" in c.lower() or "hostile" in c.lower()), df.columns[1])
    
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "raw_label"})
    df["label"] = df["raw_label"].apply(
        lambda x: 0 if str(x).strip().lower() in {"non-hostile", "non_hostile", "0", "false"} else 1
    )
    df["text"] = df["text"].apply(clean_text)
    return df[["text", "label"]].dropna()


def load_dravidian(filepath: str) -> pd.DataFrame:
    """Load DravidianLangTech abuse datasets (Tamil, Malayalam, Kannada)."""
    df = pd.read_csv(filepath)
    
    text_col  = next((c for c in df.columns if "text" in c.lower() or "comment" in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if "category" in c.lower() or "label" in c.lower()), df.columns[1])
    
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "raw_label"})
    df["label"] = df["raw_label"].apply(
        lambda x: 0 if "not_off" in str(x).lower() or "non_off" in str(x).lower() else 1
    )
    df["text"] = df["text"].apply(clean_text)
    return df[["text", "label"]].dropna()


def load_kaggle_hinglish(filepath: str) -> pd.DataFrame:
    """Load generic Kaggle Hinglish cyberbullying CSV."""
    df = pd.read_csv(filepath)
    
    text_col  = next((c for c in df.columns if c.lower() in {"tweet", "text", "sentence", "comment"}), df.columns[0])
    label_col = next((c for c in df.columns if c.lower() in {"label", "class", "target", "abusive"}), df.columns[1])
    
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "raw_label"})
    df["label"] = df["raw_label"].apply(
        lambda x: 0 if str(x).strip() in {"0", "0.0", "non-abusive", "normal", "not_abusive"} else 1
    )
    df["text"] = df["text"].apply(clean_text)
    return df[["text", "label"]].dropna()


def load_combined_csv(filepath: str) -> pd.DataFrame:
    """Load the combined hate speech CSV (combined_hate_speech_dataset.csv).
    Expected columns: text, hate_label (0=non-abusive, 1=abusive).
    """
    df = pd.read_csv(filepath)
    if "hate_label" not in df.columns:
        raise ValueError(f"Expected 'hate_label' column in {filepath}. Found: {df.columns.tolist()}")
    df = df[["text", "hate_label"]].rename(columns={"hate_label": "label"})
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(clean_text)
    return df[["text", "label"]].dropna()


def build_combined_dataset(data_dir: Path, selected: str = None, csv_file: Path = None) -> pd.DataFrame:
    """Load datasets and concatenate them.

    Priority:
      1. If `csv_file` is given (or combined_hate_speech_dataset.csv exists in
         data_dir), load that single combined CSV directly.
      2. Otherwise fall back to loading individual dataset files by name.

    `selected` filters individual datasets by key/filename (ignored when the
    combined CSV is used).
    """
    all_dfs = []

    # ── Fast path: load the combined CSV ────────────────────────────────
    combined_csv_path = csv_file or (data_dir / "combined_hate_speech_dataset.csv")
    if combined_csv_path.exists():
        print(f"\n📂 Loading combined CSV: {combined_csv_path}")
        print("─" * 70)
        try:
            df = load_combined_csv(str(combined_csv_path))
            abuse_rate = df["label"].mean()
            print(f"  ✅  combined_hate_speech_dataset.csv : {len(df):>6} rows | Abuse: {abuse_rate:>6.1%}")
            all_dfs.append(df)
        except Exception as e:
            print(f"  ❌  Failed to load combined CSV: {e}")

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined = combined[combined["text"].str.len() > 3]
            combined = combined.drop_duplicates(subset="text")
            combined = combined.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
            print("─" * 70)
            print(f"\n📊  Dataset            : {len(combined):>6} rows")
            print(f"    Non-abusive (0)    : {(combined['label']==0).sum():>6} ({(combined['label']==0).mean():.1%})")
            print(f"    Abusive     (1)    : {(combined['label']==1).sum():>6} ({(combined['label']==1).mean():.1%})")
            return combined

    # ── Fall-back: individual dataset files ─────────────────────────────
    # Define expected dataset files (key, filename, loader, description)
    dataset_configs = [
        ("hasoc_2021_hindi_train", "hasoc_2021_hindi_train.tsv", load_hasoc, "HASOC (Hindi/English)"),
        ("constraint_hindi_train", "constraint_hindi_train.csv", load_constraint2021, "CONSTRAINT 2021"),
        ("tamil_offensive_train", "tamil_offensive_train.csv", load_dravidian, "DravidianLangTech Tamil"),
        ("malayalam_offensive_train", "malayalam_offensive_train.csv", load_dravidian, "DravidianLangTech Malayalam"),
        ("hinglish_profanity", "hinglish_profanity.csv", load_kaggle_hinglish, "Kaggle Hinglish Profanity"),
        ("hinglish_hate_speech", "hinglish_hate_speech.csv", load_kaggle_hinglish, "Kaggle Hinglish Hate Speech"),
    ]
    
    print("\n📂 Loading datasets from: " + str(data_dir))
    print("─" * 70)
    
    # If a specific dataset key/filename was provided, attempt to load only that
    if selected:
        selected_lower = selected.lower()
        matched = [c for c in dataset_configs if selected_lower in c[0].lower() or selected_lower in c[1].lower() or selected_lower in c[3].lower()]
        if not matched:
            print(f"⚠️   Selected dataset '{selected}' not recognized. Falling back to loading all available datasets.")
        else:
            dataset_configs = matched

    for key, filename, loader_fn, desc in dataset_configs:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                df = loader_fn(str(filepath))
                abuse_rate = df["label"].mean() if len(df) > 0 else 0
                print(f"  ✅  {desc:<35} : {len(df):>6} rows | Abuse: {abuse_rate:>6.1%}")
                all_dfs.append(df)
            except Exception as e:
                print(f"  ⚠️   {desc:<35} : ERROR - {str(e)[:40]}")
        else:
            print(f"  ⏭️   {desc:<35} : Not found")
    
    # Fallback: synthetic demo dataset
    if not all_dfs:
        print("\n  🔶  No datasets found. Using DEMO synthetic data for dry-run.")
        print("      Replace with real datasets in: " + str(data_dir) + "\n")
        demo_texts = [
            # Non-abusive
            "aaj mausam bahut accha hai", "please share feedback", "Tamil Nadu cricket team won",
            "yeh movie bahut achi thi", "good morning everyone", "Kal market band rahega",
            "coffee peena chahta hoon", "naye updates aa gaye", "train late chal rahi",
            "mere birthday pe aana",
            # Abusive
            "tujhe toh main dekh lunga", "teri ma ki aankh", "bakwaas band kar",
            "saala kamine", "ch*tiye apna muh band", "sab gadhe ho", "besharam insaan",
            "maa ki ankh bandh kar", "ullu ke patthe", "teri aukat nahi",
        ]
        demo_labels = [0]*10 + [1]*10
        all_dfs.append(pd.DataFrame({"text": demo_texts, "label": demo_labels}))
    
    # Combine, deduplicate, shuffle
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined[combined["text"].str.len() > 3]
    combined = combined.drop_duplicates(subset="text")
    combined = combined.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
    
    print("─" * 70)
    print(f"\n📊  Combined dataset   : {len(combined):>6} rows")
    print(f"    Non-abusive (0)    : {(combined['label']==0).sum():>6} ({(combined['label']==0).mean():.1%})")
    print(f"    Abusive     (1)    : {(combined['label']==1).sum():>6} ({(combined['label']==1).mean():.1%})")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM TRAINER WITH WEIGHTED LOSS
# ─────────────────────────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Trainer with per-class loss weighting + auxiliary profanity prediction."""

    def __init__(self, class_weights_tensor, use_aux_loss=False, 
                 aux_loss_weight=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_tensor = class_weights_tensor
        self.use_aux_loss = use_aux_loss
        self.aux_loss_weight = aux_loss_weight
        
        # Add auxiliary head if using aux loss
        if use_aux_loss:
            # Use model's hidden size if available (avoid hard-coded 768)
            hidden_size = getattr(self.model.config, "hidden_size", 768) if getattr(self, "model", None) is not None else 768
            self.aux_head = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)  # Predict profanity_score (0-10)
            )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        profanity_scores = inputs.pop("profanity_scores", None)  # NEW: extract aux labels
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Main task: abuse classification
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        loss_main = loss_fn(logits, labels)
        
        # Auxiliary task (if enabled and scores provided)
        loss_aux = 0
        if self.use_aux_loss and profanity_scores is not None:
            # Get [CLS] token representation
            cls_hidden = outputs.hidden_states[-1][:, 0, :]  # (batch_size, 768)
            
            # Predict profanity score
            profanity_pred = self.aux_head(cls_hidden).squeeze(-1)  # (batch_size,)
            
            # MSE loss on profanity prediction
            loss_aux = nn.MSELoss()(
                profanity_pred, 
                profanity_scores.float() / 10.0  # Normalize to [0, 1]
            )
        
        # Combined loss
        loss = loss_main + self.aux_loss_weight * loss_aux
        
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────────────
# METRICS COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_factory(f1_metric, accuracy_metric):
    """Factory to create compute_metrics function with metric objects."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        # Trainer can hand metrics a tuple/list when the model returns extra outputs.
        # Keep only the classification logits before computing argmax.
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        logits = np.asarray(logits)
        predictions = np.argmax(logits, axis=-1)

        acc        = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1_macro   = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
        f1_weighted= f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        f1_abuse   = f1_metric.compute(predictions=predictions, references=labels, average=None)["f1"][1]

        return {
            "accuracy":         round(acc,         4),
            "f1_macro":         round(f1_macro,    4),
            "f1_weighted":      round(f1_weighted, 4),
            "f1_abusive_class": round(f1_abuse,    4),
        }
    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main(args: Optional[argparse.Namespace] = None):
    """Main training entry point."""
    
    # Parse arguments
    if args is None:
        parser = argparse.ArgumentParser(description="Local MuRIL abuse classifier training")
        parser.add_argument("--data-dir", type=Path, default=None, help="Dataset directory (default: script directory)")
        parser.add_argument("--csv-file", type=Path, default=None,
                            help="Path to combined CSV (default: <data-dir>/combined_hate_speech_dataset.csv)")
        parser.add_argument("--dataset", type=str, default=None, help="Load only this individual dataset by key/filename")
        parser.add_argument("--dry-run", action="store_true", help="Run with 200 samples for a quick smoke-test")
        parser.add_argument("--disable-fp16", action="store_true", help="Disable FP16 mixed precision")
        parser.add_argument("--disable-flash-attn", action="store_true", help="Disable Flash Attention")
        parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: 5)")
        parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size (default: 32)")
        parser.add_argument(
            "--cuda-devices",
            type=str,
            default="1",
            help="Comma-separated CUDA visible devices to use (default: 1)",
        )
        args = parser.parse_args()

    # Initialize config
    config = Config()
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir)
    csv_file = args.csv_file
    selected_dataset = args.dataset
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.disable_fp16:
        config.USE_FP16 = False
    # Respect requested CUDA devices unless a distributed launcher already set them.
    if os.environ.get("LOCAL_RANK") is None and os.environ.get("RANK") is None:
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_devices)
        except Exception:
            pass
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    visible_devices = [x.strip() for x in str(os.environ.get("CUDA_VISIBLE_DEVICES", "1")).split(",") if x.strip()]
    if len(visible_devices) > 1 and args.batch_size is None:
        config.BATCH_SIZE = 8

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank))
    
    # Setup
    setup_reproducibility(config.SEED)
    device = print_device_info()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f"training_{timestamp}.log"
    
    print(f"\n📝 Training log: {log_file}")
    print(f"📁 Model output: {config.FINAL_MODEL_DIR}")
    print(f"⚙️  Config:")
    print(f"   Batch size: {config.BATCH_SIZE} | Gradient accumulation: {config.GRADIENT_ACCUMULATION}")
    print(f"   Effective batch: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION}")
    print(f"   FP16: {config.USE_FP16} | Flash Attention: {config.USE_FLASH_ATTENTION}")
    print(f"   Gradient checkpointing: {config.GRADIENT_CHECKPOINTING}")
    
    # ─── LOAD DATA ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 1: LOADING DATASETS")
    print("="*70)
    df_all = build_combined_dataset(config.DATA_DIR, selected=selected_dataset, csv_file=csv_file)

    if args.dry_run:
        print(f"\n🏃 DRY-RUN MODE: Using only 200 samples for smoke-test")
        df_all = df_all.sample(n=min(200, len(df_all)), random_state=config.SEED).reset_index(drop=True)
    
    # ─── SPLIT DATA ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 2: STRATIFIED TRAIN / VAL / TEST SPLIT (by language)")
    print("="*70)
    
    # First split: 70% train, 30% temp (val+test)
    # If a `language` column is present, stratify by the combination of
    # label+language. Otherwise fall back to stratifying by label only.
    if "language" in df_all.columns:
        stratify_series = df_all["label"].astype(str) + "_" + df_all["language"].astype(str)
    else:
        stratify_series = df_all["label"]

    df_train, df_temp = train_test_split(
        df_all,
        test_size=0.30,
        random_state=config.SEED,
        stratify=stratify_series,
    )

    # Second split: 50/50 val/test from temp
    if "language" in df_temp.columns:
        stratify_series_temp = df_temp["label"].astype(str) + "_" + df_temp["language"].astype(str)
    else:
        stratify_series_temp = df_temp["label"]

    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=config.SEED,
        stratify=stratify_series_temp,
    )
    
    # PRINT LANGUAGE OR LABEL DISTRIBUTION IN EACH SPLIT:
    print(f"\nDistribution by split:")
    for split_name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"\n  {split_name}:")
        if "language" in df_all.columns:
            for lang in sorted(df_all["language"].unique()):
                count = (split_df["language"] == lang).sum()
                pct = 100 * count / len(split_df)
                print(f"    {lang}: {count:>5} ({pct:>5.1f}%)")
        else:
            # Fall back to label distribution (0=Non-abusive, 1=Abusive)
            for lbl in sorted(df_all["label"].unique()):
                count = (split_df["label"] == lbl).sum()
                pct = 100 * count / len(split_df)
                label_name = config.ID2LABEL.get(int(lbl), str(lbl)) if hasattr(config, 'ID2LABEL') else str(lbl)
                print(f"    {label_name} ({int(lbl)}): {count:>5} ({pct:>5.1f}%)")
    
    print(f"✂️   Train: {len(df_train):>5} rows | Val: {len(df_val):>5} rows | Test: {len(df_test):>5} rows")
    
    hf_dataset = DatasetDict({
        "train":      Dataset.from_pandas(df_train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(df_val.reset_index(drop=True)),
        "test":       Dataset.from_pandas(df_test.reset_index(drop=True)),
    })
    
    # ─── TOKENIZATION ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 3: TOKENIZATION")
    print("="*70)
    print(f"⏳  Loading tokenizer from {config.MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    print(f"✅  Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")
    
    def tokenize_batch(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding=False,
        )
        # ADD profanity_score to tokenized output
        if "profanity_score" in batch:
            tokens["profanity_scores"] = batch["profanity_score"]
        return tokens
    
    print("⏳  Tokenizing datasets (parallel with " + str(config.NUM_WORKERS) + " workers) ...")
    tokenized_dataset = hf_dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=256,
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=config.NUM_WORKERS,
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    print("✅  Tokenization complete.")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # ─── CLASS WEIGHTS ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 4: CLASS WEIGHT COMPUTATION")
    print("="*70)
    train_labels = df_train["label"].values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_labels,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"⚖️   Non-abusive: {class_weights[0]:.3f} | Abusive: {class_weights[1]:.3f}")
    
    # ─── MODEL INITIALIZATION ────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 5: MODEL INITIALIZATION")
    print("="*70)
    print(f"⏳  Loading model from {config.MODEL_NAME} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )
    # output_hidden_states is NOT set here — enabling it causes the Trainer's
    # eval loop to collect hidden states (variable seq len per batch) alongside
    # logits, making predictions inhomogeneous and crashing compute_metrics.
    # Enable only if use_aux_loss=True is passed to WeightedTrainer.
    
    # Enable gradient checkpointing
    if config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        print("✅  Gradient checkpointing enabled")
    
    # Enable Flash Attention (if available)
    if config.USE_FLASH_ATTENTION:
        try:
            # This requires: pip install flash-attn
            model.config.use_flash_attention_2 = True
            print("✅  Flash Attention 2 enabled")
        except:
            print("⚠️   Flash Attention not available (install: pip install flash-attn)")
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters    : {total_params:>12,}")
    print(f"    Trainable parameters: {trainable_params:>12,}")
    
    # ─── TRAINING ARGUMENTS ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 6: TRAINING CONFIGURATION")
    print("="*70)
    
    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        
        # Epochs & batching
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        # Reduce eval batch size to avoid OOM during evaluation
        per_device_eval_batch_size=max(1, config.BATCH_SIZE),
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        
        # Learning rate
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type="cosine",
        weight_decay=config.WEIGHT_DECAY,
        
        # FP16 mixed precision
        fp16=config.USE_FP16,
        label_smoothing_factor=0.1,

        # Evaluation & saving (transformers ≥ 4.41 uses eval_strategy)
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,

        # Logging
        logging_dir=str(config.LOGS_DIR),
        logging_steps=50,
        report_to="none",
        eval_accumulation_steps=4,
        
        # Reproducibility
        seed=config.SEED,
        data_seed=config.SEED,
        
        # Data loading
        dataloader_num_workers=config.NUM_WORKERS,
        
        # Other
        push_to_hub=False,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
    )
    
    print(f"✅  Training config set")
    print(f"    Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION}")
    print(f"    Gradient accumulation steps: {config.GRADIENT_ACCUMULATION}")
    print(f"    Total steps: {(len(tokenized_dataset['train']) // config.BATCH_SIZE) * config.EPOCHS}")
    
    # ─── METRICS ─────────────────────────────────────────────────────────
    accuracy_metric = evaluate.load("accuracy")
    f1_metric       = evaluate.load("f1")
    compute_metrics = compute_metrics_factory(f1_metric, accuracy_metric)
    
    # ─── TRAINER ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 7: INITIALIZING TRAINER")
    print("="*70)
    
    trainer = WeightedTrainer(
        class_weights_tensor=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=4,
                early_stopping_threshold=0.001
            )
        ],
    )
    
    # ─── TRAINING ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 8: STARTING TRAINING")
    print("="*70)
    print(f"\n🚀  Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    train_result = trainer.train()
    
    print(f"\n✅  Training complete!")
    print(f"    Runtime           : {train_result.metrics['train_runtime']:.0f}s ({train_result.metrics['train_runtime']/60:.1f}m)")
    print(f"    Samples/sec       : {train_result.metrics['train_samples_per_second']:.1f}")
    
    # ─── TEST EVALUATION ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 9: TEST SET EVALUATION")
    print("="*70)
    print("⏳  Evaluating on test set ...\n")
    
    test_results = trainer.predict(tokenized_dataset["test"])
    test_preds  = np.argmax(test_results.predictions, axis=-1)
    test_labels = test_results.label_ids
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT (Test Set)")
    print("="*70 + "\n")
    print(classification_report(
        test_labels, test_preds,
        target_names=["Non-Abusive", "Abusive"],
        digits=4,
    ))
    
    # Save metrics
    test_metrics = {
        "test_accuracy":         float(np.mean(test_preds == test_labels)),
        "test_f1_macro":         float(f1_metric.compute(predictions=test_preds, references=test_labels, average="macro")["f1"]),
        "test_f1_abusive_class": float(f1_metric.compute(predictions=test_preds, references=test_labels, average=None)["f1"][1]),
        "training_time_sec":     float(train_result.metrics['train_runtime']),
    }
    
    print("\n📈  Summary metrics:")
    for k, v in test_metrics.items():
        print(f"    {k:<30}: {v:.4f}")
    
    # ─── SAVE MODEL ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 10: SAVING MODEL")
    print("="*70)
    print(f"\n💾  Saving to {config.FINAL_MODEL_DIR} ...")
    
    trainer.save_model(str(config.FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(config.FINAL_MODEL_DIR))
    
    with open(config.FINAL_MODEL_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # Save config
    with open(config.FINAL_MODEL_DIR / "training_config.json", "w") as f:
        json.dump({
            "model_name": config.MODEL_NAME,
            "max_length": config.MAX_LENGTH,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "use_fp16": config.USE_FP16,
        }, f, indent=2)
    
    print(f"✅  Model saved.")
    print(f"\n📁 Model files:")
    for f in sorted(config.FINAL_MODEL_DIR.glob("*")):
        size_mb = f.stat().st_size / 1e6 if f.is_file() else 0
        marker = "📁" if f.is_dir() else "📄"
        if f.is_file():
            print(f"    {marker} {f.name:<40} {size_mb:>7.1f} MB")
        else:
            print(f"    {marker} {f.name}/")
    
    print("\n" + "="*70)
    print("✨ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📍 Trained model location:\n   {config.FINAL_MODEL_DIR}\n")
    print("🔥 To load for inference:\n")
    print("   from transformers import pipeline")
    print(f"   classifier = pipeline('text-classification', model='{config.FINAL_MODEL_DIR}')")
    print("   result = classifier('your text here')\n")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_trained_classifier(model_dir: Path):
    """Load fine-tuned model for inference."""
    from transformers import AutoTokenizer, pipeline

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    
    classifier = pipeline(
        task="text-classification",
        model=str(model_dir),
        tokenizer=tokenizer,
        device=1 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=128,
    )
    return classifier


def moderate_ocr_output(
    ocr_lines,
    classifier,
    abuse_threshold: float = 0.75,
    image_flag_ratio: float = 0.20,
):
    """
    Moderate OCR-extracted text lines using fine-tuned MuRIL classifier.
    
    Args:
        ocr_lines       : List of text strings from OCR
        classifier      : Loaded pipeline from load_trained_classifier()
        abuse_threshold : Min confidence to flag a line as abusive
        image_flag_ratio: Min fraction of abusive lines to flag entire image
    
    Returns:
        {
          "flagged": bool,
          "abuse_count": int,
          "abuse_ratio": float,
          "total_lines": int,
          "lines": [{text, label, score, flagged}]
        }
    """
    if not ocr_lines:
        return {"flagged": False, "abuse_count": 0, "total_lines": 0, "lines": []}
    
    # Clean lines
    valid_lines = [clean_text(l) for l in ocr_lines if clean_text(l)]
    if not valid_lines:
        return {"flagged": False, "abuse_count": 0, "total_lines": 0, "lines": []}
    
    # Batch inference
    results = classifier(valid_lines, batch_size=32)
    
    line_results = []
    abuse_count = 0
    
    for text, res in zip(valid_lines, results):
        is_abusive = (res["label"] == "abusive" and res["score"] >= abuse_threshold)
        if is_abusive:
            abuse_count += 1
        line_results.append({
            "text": text,
            "label": res["label"],
            "score": round(res["score"], 4),
            "flagged": is_abusive,
        })
    
    abuse_ratio = abuse_count / len(valid_lines)
    image_flagged = abuse_ratio >= image_flag_ratio
    
    return {
        "flagged": image_flagged,
        "abuse_count": abuse_count,
        "abuse_ratio": round(abuse_ratio, 3),
        "total_lines": len(valid_lines),
        "lines": line_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
