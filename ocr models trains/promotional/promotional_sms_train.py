from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


LABEL_MAP = {"service": 0, "promotional": 1}
ID_TO_LABEL = {0: "service", 1: "promotional"}


@dataclass
class TrainConfig:
    data_path: Path
    output_dir: Path
    random_state: int = 42
    test_size: float = 0.10
    val_size: float = 0.10
    max_samples: int | None = None
    c_value: float = 4.0
    max_iter: int = 3000


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^\w\s%₹.-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: Path, max_samples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"message", "category"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns {sorted(expected)}; missing {sorted(missing)}")

    df = df[["message", "category"]].copy()
    df["message"] = df["message"].astype(str).map(clean_text)
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df = df[df["category"].isin(LABEL_MAP)]
    df["label"] = df["category"].map(LABEL_MAP).astype(int)
    df = df[df["message"].str.len() > 2].drop_duplicates(subset=["message", "label"])

    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    return df[["message", "label"]].reset_index(drop=True)


def build_pipeline(c_value: float, max_iter: int) -> Pipeline:
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=120_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.98,
        max_features=160_000,
        sublinear_tf=True,
    )

    features = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    classifier = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
        verbose=0,
        random_state=42,
    )

    return Pipeline([
        ("features", features),
        ("classifier", classifier),
    ])


def evaluate_split(model: Pipeline, x: pd.Series, y: pd.Series) -> Dict[str, float]:
    predictions = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1_macro": float(f1_score(y, predictions, average="macro")),
        "f1_weighted": float(f1_score(y, predictions, average="weighted")),
    }


def train(config: TrainConfig) -> Tuple[Pipeline, Dict[str, Dict[str, float]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_dataset(config.data_path, config.max_samples)

    train_val_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["label"],
    )

    val_ratio = config.val_size / (1.0 - config.test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=config.random_state,
        stratify=train_val_df["label"],
    )

    model = build_pipeline(config.c_value, config.max_iter)
    model.fit(train_df["message"], train_df["label"])

    metrics = {
        "train": evaluate_split(model, train_df["message"], train_df["label"]),
        "validation": evaluate_split(model, val_df["message"], val_df["label"]),
        "test": evaluate_split(model, test_df["message"], test_df["label"]),
    }

    return model, metrics, train_df, val_df, test_df


def save_artifacts(model: Pipeline, metrics: Dict[str, Dict[str, float]], config: TrainConfig, data_stats: Dict[str, int]) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.output_dir / "promotional_service_model.joblib")

    serializable_config = {
        "data_path": str(config.data_path),
        "output_dir": str(config.output_dir),
        "random_state": config.random_state,
        "test_size": config.test_size,
        "val_size": config.val_size,
        "max_samples": config.max_samples,
        "c_value": config.c_value,
        "max_iter": config.max_iter,
    }

    with open(config.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(config.output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(serializable_config, f, indent=2)

    with open(config.output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in ID_TO_LABEL.items()}, f, indent=2)

    with open(config.output_dir / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump(data_stats, f, indent=2)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a service vs promotional SMS classifier")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).parent / "FINAL_MERGED_SMS_DATASET.csv",
        help="Path to FINAL_MERGED_SMS_DATASET.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "models" / "promotional_service_model",
        help="Directory to save model and metrics",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick experiments")
    parser.add_argument("--c-value", type=float, default=4.0)
    parser.add_argument("--max-iter", type=int, default=3000)
    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=args.val_size,
        max_samples=args.max_samples,
        c_value=args.c_value,
        max_iter=args.max_iter,
    )


def main() -> None:
    config = parse_args()
    model, metrics, train_df, val_df, test_df = train(config)

    data_stats = {
        "total_rows": int(len(train_df) + len(val_df) + len(test_df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "service_rows": int((train_df["label"] == 0).sum() + (val_df["label"] == 0).sum() + (test_df["label"] == 0).sum()),
        "promotional_rows": int((train_df["label"] == 1).sum() + (val_df["label"] == 1).sum() + (test_df["label"] == 1).sum()),
    }

    save_artifacts(model, metrics, config, data_stats)

    print("\n=== DATA SPLIT ===")
    print(data_stats)
    print("\n=== METRICS ===")
    for split_name, split_metrics in metrics.items():
        print(split_name, split_metrics)

    print("\n=== TEST CLASSIFICATION REPORT ===")
    test_pred = model.predict(test_df["message"])
    print(classification_report(test_df["label"], test_pred, target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]], digits=4))
    print(f"\nSaved model to: {config.output_dir}")


if __name__ == "__main__":
    main()