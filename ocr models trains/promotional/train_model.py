import os
import argparse
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
import evaluate
from sklearn.model_selection import train_test_split

from config import Config
from dataset_analysis import clean_text

def compute_metrics_factory(f1_metric, accuracy_metric, precision_metric, recall_metric):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)

        acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
        precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"]

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    return compute_metrics

def prepare_dataset(df: pd.DataFrame) -> DatasetDict:
    # Basic cleaning
    df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].apply(clean_text)
    
    # Drop empty and map labels
    df = df[df[Config.TEXT_COLUMN].str.len() > 0]
    df['label'] = df[Config.LABEL_COLUMN].map(Config.LABEL_MAP)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Train, Val, Test split
    train_df, temp_df = train_test_split(
        df, 
        test_size=(Config.VAL_SIZE + Config.TEST_SIZE), 
        random_state=Config.RANDOM_STATE, 
        stratify=df['label']
    )
    
    val_ratio = Config.VAL_SIZE / (Config.VAL_SIZE + Config.TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        random_state=Config.RANDOM_STATE,
        stratify=temp_df['label']
    )

    hf_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df[[Config.TEXT_COLUMN, "label"]].reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df[[Config.TEXT_COLUMN, "label"]].reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df[[Config.TEXT_COLUMN, "label"]].reset_index(drop=True)),
    })
    
    return hf_dataset

def train_model(model_key: str, hf_dataset: DatasetDict, output_dir: str, epochs: int = Config.EPOCHS, batch_size: int = Config.BATCH_SIZE, lr: float = Config.LEARNING_RATE):
    model_name_or_path = Config.MODELS[model_key]
    print(f"\n--- Training {model_key} ({model_name_or_path}) ---")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def tokenize_function(examples):
        return tokenizer(examples[Config.TEXT_COLUMN], truncation=True, max_length=Config.MAX_LENGTH)

    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True, remove_columns=[Config.TEXT_COLUMN])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, 
        num_labels=len(Config.LABEL_MAP),
        id2label=Config.ID_TO_LABEL,
        label2id={v: k for k, v in Config.ID_TO_LABEL.items()}
    )

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, model_key),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=Config.WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir=str(Config.REPORTS_DIR / f"logs_{model_key}"),
        logging_steps=50,
        warmup_ratio=Config.WARMUP_RATIO,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(f1_metric, accuracy_metric, precision_metric, recall_metric),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving best model...")
    best_model_path = os.path.join(output_dir, model_key, "best")
    trainer.save_model(best_model_path)
    print(f"Model saved to {best_model_path}")
    
    return best_model_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(Config.MODELS.keys()) + ["all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    Config.setup_dirs()
    
    print(f"Loading data from {Config.DATA_PATH}")
    df = pd.read_csv(Config.DATA_PATH)
    
    if args.dry_run:
        df = df.sample(1000, random_state=Config.RANDOM_STATE)
        Config.EPOCHS = 1
        print("DRY RUN: Using 1000 samples and 1 epoch")

    hf_dataset = prepare_dataset(df)
    
    models_to_train = list(Config.MODELS.keys()) if args.model == "all" else [args.model]
    
    for m in models_to_train:
        train_model(m, hf_dataset, output_dir=str(Config.MODELS_DIR))

if __name__ == "__main__":
    main()
