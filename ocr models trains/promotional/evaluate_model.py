import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config import Config
from train_model import prepare_dataset

def evaluate_all_models(dry_run=False):
    Config.setup_dirs()
    
    print(f"Loading data from {Config.DATA_PATH}")
    df = pd.read_csv(Config.DATA_PATH)
    if dry_run:
        df = df.sample(1000, random_state=Config.RANDOM_STATE)
    
    hf_dataset = prepare_dataset(df)
    test_texts = hf_dataset["test"][Config.TEXT_COLUMN]
    test_labels = hf_dataset["test"]["label"]

    results = []
    best_f1 = 0
    best_model_name = None

    for model_key in Config.MODELS.keys():
        model_path = os.path.join(Config.MODELS_DIR, model_key, "best")
        if not os.path.exists(model_path):
            print(f"Model {model_key} not found at {model_path}. Skipping.")
            continue
            
        print(f"\n--- Evaluating {model_key} ---")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        
        # Predict in batches
        predictions = []
        for i in range(0, len(test_texts), Config.BATCH_SIZE):
            batch = test_texts[i:i+Config.BATCH_SIZE]
            batch_preds = classifier(batch, truncation=True, max_length=Config.MAX_LENGTH)
            for p in batch_preds:
                # convert label string back to int
                pred_int = 1 if p['label'] == Config.ID_TO_LABEL[1] else 0
                predictions.append(pred_int)
                
        # Metrics
        acc = accuracy_score(test_labels, predictions)
        prec = precision_score(test_labels, predictions, average='macro')
        rec = recall_score(test_labels, predictions, average='macro')
        f1 = f1_score(test_labels, predictions, average='macro')
        
        results.append({
            "Model": model_key,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_key
            
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[Config.ID_TO_LABEL[0], Config.ID_TO_LABEL[1]],
                    yticklabels=[Config.ID_TO_LABEL[0], Config.ID_TO_LABEL[1]])
        plt.title(f"Confusion Matrix - {model_key}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(Config.REPORTS_DIR / f"confusion_matrix_{model_key}.png")
        plt.close()
        
    if not results:
        print("No models evaluated.")
        return
        
    results_df = pd.DataFrame(results)
    
    # Save report
    report = ["# Model Evaluation Report\n"]
    report.append(results_df.to_markdown(index=False))
    report.append(f"\n\n**Best Model selected**: {best_model_name} (F1: {best_f1:.4f})")
    
    report_path = Config.REPORTS_DIR / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"\nEvaluation complete. Report saved to {report_path}")
    print(f"Overall Best Model: {best_model_name}")
    
    # Save a marker for best model
    with open(Config.MODELS_DIR / "best_model_info.txt", "w") as f:
        f.write(best_model_name)

if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    evaluate_all_models(args.dry_run)
