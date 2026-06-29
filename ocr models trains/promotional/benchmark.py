import time
import os
import argparse
import pandas as pd
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from config import Config
from train_model import prepare_dataset

def benchmark_models(dry_run=False):
    Config.setup_dirs()
    print(f"Loading data for benchmark from {Config.DATA_PATH}")
    
    df = pd.read_csv(Config.DATA_PATH)
    if dry_run:
        df = df.sample(100, random_state=Config.RANDOM_STATE)
        
    hf_dataset = prepare_dataset(df)
    test_texts = hf_dataset["test"][Config.TEXT_COLUMN].tolist()
    
    # We'll use a subset of 1000 items for pure inference benchmarking
    if len(test_texts) > 1000:
        test_texts = test_texts[:1000]
        
    results = []
    
    # Check baseline (TF-IDF if possible, but we'll focus on transformer models)
    
    for model_key in Config.MODELS.keys():
        model_path = os.path.join(Config.MODELS_DIR, model_key, "best")
        if not os.path.exists(model_path):
            print(f"Model {model_key} not found at {model_path}. Skipping.")
            continue
            
        print(f"\n--- Benchmarking {model_key} ---")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
        
        # Warmup
        _ = classifier(test_texts[:10], truncation=True, max_length=Config.MAX_LENGTH)
        
        # CPU/GPU Benchmarking (sequential)
        start_time = time.time()
        for t in test_texts:
            _ = classifier(t, truncation=True, max_length=Config.MAX_LENGTH)
        end_time = time.time()
        
        seq_time = end_time - start_time
        seq_per_item = (seq_time / len(test_texts)) * 1000  # in ms
        
        # Batch Benchmarking
        start_time = time.time()
        for i in range(0, len(test_texts), Config.BATCH_SIZE):
            batch = test_texts[i:i+Config.BATCH_SIZE]
            _ = classifier(batch, truncation=True, max_length=Config.MAX_LENGTH)
        end_time = time.time()
        
        batch_time = end_time - start_time
        batch_per_item = (batch_time / len(test_texts)) * 1000 # in ms
        
        results.append({
            "Model": model_key,
            "Parameters": sum(p.numel() for p in model.parameters()),
            "Seq Time (ms/item)": round(seq_per_item, 2),
            "Batch Time (ms/item)": round(batch_per_item, 2),
            "Samples/sec (Batch)": round(len(test_texts) / batch_time, 2)
        })
        
    if not results:
        print("No models available to benchmark.")
        return
        
    results_df = pd.DataFrame(results)
    print("\n=== Benchmark Results ===")
    print(results_df.to_markdown(index=False))
    
    report_path = Config.REPORTS_DIR / "benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model Inference Benchmark\n\n")
        f.write(f"Environment: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Batch Size: {Config.BATCH_SIZE}\n\n")
        f.write(results_df.to_markdown(index=False))
        
    print(f"\nBenchmark report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    benchmark_models(args.dry_run)
