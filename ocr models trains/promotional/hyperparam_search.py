import os
import argparse
import pandas as pd
from config import Config
from train_model import prepare_dataset, train_model

def hyperparameter_search(model_key: str, dry_run: bool = False):
    Config.setup_dirs()
    
    print(f"Loading data for HP search from {Config.DATA_PATH}")
    df = pd.read_csv(Config.DATA_PATH)
    if dry_run:
        df = df.sample(1000, random_state=Config.RANDOM_STATE)
        lrs = [5e-5]
        batch_sizes = [32]
        epochs_list = [1]
    else:
        lrs = [2e-5, 5e-5]
        batch_sizes = [16, 32]
        epochs_list = [3, 5]
        
    hf_dataset = prepare_dataset(df)
    
    results = []
    best_f1 = 0
    best_run_path = None
    
    run_idx = 1
    
    # Simple grid search
    for lr in lrs:
        for bs in batch_sizes:
            for epochs in epochs_list:
                print(f"\n=== HP Run {run_idx}: LR={lr}, BS={bs}, Epochs={epochs} ===")
                
                run_dir = os.path.join(Config.MODELS_DIR, "hp_search", f"run_{run_idx}")
                
                # Train model
                model_path = train_model(
                    model_key, 
                    hf_dataset, 
                    output_dir=run_dir, 
                    epochs=epochs, 
                    batch_size=bs, 
                    lr=lr
                )
                
                # We would normally evaluate on validation set here, but Trainer already
                # evaluates. We can read the trainer_state.json to get best val f1.
                import json
                state_path = os.path.join(model_path, "trainer_state.json")
                if os.path.exists(state_path):
                    with open(state_path, "r") as f:
                        state = json.load(f)
                    # Get the best metric
                    val_f1 = state.get("best_metric", 0)
                else:
                    val_f1 = 0 # Fallback
                
                results.append({
                    "Run": run_idx,
                    "LR": lr,
                    "Batch_Size": bs,
                    "Epochs": epochs,
                    "Val_F1": val_f1,
                    "Model_Path": model_path
                })
                
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_run_path = model_path
                
                run_idx += 1
                
    results_df = pd.DataFrame(results)
    print("\n--- HP Search Results ---")
    print(results_df.to_markdown(index=False))
    
    results_df.to_csv(Config.REPORTS_DIR / "hp_search_results.csv", index=False)
    
    print(f"\nBest configuration achieved Val_F1 = {best_f1} at path {best_run_path}")
    return best_run_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="minilm", choices=list(Config.MODELS.keys()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    hyperparameter_search(args.model, args.dry_run)
