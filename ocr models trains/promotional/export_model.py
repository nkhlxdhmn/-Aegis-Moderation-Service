import os
import argparse
import shutil
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import Config

def export_model(model_dir: str):
    Config.setup_dirs()
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist.")
        return
        
    print(f"Exporting model from {model_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # 1. Save standard Hugging Face format
    model.save_pretrained(Config.BEST_MODEL_DIR)
    tokenizer.save_pretrained(Config.BEST_MODEL_DIR)
    
    # 2. Export to .pth (State Dict)
    pth_path = os.path.join(Config.BEST_MODEL_DIR, "promotional_detector.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"Exported to {pth_path}")
    
    # Create dummy input for tracing (max length 128)
    dummy_input = tokenizer("Join our free AI course today", return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]
    
    # 3. Export to TorchScript (.pt)
    model.eval()
    try:
        traced_model = torch.jit.trace(model, (input_ids, attention_mask), strict=False)
        pt_path = os.path.join(Config.BEST_MODEL_DIR, "promotional_detector.pt")
        torch.jit.save(traced_model, pt_path)
        print(f"Exported to {pt_path}")
    except Exception as e:
        print(f"Warning: Could not export to TorchScript: {e}")
        
    # 4. Export to ONNX
    try:
        onnx_path = os.path.join(Config.BEST_MODEL_DIR, "promotional_detector.onnx")
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            }
        )
        print(f"Exported to {onnx_path}")
    except Exception as e:
        print(f"Warning: Could not export to ONNX: {e}")
        
    # 5. Copy metrics if available
    metrics_path = os.path.join(model_dir, "trainer_state.json")
    if os.path.exists(metrics_path):
        dest_metrics = os.path.join(Config.BEST_MODEL_DIR, "training_metrics.json")
        shutil.copy2(metrics_path, dest_metrics)
        print(f"Copied metrics to {dest_metrics}")
        
    print("\nModel export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, help="Path to the trained model directory to export")
    args = parser.parse_args()
    
    model_path = args.model_dir
    if not model_path:
        # Default to reading from best_model_info.txt if available
        info_file = Config.MODELS_DIR / "best_model_info.txt"
        if os.path.exists(info_file):
            with open(info_file, "r") as f:
                best_model_name = f.read().strip()
                model_path = os.path.join(Config.MODELS_DIR, best_model_name, "best")
        else:
            print("Please specify a model directory with --model-dir")
            exit(1)
            
    export_model(model_path)
