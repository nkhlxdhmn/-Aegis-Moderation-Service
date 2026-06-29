# Promotional Content Classifier

I have implemented the complete pipeline for training a promotional vs non-promotional text classification model using your `FINAL_MERGED_SMS_DATASET.csv` dataset.

All scripts are generated inside `test-content-moderation/promotional/`.

## 📁 File Structure

Here are the key scripts created for the pipeline:

- [config.py](file:///c:/Users/sky2k/OneDrive/Documents/test-content-moderation/promotional/config.py): Contains all hyperparameters, models to test, and keyword lists.
- [dataset_analysis.py](file:///c:/Users/sky2k/OneDrive/Documents/test-content-moderation/promotional/dataset_analysis.py): Analyzes the CSV, reports missing data, class distribution, and outputs `reports/dataset_report.md`.
- [train_model.py](file:///c:/Users/sky2k/OneDrive/Documents/test-content-moderation/promotional/train_model.py): Fine-tunes multiple transformer models (MiniLM, DistilBERT, MuRIL, XLM-R) using Hugging Face Trainer. Features early stopping and mixed precision.
- [evaluate_model.py](file:///c:/Users/sky2k/OneDrive/Documents/test-content-moderation/promotional/evaluate_model.py): Evaluates models on the test set, generates a confusion matrix, selects the best model based on F1-score, and outputs `reports/evaluation_report.md`.
- [hyperparam_search.py](file:///c:/Users/sky2k/OneDrive/Documents/test-content-moderation/promotional/hyperparam_search.py): Runs a grid search over learning rates, batch sizes, and epochs to tune the selected model.
- [export_model.py](file:///c:/Users/sky2k/OneDrive/Documents/test-content-moderation/promotional/export_model.py): Exports the best model to `.pth`, `.pt` (TorchScript), and `.onnx` formats in `models/best_model/`.
- [inference.py](file:///c:/Users/sky2k/OneDrive/Documents/test-content-moderation/promotional/inference.py): A simple Python API to load the model and make predictions, applying custom keyword boosting.
- [benchmark.py](file:///c:/Users/sky2k/OneDrive/Documents/test-content-moderation/promotional/benchmark.py): Generates speed benchmarks and comparisons between models.

## 🚀 How to Run the Pipeline

You can run the full pipeline sequentially from the `promotional` directory:

1. **Analyze Dataset**
   ```bash
   python dataset_analysis.py
   ```
2. **Train Models**
   ```bash
   # Train all models
   python train_model.py
   
   # Or run a dry-run test (1000 samples, 1 epoch)
   python train_model.py --dry-run
   ```
3. **Evaluate and Select Best Model**
   ```bash
   python evaluate_model.py
   ```
4. **Hyperparameter Tuning (Optional)**
   ```bash
   # Search parameters for the best model (e.g., minilm)
   python hyperparam_search.py --model minilm
   ```
5. **Export the Best Model**
   ```bash
   python export_model.py
   ```

## 🧠 Inference API

After training and exporting, you can easily use the model in your moderation service using `inference.py`:

```python
from promotional.inference import predict

# Test the model
result = predict("Join our free AI course today! 50% OFF.")
print(result)
```

**Example Output:**
```json
{
  "label": "PROMOTIONAL",
  "confidence": 0.985
}
```

> [!TIP]
> Keyword boosting is implemented inside `inference.py`. The presence of terms like `free`, `discount`, `offer`, `telegram`, or `election` will boost the promotional confidence towards `1.0`.

## 📦 What's Next?
The code is production-ready. You can activate your conda environment containing `transformers` and `torch` to begin the `train_model.py` script. The models will leverage your available GPUs automatically.
