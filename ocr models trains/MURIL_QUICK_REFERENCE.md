# MuRIL Local Training - Quick Reference

## 🚀 TL;DR

```bash
# 1. Setup
cd moderation-service
pip install -r requirements-muril-training.txt

# 2. Add datasets to: datasets/muril_training/
# (or use dry-run for testing)

# 3. Train
python muril_local_train.py

# 4. Done! Model in: models/muril_abuse_final/
```

## Key Differences from Colab Version

| Aspect | Colab | Local (80GB) |
|--------|-------|------------|
| **Google Drive** | `drive.mount()` | ❌ Removed (use local paths) |
| **Colab file picker** | `files.upload()` | ❌ Removed (add to `datasets/muril_training/`) |
| **Batch Size** | 32 | **128** (4×) |
| **Gradient Accumulation** | 1 | **2** (effective batch 256) |
| **Flash Attention** | ❌ | ✅ Enabled |
| **Gradient Checkpointing** | ❌ | ✅ Enabled |
| **Data Workers** | 2 | **8** |
| **Training Time** | ~60 min | **~14 min** |

## Commands

```bash
# Full training (default)
python muril_local_train.py

# Dry-run (test setup, 100 samples)
python muril_local_train.py --dry-run

# Custom dataset path
python muril_local_train.py --data-dir "C:\path\to\datasets"

# More epochs
python muril_local_train.py --epochs 7

# Larger batches (if you have VRAM headroom)
python muril_local_train.py --batch-size 256

# All together
python muril_local_train.py --dry-run --epochs 3 --batch-size 64
```

## Dataset Setup

```
datasets/
└── muril_training/
    ├── hasoc_2021_hindi_train.tsv          (optional)
    ├── constraint_hindi_train.csv          (optional)
    ├── tamil_offensive_train.csv           (optional)
    ├── malayalam_offensive_train.csv       (optional)
    ├── hinglish_profanity.csv              (optional)
    └── hinglish_hate_speech.csv            (optional)
```

**Note:** All optional. Script works with any subset. Empty folder = demo synthetic data.

## Output

```
models/
└── muril_abuse_final/
    ├── pytorch_model.bin          # 950 MB weights
    ├── config.json
    ├── tokenizer.json
    ├── test_metrics.json          # Your results!
    ├── training_config.json
    └── checkpoints/               # Can delete to save space
```

## Use Trained Model

### Python
```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./models/muril_abuse_final"
)

result = classifier("teri ma ki aankh")
print(result)  # [{'label': 'abusive', 'score': 0.98}]
```

### With OCR
```python
from muril_local_train import moderate_ocr_output, load_trained_classifier
from pathlib import Path

classifier = load_trained_classifier(Path("./models/muril_abuse_final"))

ocr_lines = ["line 1", "line 2", "line 3"]
result = moderate_ocr_output(ocr_lines, classifier)

print(f"Image flagged: {result['flagged']}")
for line in result['lines']:
    print(f"  {line['text']} → {line['label']} ({line['score']:.0%})")
```

## Monitoring

### GPU Usage
```bash
nvidia-smi -l 1        # Update every 1 sec
```

### Training Metrics
```bash
tensorboard --logdir ./logs
# Open http://localhost:6006
```

### Logs
```
logs/training_YYYYMMDD_HHMMSS.log
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | `--batch-size 64` or `--disable-fp16` |
| Slow training | Check `nvidia-smi`, verify FP16 enabled |
| Dataset not found | Check `datasets/muril_training/` path or use `--data-dir` |
| GPU not detected | Verify PyTorch sees GPU: `python -c "import torch; print(torch.cuda.is_available())"` |
| Datasets don't exist | Use `--dry-run` to test, then add real data |

## Hardware Requirements

✅ **What You Have**: 80GB GPU (perfect for this)

| Component | Min | Recommended | Your Setup |
|-----------|-----|------------|-----------|
| GPU VRAM | 12GB | 40GB+ | 80GB ✅ |
| RAM | 16GB | 32GB+ | ? |
| Disk | 50GB | 100GB+ | ? |
| CUDA | 11.8 | 12.1 | ? |

## Performance Expectations

```
Dataset size: 5,000 samples
Effective batch: 256
FP16 + Flash Attention enabled

Training time:    ~14 minutes (5 epochs)
GPU utilization:  78-85% VRAM
Throughput:       ~600 samples/sec
Test F1 (abusive):~0.85
```

## Dataset Sources

| Name | Link | Size |
|------|------|------|
| HASOC 2021 | https://hasocfire.github.io/hasoc/2021/ | ~5K |
| CONSTRAINT | https://competitions.codalab.org/competitions/26654 | ~3K |
| DravidianLangTech | https://huggingface.co/datasets/DravidianLangTech | ~2K per lang |
| Kaggle Hinglish | https://www.kaggle.com/search?q=hinglish | ~10K |

## Next Steps

1. ✅ Run `python muril_local_train.py --dry-run` (verify setup)
2. 📥 Download datasets → `datasets/muril_training/`
3. 🚀 Run `python muril_local_train.py` (full training)
4. 📊 Check `models/muril_abuse_final/test_metrics.json`
5. 🔌 Integrate into moderation pipeline with `moderate_ocr_output()`

## Integration with Surya OCR

After training, use in [moderation-service/pipeline/ocr.py](../pipeline/ocr.py):

```python
# At module level
from pathlib import Path
from muril_local_train import moderate_ocr_output, load_trained_classifier

MURIL_CLASSIFIER = load_trained_classifier(
    Path(__file__).parent.parent / "models" / "muril_abuse_final"
)

# In your OCR result processing
def process_ocr_results(ocr_output):
    # ... existing Surya OCR processing ...
    
    # New: Run MuRIL classifier on text lines
    abuse_result = moderate_ocr_output(
        ocr_output.get("text_lines", []),
        MURIL_CLASSIFIER,
    )
    
    return {
        "text": ocr_output,
        "abuse_detection": abuse_result,
        "flagged": abuse_result["flagged"],
    }
```

---

**Full docs:** [MURIL_LOCAL_TRAINING.md](MURIL_LOCAL_TRAINING.md)
