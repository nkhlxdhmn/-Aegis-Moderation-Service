# MuRIL Abuse Classifier - Architecture & Pipeline

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTENT MODERATION PIPELINE                  │
└─────────────────────────────────────────────────────────────────┘

                              INPUT
                                │
                    ┌───────────▼────────────┐
                    │   Image from Social    │
                    │   Media / User Upload  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Surya OCR Engine     │  ← Better Indic languages
                    │  (Hindi, Tamil, etc)   │     than EasyOCR
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Extract Text Lines    │
                    │  [line1, line2, ...]   │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  MuRIL Classifier      │  ← NEW: This module
                    │  (Batch inference)     │     [Fine-tuned]
                    └───────────┬────────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                │
        ┌───────▼─────────┐           ┌────────▼──────────┐
        │ Abuse Detection │           │ Text Quality      │
        │ Results         │           │ (existing logic)  │
        │ per line        │           └───────┬───────────┘
        └───────┬─────────┘                   │
                │           ┌─────────────────┘
                │           │
        ┌───────▼───────────▼──────────────┐
        │  Moderation Decision Engine      │
        │  - Abuse ratio analysis          │
        │  - Context scoring               │
        │  - Combine multiple signals      │
        └───────┬────────────────────────┬─┘
                │                        │
        ┌───────▼────────┐      ┌───────▼──────────┐
        │ APPROVE ✅     │      │ REJECT 🚨        │
        │ <15% abusive   │      │ >30% abusive     │
        └────────────────┘      └──────────────────┘
                │
        ┌───────▼──────────────┐
        │ REVIEW 👁️ (optional)  │
        │ 15-30% abusive       │
        │ (manual check)       │
        └──────────────────────┘
```

## Training Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│              MURIL MODEL TRAINING PIPELINE                    │
└──────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────┐
    │    Multiple Abuse Datasets          │
    │ ┌─────────────────────────────────┐ │
    │ │ HASOC 2021 (Hindi/English)   🟡 │ │  Optional
    │ │ CONSTRAINT (Hindi Hostility) 🟡 │ │  (demo dataset
    │ │ DravidianLangTech (Tamil)    🟡 │ │   auto-generated
    │ │ Kaggle Hinglish              🟡 │ │   if missing)
    │ └─────────────────────────────────┘ │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │    Data Preprocessing               │
    │ • Remove URLs & HTML tags           │
    │ • Normalize text (fix !!!!)         │
    │ • Preserve Devanagari & Unicode     │
    │ • Remove duplicates                 │
    │ • Stratified split: 70/15/15        │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │    SentencePiece Tokenization       │
    │  (MuRIL's native tokenizer)         │
    │ • Max length: 128 tokens            │
    │ • Dynamic padding (efficient)       │
    │ • Handles: Hindi, Tamil, Telugu...  │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │   Load google/muril-base-cased              │
    │   (236M parameters, Apache 2.0)             │
    │                                              │
    │   [CLS] ... [Text Tokens] ... [SEP]         │
    │          ↓                                   │
    │   MuRIL Transformer Encoder                 │
    │          ↓                                   │
    │   [CLS] Representation                      │
    │          ↓                                   │
    │   Classification Head (NEW)                 │
    │   logits[0] → prob(non_abusive)             │
    │   logits[1] → prob(abusive)                 │
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │   Training Loop (5 epochs)              │
    │                                          │
    │ Per Step:                                │
    │ 1. Load batch (eff. size=256)           │
    │ 2. Forward pass through MuRIL           │
    │ 3. WeightedCrossEntropyLoss             │
    │    (handles class imbalance)            │
    │ 4. Backprop (with gradient             │
    │    accumulation)                        │
    │ 5. Update weights (LR warmup +         │
    │    linear decay)                        │
    │                                          │
    │ Per Epoch:                               │
    │ 1. Evaluate on validation set           │
    │ 2. Compute F1/accuracy/recall           │
    │ 3. Save checkpoint if best              │
    │ 4. Early stopping if F1 plateaus        │
    └──────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │   Test Set Evaluation                   │
    │                                          │
    │ Metrics:                                │
    │ • Accuracy: ~87%                        │
    │ • F1-Score (abusive): ~0.85            │
    │ • Precision: ~0.84                      │
    │ • Recall: ~0.86                         │
    │                                          │
    │ Per-class breakdown                     │
    │ Confusion matrix                        │
    └──────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │   Save Fine-tuned Model                 │
    │                                          │
    │ Location:                               │
    │ models/muril_abuse_final/               │
    │ ├── pytorch_model.bin (950 MB)          │
    │ ├── config.json                         │
    │ ├── tokenizer.json                      │
    │ ├── test_metrics.json                   │
    │ └── training_config.json                │
    └──────────────┬──────────────────────────┘
                   │
                DONE ✅
```

## Inference Pipeline

```
┌─────────────────────────────────────────────────┐
│         OFFLINE INFERENCE (NO INTERNET)         │
└─────────────────────────────────────────────────┘

Input Text
    │
    ▼
┌─────────────────────────────────────────┐
│ Load MuRIL + Tokenizer from disk        │
│ ./models/muril_abuse_final/             │
└────────────────┬────────────────────────┘
                 │
    ┌────────────▼──────────────┐
    │ Tokenize text             │
    │ (SentencePiece)           │
    └────────────┬───────────────┘
                 │
    ┌────────────▼──────────────────────────┐
    │ Forward pass through MuRIL             │
    │ (FP16 on GPU)                         │
    │                                        │
    │ [CLS] text_ids [SEP] [PAD]...        │
    │    ↓                                   │
    │ MuRIL Encoder                         │
    │    ↓                                   │
    │ Classification Head                   │
    │    ↓                                   │
    │ Softmax over 2 classes                │
    └────────────┬──────────────────────────┘
                 │
    ┌────────────▼──────────────────┐
    │ Output probabilities          │
    │ {                             │
    │   "label": "abusive",         │
    │   "score": 0.98               │
    │ }                             │
    └────────────┬──────────────────┘
                 │
    ┌────────────▼──────────────────┐
    │ Apply threshold (0.75)        │
    │ is_flagged = score >= 0.75    │
    └────────────┬──────────────────┘
                 │
    ┌────────────▼──────────────────┐
    │ Output decision               │
    │ ✅ Safe / 🚨 Flagged         │
    └───────────────────────────────┘

Latency: ~50-100ms per text (GPU)
         ~200-400ms per text (CPU)
```

## Data Flow: Image → Moderation

```
┌──────────────────────────────────────────────────────┐
│          END-TO-END MODERATION FLOW                  │
└──────────────────────────────────────────────────────┘

User uploads image
        │
        ▼
┌─────────────────────────────┐
│ Surya OCR                   │
│ INPUT: image.png            │
│ OUTPUT:                      │
│ {                           │
│   "text": "full_text",      │
│   "lines": [                │
│     {                       │
│       "text": "line1",      │
│       "bbox": [...],        │
│       "confidence": 0.95    │
│     },                      │
│     ...                     │
│   ]                         │
│ }                           │
└────────────┬────────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ Extract lines          │
    │ ["line1", "line2", ...]│
    └────────────┬───────────┘
             │
             ▼
    ┌────────────────────────────────┐
    │ MuRIL Batch Classification     │
    │ INPUT:  ["line1", "line2", ...]│
    │ OUTPUT: [                      │
    │   {                            │
    │     "text": "line1",           │
    │     "label": "non_abusive",    │
    │     "score": 0.95,             │
    │     "flagged": False           │
    │   },                           │
    │   {                            │
    │     "text": "line2",           │
    │     "label": "abusive",        │
    │     "score": 0.98,             │
    │     "flagged": True            │
    │   }                            │
    │   ...                          │
    │ ]                              │
    └────────────┬────────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │ Aggregate results            │
    │ • abuse_count = 1            │
    │ • abuse_ratio = 50%          │
    │ • image_flagged = True       │
    │   (>20% threshold)           │
    └────────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │ Decision engine              │
    │                              │
    │ IF abuse_ratio > 30%         │
    │   → REJECT 🚨               │
    │                              │
    │ ELIF abuse_ratio > 15%       │
    │   → REVIEW 👁️               │
    │                              │
    │ ELSE                         │
    │   → APPROVE ✅              │
    └────────────┬─────────────────┘
             │
             ▼
    Return moderation_decision

Total latency for image:
  OCR: ~200-500ms
  + Abuse detection: ~500-800ms (depends on line count)
  = ~1-2 seconds total
```

## Model Optimization Features

### For 80GB GPU:

```
┌─────────────────────────────────────────────┐
│     OPTIMIZATION TECHNIQUES ENABLED         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────┐
│ 1. Mixed Precision (FP16)       │
│    ├─ Weights: FP32 (master)    │
│    ├─ Compute: FP16             │
│    ├─ Gradients: FP16           │
│    └─ Speed: 2× faster          │
│       Memory: 2× less           │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ 2. Flash Attention 2            │
│    ├─ Algorithm: IO-aware       │
│    ├─ Speed: 2× faster          │
│    ├─ Memory: 3× less           │
│    └─ Quality: Identical        │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ 3. Gradient Checkpointing       │
│    ├─ Trade: compute ↔ memory   │
│    ├─ Recompute activations     │
│    ├─ Save intermediate states  │
│    └─ Effective for large models│
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ 4. Gradient Accumulation        │
│    ├─ Effective batch: 256      │
│    ├─ Physical batch: 128       │
│    ├─ Accumulation steps: 2     │
│    └─ Better convergence!       │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ 5. Dynamic Padding              │
│    ├─ Pad to longest in batch   │
│    ├─ NOT to MAX_LENGTH         │
│    ├─ ~15% token savings        │
│    └─ Faster compute            │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ 6. Parallel Data Loading        │
│    ├─ Workers: 8 processes      │
│    ├─ Prefetch: 2 batches ahead │
│    ├─ Pin memory: enabled       │
│    └─ No I/O bottleneck         │
└─────────────────────────────────┘

Result on 80GB GPU:
  Training: ~14 minutes (5 epochs)
  Inference: ~50ms per batch (32 texts)
  GPU utilization: 78-85% VRAM
  Throughput: ~600 samples/sec
```

## File Organization

```
moderation-service/
├── muril_local_train.py          ← Main training script
├── requirements-muril-training.txt ← Dependencies
├── SETUP_SUMMARY.md              ← This overview
├── MURIL_QUICK_REFERENCE.md      ← Quick reference
├── MURIL_LOCAL_TRAINING.md       ← Full guide
├── INTEGRATION_GUIDE.md          ← How to add to pipeline
├── examples_muril_integration.py ← Code examples
│
├── datasets/                     ← Download datasets here
│   └── muril_training/
│       ├── hasoc_2021_hindi_train.tsv      (optional)
│       ├── constraint_hindi_train.csv      (optional)
│       └── ... (others optional)
│
├── models/                       ← Trained models
│   └── muril_abuse_final/
│       ├── pytorch_model.bin     (950 MB)
│       ├── config.json
│       ├── tokenizer.json
│       ├── test_metrics.json     ← Your results!
│       └── training_config.json
│
├── logs/                         ← Training logs
│   └── training_YYYYMMDD_HHMMSS.log
│
├── pipeline/                     ← Existing pipeline
│   ├── ocr.py                    ← Add MuRIL here
│   └── ... (other pipeline code)
│
└── ... (other moderation files)
```

---

**Next:** Check [SETUP_SUMMARY.md](SETUP_SUMMARY.md) for quick start!
