# Performance and Tuning

Aegis Moderation runs multiple deep learning models per request. Optimizing hardware and configuration is critical for production throughput.

## GPU Requirements

- **Minimum:** 8GB VRAM (e.g., RTX 3070, T4)
- **Recommended:** 16GB+ VRAM (e.g., RTX 4080, A10G)

The pipeline loads the following into VRAM simultaneously:
- YOLO11x (Object Detection)
- SigLIP2 (Vision Embeddings)
- Surya OCR (Vision Encoder)
- BLIP (Image Captioning)
- Detoxify (Text Moderation)

## Multi-GPU setups

If you have multiple smaller GPUs (e.g., 2x 8GB), you can split the load by assigning the heaviest models (BLIP and Whisper) to the second GPU.

Set in your `.env`:
```ini
CUDA_VISIBLE_DEVICES=0,1
VLM_DEVICE=cuda:1
```

## Concurrency

The FastAPI server uses `ThreadPoolExecutor` to handle blocking ML inference.
By default, FastAPI allows 40 concurrent threads. 

If you experience GPU Out-Of-Memory (OOM) errors during traffic spikes, you can restrict concurrency in `backend/main.py` or scale horizontally across multiple containers.

## Benchmarking

A benchmarking script is included to test your local hardware:

```bash
# Test full pipeline latency
python scripts/benchmark.py --image examples/image.jpg

# Test OCR isolated latency
python scripts/benchmark.py --image examples/image.jpg --ocr-only
```
