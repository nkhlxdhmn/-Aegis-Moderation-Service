# Performance

Use the benchmark script to measure OCR latency, end-to-end latency, and throughput on your hardware.

```bash
python scripts/benchmark.py --image path/to/image.jpg --iterations 5
```

Metrics to record for release notes:

- OCR latency: average, p50, max.
- End-to-end latency: average, p50, max.
- Throughput: images per second.
- GPU utilization: sample with `nvidia-smi dmon` or a Prometheus GPU exporter.
- CPU utilization: sample with OS tools or a Prometheus node exporter.

Compare CPU and GPU by running the same image set with and without GPU access in Docker.
