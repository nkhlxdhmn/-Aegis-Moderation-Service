# Deployment Guide

Aegis Moderation is fully containerized. 

## Docker Configurations

The repository provides two separate Docker pipelines:

1. **CPU Default**: `Dockerfile` + `docker-compose.yml`
2. **GPU Optimized**: `Dockerfile.gpu` + `docker-compose.gpu.yml`

## 1. Deploying on CPU (Default)

Use this for local development, testing, or environments without NVIDIA GPUs (e.g., standard EC2, DigitalOcean droplets, Windows with Docker Desktop).

```bash
docker compose up --build -d
```

*Note: Inference on CPU is slower. Image processing takes ~2-5s instead of ~200ms.*

## 2. Deploying on GPU (Production)

Use this for production environments. Requires a CUDA-capable NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose -f docker-compose.gpu.yml up --build -d
```

### Pre-warming Models

In production, you can pre-warm models so the first request doesn't suffer cold-start latency. Keep this disabled for local Docker or small hosts; eager warmup can exceed available RAM/VRAM and cause container exit code 137.

Edit `docker-compose.gpu.yml`:
```yaml
environment:
  MODEL_WARMUP: "true"
```

The container healthcheck will report `unhealthy` until all models are successfully loaded into VRAM.

## 3. Scaling

Aegis Moderation is stateless. You can run multiple replicas behind a load balancer (e.g., NGINX, ALB, Traefik). 

Since there is no database or shared state, horizontal scaling is trivial:

```bash
docker compose -f docker-compose.gpu.yml up --scale aegis-moderation=3 -d
```
