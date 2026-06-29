# Deployment

## Docker Compose

```bash
docker compose up --build
```

Open `http://localhost:8000`.

The production Compose file runs one service, exposes port `8000`, keeps Hugging Face/model caches in Docker volumes, and uses `/api/v1/health` for health checks.

## Development Container

```bash
docker compose -f docker-compose.dev.yml up --build
```

## Local Python

```bash
make install
make run
```

## GPU Notes

The production image uses an NVIDIA CUDA runtime base image. Install the NVIDIA Container Toolkit on the host if you want GPU acceleration. CPU execution is still possible if model dependencies support it in your environment.

## Model Downloads

Model weights are not bundled in the repository. They are downloaded lazily by the underlying libraries into the configured cache volume on first use. Operators can also run a warmup script before serving high-traffic environments.
