# Release Checklist

- Tests pass.
- Ruff passes.
- Black check passes.
- Mypy passes or documented exceptions are accepted.
- Production Docker image builds.
- Development Docker image builds.
- `docker compose up` serves `http://localhost:8000`.
- `/api/v1/health` returns `ok`.
- Secret scan is clean.
- README is current.
- CHANGELOG is current.
- LICENSE is present.
- Version is bumped.
- Release notes are written.
- Release is tagged, for example `v1.0.0`.
