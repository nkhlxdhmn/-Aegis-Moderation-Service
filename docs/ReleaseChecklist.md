# Release Checklist

Before tagging a new release, verify the following:

- [ ] Run `make lint` (Ruff, Black, Mypy) — 0 errors.
- [ ] Run `make test` (Pytest) — 100% pass.
- [ ] Run `python backend/validate_models.py` — All checks pass.
- [ ] Verify `docker-compose.yml` (CPU) builds successfully.
- [ ] Verify `docker-compose.gpu.yml` (GPU) builds successfully.
- [ ] Ensure frontend `index.html` and `dashboard.html` load without console errors.
- [ ] Check `.env.example` has all current config values documented.
- [ ] Secret scan is clean.
- [ ] Update version strings in `backend/main.py`, `docker-compose*.yml`, and `pyproject.toml`.
- [ ] Update `CHANGELOG.md`.
