# Contributing

Thanks for helping improve Aegis Moderation.

---

## Development Setup

```bash
git clone https://github.com/your-org/aegis-moderation.git
cd aegis-moderation

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Before You Start

- Open an issue or draft PR to discuss changes before writing substantial code.
- Preserve existing AI model and pipeline behaviour — changes to `backend/pipeline/` require test coverage.
- Do not introduce Supabase, Redis, PostgreSQL, or any external service dependencies.

---

## Running Tests

```bash
# Core endpoint + report tests (fast, no models required)
python -m pytest tests/test_main.py tests/test_standalone_report.py -v

# Full test suite (requires model stack)
python -m pytest -v
```

---

## Lint & Format

```bash
ruff check .
black --check .
mypy backend/ --ignore-missing-imports --no-error-summary

# Auto-fix formatting
black .
ruff check . --fix
```

All CI checks (tests, ruff, black, mypy, Docker build) must pass before a PR can be merged.

---

## Adding a New Moderation Pipeline Stage

1. Create or extend a module under `backend/pipeline/`.
2. Import and call it from the appropriate orchestrator (`safety_flags.py`, `text_moderation.py`, or `video_moderation.py`).
3. Map any new signal to an existing category in `backend/reports.py` (or add a category with a clear taxonomy label).
4. Write a unit test that mocks the model call and verifies the report output.

---

## Pull Request Guidelines

- Keep PRs focused: one feature or bug fix per PR.
- Include a short description of the change and why it is needed.
- Reference any related issue with `Closes #N`.
- Ensure `python -m pytest tests/test_main.py tests/test_standalone_report.py -v` passes.

---

## Commit Style

```
feat: short imperative description
fix: what was broken and how it is fixed
docs: what was documented
chore: dependency bump, cleanup
```

No period at the end of the subject line. Keep the subject under 72 characters.
