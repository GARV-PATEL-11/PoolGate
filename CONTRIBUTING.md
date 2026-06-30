# Contributing to PoolGate

Thank you for contributing! This guide covers everything from a fresh clone to a merged pull request.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Repository Hygiene — First-Time Fix](#repository-hygiene--first-time-fix)
- [Project Layout](#project-layout)
- [Coding Conventions](#coding-conventions)
- [Running Tests](#running-tests)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Branch & Commit Conventions](#branch--commit-conventions)
- [Pull Request Checklist](#pull-request-checklist)
- [Reporting Issues](#reporting-issues)

---

## Getting Started

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/PoolGate.git
cd PoolGate

# 2. Install all dependencies (Python 3.10+, uv required)
#    https://docs.astral.sh/uv/getting-started/installation/
uv sync --group dev

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Copy the env template
cp .env.example .env
# → Fill in at least one GROQ_API_KEY_* for integration / e2e tests
```

> **Alternative (plain pip):** `python -m venv .venv && source .venv/bin/activate && pip install -e . && pip install pytest pytest-asyncio pytest-mock httpx pytest-cov`

---

## Repository Hygiene — First-Time Fix

Several generated artefacts were accidentally committed before `.gitignore` was fully enforced.
Run this once on a fresh clone to clean up the tracking index:

```bash
# Remove artefacts already tracked by git but covered by .gitignore
git rm -r --cached htmlcov/
git rm -r --cached data/
git rm --cached coverage.xml
git rm -r --cached __pycache__/       # root-level only; sub-package ones too if present

git commit -m "chore: untrack generated artefacts (htmlcov, data, coverage.xml)"
```

After this, git will no longer track those paths, even if the local files still exist.

---

## Project Layout

| Package                     | Role                                                                                         |
|-----------------------------|----------------------------------------------------------------------------------------------|
| `poolgate/core/`            | Config, path resolution, structured logging                                                  |
| `poolgate/capabilities/`    | Public multi-modal API surface (chat, tool, structured, moderation, …)                       |
| `poolgate/providers/`       | Groq SDK adapter, model rate-limit configs                                                   |
| `poolgate/services/`        | Orchestration — GroqService, RetryService, HealthService                                     |
| `poolgate/pool/`            | Key selection — RequestScheduler + 6 SchedulingStrategy variants                             |
| `poolgate/persistence/`     | Session management and snapshot persistence                                                  |
| `poolgate/tracking/`        | Usage, token, quota, account tracking + persistence                                          |
| `poolgate/schemas/`         | Pydantic v2 service-boundary contracts                                                       |
| `poolgate/exceptions/`      | Typed hierarchy rooted at GroqServiceError                                                   |
| `poolgate/utils.py`         | SlidingWindowCounter + shared helpers                                                        |
| `examples/`                 | 13 runnable usage scripts                                                                    |
| `scripts/`                  | smoke_test.py — quick sanity check                                                           |
| `tests/`                    | pytest suite mirroring the poolgate/ tree                                                    |

---

## Coding Conventions

### Style

- **PEP 8** — enforced by `black` (line length 100)
- **`from __future__ import annotations`** — at the top of every module
- **Type hints** on every function and method signature
- **`black format`** for consistent formatting.

### Docstrings

Follow the module-level docstring convention already in the codebase:

```python
"""
module_name.py
──────────────
One-sentence summary.

Explain *why* this module exists — its design decision, not just what it imports.
"""
```

Public classes and functions use concise plain-English docstrings.

### Thread safety

All mutable shared state **must** be guarded by `threading.Lock`. See `poolgate/tracking/usage.py`
for the standard pattern.

### Key masking

Raw API keys (`gsk_...`) must **never** appear in logs, exception messages, or serialised state.
Always call `poolgate.core.logger.mask_key()` before any output.

### Exceptions

Raise from the typed hierarchy in `poolgate/exceptions/`. Never raise bare `ValueError` or `RuntimeError`
from library code — callers must be able to catch specific types.

### Path handling

All filesystem paths are constructed via `PathConfig` (`poolgate/core/paths.py`). No `os.path.join()`
or f-string paths anywhere in source code.

### Model rate limits

When adding a new Groq model, add a `ModelRateLimitConfig` subclass in `poolgate/providers/groq/models.py`,
register it in the `MODEL_REGISTRY` dict and `get_model_config` function in the same file, and add a
corresponding unit test in `tests/unit/providers/groq/test_models.py`.

---

## Running Tests

```bash
# Full suite
uv run pytest

# With coverage
uv run pytest --cov=. --cov-report=term-missing

# Unit tests only — no Groq credentials required
uv run pytest tests/unit/ -v

# Client-layer tests
uv run pytest tests/providers/ -v

# Cross-module integration tests (mock SDK — no real key)
uv run pytest tests/integration/ -v

# Full lifecycle e2e tests — requires GROQ_API_KEY_* in .env
uv run pytest tests/e2e/ -v -m e2e

# Single file
uv run pytest tests/unit/pool/test_key_pool.py -v
```

**Test markers:**

| Marker   | Layer                   | Needs real key? |
|----------|-------------------------|-----------------|
| *(none)* | unit                    | ❌               |
| *(none)* | providers / integration | ❌ (mock SDK)    |
| `e2e`    | end-to-end              | ✅               |

Unit and provider/integration tests must be fully offline — use `monkeypatch` or `pytest-mock`
to stub the Groq SDK. Never make real API calls in unit or integration tests.

---

## Pre-commit Hooks

The `.pre-commit-config.yaml` runs on every `git commit`:

| Hook                        | What it checks                                            |
|-----------------------------|-----------------------------------------------------------|
| `black`                     | Linting — imports, unused vars, style violations, bugbear |
| `black-format`              | Formatting consistency                                    |
| `mypy --strict`             | Static type checking                                      |
| `trailing-whitespace`       | No trailing spaces                                        |
| `end-of-file-fixer`         | Files end with `\n`                                       |
| `check-yaml` / `check-toml` | Syntax validity                                           |
| `detect-private-key`        | **Blocks accidental `gsk_` commits**                      |
| `commitizen`                | Conventional Commits format on commit messages            |

Run all hooks manually:

```bash
uv run pre-commit run --all-files
```

---

## Branch & Commit Conventions

### Branch names

| Prefix      | Use for                                |
|-------------|----------------------------------------|
| `feat/`     | New feature                            |
| `fix/`      | Bug fix                                |
| `refactor/` | Internal refactor, no behaviour change |
| `docs/`     | Documentation only                     |
| `test/`     | New or updated tests                   |
| `chore/`    | Tooling, CI, dependency bumps          |

Example: `feat/async-key-pool`, `fix/jsonl-trailing-newline`

### Commit messages — Conventional Commits

```
<type>(<scope>): <short imperative summary>

[optional body]

[optional footer — Fixes #123]
```

Scopes map to packages: `tracking`, `capabilities`, `services`, `pool`, `persistence`,
`providers`, `schemas`, `exceptions`, `core`, `tests`, `ci`, `docs`

Examples:

```
feat(pool): add least-remaining-capacity strategy
fix(tracking): handle trailing newlines in JSONL persistence load
docs(readme): add full project structure and architecture diagram
test(capabilities): add unit tests for StructuredCapability JSON validation
chore(ci): add GitHub Actions test workflow with uv
```

---

## Pull Request Checklist

- [ ] All tests pass: `uv run pytest`
- [ ] Coverage has not regressed: `uv run pytest --cov=. --cov-report=term-missing`
- [ ] Pre-commit hooks pass: `uv run pre-commit run --all-files`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Type hints on all new functions / methods
- [ ] No raw API keys in the diff (verified by `detect-private-key` hook)
- [ ] `data/`, `htmlcov/`, `coverage.xml` not in the diff
- [ ] PR description explains *what* changed and *why*
- [ ] New models added to `poolgate/providers/groq/models.py` have a corresponding test

---

## Reporting Issues

Use [GitHub Issues](https://github.com/GARV-PATEL-11/PoolGate/issues).

Please include:

- Python version (`python --version`) and uv version (`uv --version`)
- PoolGate commit hash (`git rev-parse --short HEAD`)
- Minimal reproduction steps
- Full traceback with API keys redacted (`gsk_****abcd` form)
