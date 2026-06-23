# Changelog

All notable changes to PoolGate are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Fixed

- Trailing commas in generator `for`-clauses in `core/logger_manager.py` that caused a `SyntaxError` on Python 3.14 (blocked all tests).
- `_fh` type annotation in `LoggerManager` corrected from `str` to `Path`.
- `CapabilityError` added to the exception hierarchy docstring in `exceptions/__init__.py`.
- `groq_config` test fixture now declares `TOTAL_GROQ_KEYS=2` to match the two keys it actually provides.
- All `llm_models/` docstrings corrected to use the full env-var names (`_REQUESTS_PER_MINUTE`, `_TOKENS_PER_MINUTE`, etc.) that `ModelRateLimitConfig.from_env()` actually reads.
- `CLAUDE.md` path documentation corrected: constant is `_DEFAULT_BASE_DIR`, path is `<project_root>/poolgate_data`.

### Added

- GitHub Actions CI workflow (`.github/workflows/ci.yml`) running tests on Python 3.10–3.13.
- `CHANGELOG.md` (this file), required by the CONTRIBUTING.md PR checklist.
- `[tool.black]` and `[tool.mypy]` sections in `pyproject.toml` for consistent local tooling.
- `poolgate_data/` added to `.gitignore` (previously only its subdirectories were listed).

### Changed

- `pre-commit-config.yaml` renamed to `.pre-commit-config.yaml` so `pre-commit install` finds it.
- `.env.example` comments clarify that values shown are recommended starting points, not built-in defaults.
- `services/health_service.py` gains `from __future__ import annotations` to match the rest of the codebase.
- `CONTRIBUTING.md` plain-pip alternative corrected to use `uv pip install -e . --group dev`.

---

## [0.1.1] — 2026-06-23

### Added

- EditorConfig (`.editorconfig`) with per-file-type formatting rules.
- CONTRIBUTING.md guide covering setup, conventions, test instructions, and PR checklist.

### Changed

- Max line length in EditorConfig adjusted to 100 for non-Python files; 120 for Python.
- Added `.claude` to `.gitignore`.

---

## [0.1.0] — 2026-06-18

Initial release of PoolGate.

### Added

- `GroqService` public facade with `invoke`, `chat`, `stream`, `structured`, `invoke_tools`, `moderate`, `transcribe`, `synthesize` and async/batch variants.
- `RequestScheduler` with six pluggable `SchedulingStrategy` implementations.
- `KeyPool` / `APIKeyState` with thread-safe sliding-window RPM/RPH/RPD counters and circuit-breaker.
- `TrackingManager` with `UsageTracker`, `TokenTracker`, `AccountTracker`, `QuotaTracker`.
- `LoggerManager` with structured JSON-line category log files and `mask_key()` safety.
- `PathConfig` as the single source of truth for all filesystem paths.
- Per-model rate-limit configs for 17 Groq models in `llm_models/`.
- `RetryPolicy` / `AsyncRetryPolicy` built on tenacity.
- `SessionManager` with TTL-based session expiry.
- `PersistenceService` with JSON and SQLite backends.
- Full exception hierarchy rooted at `GroqServiceError`.
- 632-test suite across unit, integration, providers, and e2e layers.
