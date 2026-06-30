# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
uv lock && uv sync --all-groups
cp .env.example .env   # fill in GROQ_API_KEY_01 … and set TOTAL_GROQ_KEYS
```

## Commands

```bash
# Sanity check (no network calls)
uv run python scripts/smoke_test.py

# Run all tests with coverage
uv run pytest --cov=. --cov-report=term-missing

# Run a single test file
uv run pytest tests/unit/pool/test_key_pool.py

# Run a single test by name
uv run pytest tests/unit/pool/test_key_pool.py::test_function_name -v
```

## Environment variables

### API keys (required)

| Variable                             | Purpose                              |
|--------------------------------------|--------------------------------------|
| `TOTAL_GROQ_KEYS`                    | Number of keys configured (e.g. `3`) |
| `GROQ_API_KEY_01` … `GROQ_API_KEY_N` | Individual Groq API keys             |

### Behaviour tuning

| Variable                 | Default | Purpose                                          |
|--------------------------|---------|--------------------------------------------------|
| `GROQ_MAX_RPM`           | `30`    | Max requests/minute per key before penalty       |
| `GROQ_MAX_ACTIVE`        | `10`    | Max concurrent requests per key                  |
| `GROQ_COOLDOWN_SECS`     | `60`    | Cooldown duration after 429                      |
| `GROQ_FAILURE_THRESHOLD` | `5`     | Consecutive failures before key is marked FAILED |
| `GROQ_BATCH_CONCURRENCY` | `20`    | Semaphore limit for batch calls                  |
| `GROQ_MAX_RETRIES`       | `3`     | Max retry attempts                               |
| `GROQ_BASE_BACKOFF`      | `1.0`   | Base backoff seconds                             |
| `GROQ_MAX_BACKOFF`       | `30.0`  | Max backoff cap                                  |
| `GROQ_SESSION_TTL_HOURS` | `24`    | Session expiry                                   |
| `GROQ_DEBUG`             | `""`    | Set to `1`/`true` for debug logging              |
| `GROQ_LOG_LEVEL`         | `INFO`  | Logging level                                    |

### Local storage

Filesystem paths are hardcoded via `poolgate/core/paths.py:PathConfig` — no environment variables control path resolution.
The default data root is `<project_root>/data` (resolved at import time from
`PathConfig._DEFAULT_BASE_DIR`). Pass an explicit `PathConfig` to `GroqConfig` to override (e.g. in tests).

PoolGate creates the following layout under the data directory:

```
<data_dir>/
  tracking/
    usage.json      # daily request counts
    tokens.json     # per-model token usage
    account.json    # per-key usage
  requests/
    YYYY-MM-DD.jsonl  # one JSON line per request (execution details)
  logs/
    general.log       # all text log messages
    error.log         # ERROR and above
    request.log       # per-request start (JSON lines)
    response.log      # per-request outcome with tokens + latency (JSON lines)
    trace.log         # lifecycle stage events: key_acquired, retry, exhausted (JSON lines)
    tool_calls.log    # tool-calling invocations (JSON lines)
    performance.log   # latency and throughput metrics (JSON lines)
    storage.log       # persistence layer events: load, flush (JSON lines)
    debug.log         # DEBUG detail (only when GROQ_DEBUG=true)
```

## Architecture

PoolGate is a library (not a server) that pools multiple Groq API keys into one facade. The entry point for all usage is
`poolgate/services/provider.py:GroqService`.

### Request flow

1. **`GroqService`** (poolgate/services/provider.py) — public facade. Exposes `invoke`, `stream`, `structured`,
   `invoke_tools`, `moderate`, `transcribe`, `synthesize`, and their async/batch variants. It validates input, acquires
   a key, calls a capability client, records outcomes, and retries on failure.

2. **`RequestScheduler`** (poolgate/pool/scheduler.py) — `acquire_key(request_id, model=...)` picks an
   `APIKeyState` from the pool using a pluggable `SchedulingStrategy`. It enforces cooldowns and RPM limits. After the
   call, `release_key` or `mark_key_failure` is called to update per-key state.

3. **`KeyPool` / `APIKeyState`** (poolgate/pool/key_pool.py) — `APIKeyState` is the thread-safe per-key runtime state:
   sliding-window RPM/RPH/RPD counters, latency tracker, circuit-breaker (`consecutive_failures >= failure_threshold` →
   FAILED), and a composite `health_score()`. `KeyPool` is the collection.

4. **Capability clients** (poolgate/capabilities/) — one per Groq capability. All inherit `BaseGroqClient`
   (poolgate/providers/groq/client.py) which provides SDK construction, error-to-exception mapping, and parsing helpers.
   The capability ABCs in `poolgate/providers/groq/capabilities.py` define the contracts.

5. **`poolgate/providers/groq/models.py`** — per-model rate-limit configs (RPM/TPM/RPD). When a model ID is passed to
   `acquire_key`, the scheduler looks up the model's own RPM limit here and uses it instead of the global
   `max_rpm_per_key`, so a Whisper call and a Llama call are scored against their real Groq limits independently.

6. **`poolgate/tracking/`** — `TrackingManager` aggregates per-request stats into `UsageTracker`, `TokenTracker`,
   `AccountTracker`, and `QuotaTracker`. Auto-persisted to `<data_dir>/tracking/` when
   `config.paths.persistence_enabled`; `flush_tracking()` writes each tracker to its own JSON file.

7. **`poolgate/services/retry.py`** — `RetryPolicy` / `AsyncRetryPolicy` built on tenacity. Retryable: 429, 5xx,
   transport errors. Non-retryable: 401/403, 400, `APIKeyDisabledError`, `StructuredOutputError`,
   `NoAvailableAPIKeyError`.

### Path and logging infrastructure

**`poolgate/core/paths.py:PathConfig`** — Single source of truth for all filesystem paths. All modules must resolve
paths through `config.paths` rather than constructing them inline. Key properties: `tracking_dir`, `requests_dir`,
`log_dir`, and per-file paths (`general_log`, `request_log`, `response_log`, `trace_log`, `tool_calls_log`,
`performance_log`, `storage_log`, `error_log`).

**`poolgate/core/logger.py:LoggerManager`** — Centralized owner of all log handlers. Created per `GroqService` in
`__init__`. Provides `get()` for human-readable text logging and typed structured methods (`log_request`,
`log_response`, `log_trace`, `log_tool_call`, `log_performance`, `log_storage`) that write JSON lines to their dedicated
category files. No other module may create `RotatingFileHandler` instances or hard-code log file paths.

### Scheduling strategies

Six strategies live in `poolgate/pool/strategies/`, selectable via `SchedulingStrategyType`:

- `HEALTH_AWARE` (default) — composite health score (RPM penalty + active requests + latency + failure rate)
- `ROUND_ROBIN` — equal rotation
- `LEAST_USED` — lowest RPM/active wins
- `WEIGHTED_ROUND_ROBIN` — proportional to per-key `weight` attribute
- `LEAST_REMAINING_CAPACITY` — key with most unused RPM budget
- `PRIORITY_FAILOVER` — primary key first, then backup by priority

Swap at runtime: `scheduler.set_strategy(SchedulingStrategyType.ROUND_ROBIN)`.

### Exception hierarchy

All exceptions inherit from `GroqServiceError` (poolgate/exceptions/base.py). Import from `poolgate.exceptions`:

```
GroqServiceError
├── ConfigurationError → EnvironmentParseError, InvalidRateLimitConfigError, EmptyKeyPoolError
├── InvalidRequestError → MissingPromptError, InvalidMessageRoleError, UnknownModelError
├── APIKeyError → NoAvailableAPIKeyError, APIKeyDisabledError
├── RateLimitExceededError
├── QuotaExceededError → DailyLimitExceededError
├── TokenBudgetExceededError
├── TransportError → UpstreamTimeoutError
├── UpstreamServiceError, InvalidResponseError, RetryExhaustedError
├── StructuredOutputError
└── SessionError → SessionExpiredError
```

### Adding a new model

Add a class to `poolgate/providers/groq/models.py` following the pattern of an existing one — define RPM/TPM/RPD limits
and register it in the `MODEL_REGISTRY` dict and `get_model_config` function in the same file.

### Key safety invariant

Raw API keys (`APIKeyState.raw_key`) are never logged. All logging uses `masked_key` (format: `gsk_****abcd`), enforced
by `poolgate.core.logger.mask_key()`.
