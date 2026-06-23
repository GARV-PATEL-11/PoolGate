# PoolGate

> **Rate-aware API key pooling and orchestration for the Groq API.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PoolGate is an intelligent API orchestration and pooling system that aggregates multiple Groq API keys into a **unified, rate-aware gateway**. It automatically manages key rotation, request scheduling, usage tracking, and failover — maximising throughput and reliability while eliminating rate-limit interruptions.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Scheduling Strategies](#scheduling-strategies)
- [Supported Models](#supported-models)
- [Tracking System](#tracking-system)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Capability | Description |
|---|---|
| **Key Pooling** | Aggregate N Groq API keys behind a single client interface |
| **6 Scheduling Strategies** | Round-robin, least-used, weighted, least-remaining-capacity, priority-failover, health-score |
| **Rate-Aware Routing** | Per-key RPM / RPH / RPD sliding-window counters prevent 429s before they happen |
| **Automatic Failover** | Keys that hit thresholds cool down; healthy keys absorb traffic instantly |
| **Multi-Modal Clients** | Chat, tool-calling, structured output, moderation, transcription, synthesis |
| **15 Model Configs** | Per-model rate-limit envelopes (Llama, Qwen, Whisper, Compound, PromptGuard, …) |
| **Token Tracking** | Rolling-window + calendar-day views for input/output tokens per model |
| **Quota Mirroring** | Reflects Groq's `x-ratelimit-remaining-*` headers — provider truth beats local estimates |
| **Service Layer** | Health checks, retry with tenacity, session management, persistence |
| **Structured Logging** | JSON log lines per category; `mask_key()` ensures raw keys never hit disk |
| **Pluggable Persistence** | JSON (JSONL) or SQLite backends; survives process restarts |
| **Thread + Async Safe** | All counters use `threading.Lock`; `RequestScheduler` exposes both sync and async acquisition |
| **Zero-Friction Config** | Single `.env` file; per-model limits overridable via env vars |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Your Application                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │  chat / tool / structured /
                               │  moderation / transcription / synthesis
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    clients/  (public API surface)               │
│  ChatClient  ToolClient  StructuredClient  ModerationClient     │
│  TranscriptionClient  SynthesisClient  →  registry.py           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     services/  (orchestration)                  │
│   GroqService  ─┬─▶  RetryService  (tenacity backoff)           │
│                 ├─▶  SessionManager  (TTL-based sessions)       │
│                 ├─▶  HealthService  (key health scoring)        │
│                 └─▶  PersistenceService  (load / save state)    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               schedulers/  (key selection)                      │
│   RequestScheduler  ──▶  SchedulingStrategy (6 variants)        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                   ┌───────────┴────────────┐
                   ▼                        ▼
┌──────────────────────────┐  ┌──────────────────────────────────┐
│    key_manager/          │  │   tracking/                      │
│  KeyPool  APIKeyState    │  │  UsageTracker     TokenTracker   │
│  SlidingWindowCounter    │  │  QuotaTracker     AccountTracker │
│  (RPM / RPH / RPD)       │  │  RequestTracker   RollingWindow  │
|                          |  │  TrackingManager  Persistence    │
└──────────────────────────┘  └──────────────────────────────────┘
                  │                        │
                  └───────────┬────────────┘
                              │
                              │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   poolgate_data/  (runtime I/O)                 │
│   logs/  ·  tracking/*.json  ·  requests/*.jsonl  ·  audio/     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
PoolGate/
│
├── core/                          # Central config, path resolution, logging
│   ├── config.py                  # GroqConfig — all settings from env vars
│   ├── path_config.py             # PathConfig — single source of truth for FS paths
│   └── logger_manager.py          # Structured JSON logging + mask_key()
│
├── clients/                       # Public multi-modal API surface
│   ├── base.py                    # BaseGroqClient — shared lifecycle logic
│   ├── capabilities.py            # Per-model capability checks
│   ├── registry.py                # Client registry and model → client routing
│   ├── chat_client.py             # Chat completions (streaming + non-streaming)
│   ├── tool_client.py             # Tool / function calling
│   ├── structured_client.py       # JSON-mode structured generation
│   ├── moderation_client.py       # Content moderation
│   ├── transcription_client.py    # Whisper audio transcription
│   └── synthesis_client.py        # Orpheus TTS synthesis
│
├── services/                      # Orchestration layer
│   ├── provider_service.py        # GroqService — main facade; DEFAULT_MODEL
│   ├── retry_service.py           # RetryService — tenacity-based backoff
│   ├── session_service.py         # SessionManager, SessionUsageTracker
│   ├── health_service.py          # HealthService — key health scoring
│   └── persistence_service.py     # PersistenceService — load/save tracker state
│
├── schedulers/                    # Key selection and dispatch
│   ├── request_scheduler.py       # RequestScheduler (sync + async acquire/release)
│   └── scheduling_strategies.py  # 6 strategies: RoundRobin, LeastUsed, Weighted,
│                                  #   LeastRemainingCapacity, PriorityFailover, HealthScore
│
├── key_manager/                   # Per-key runtime state
│   └── key_pool.py                # APIKeyState (RPM/RPH/RPD counters), KeyPool
│
├── llm_models/                    # Per-model rate-limit envelopes
│   ├── base.py                    # ModelRateLimitConfig — env-overridable base class
│   ├── llama_3_1_8b_instant.py
│   ├── llama_3_3_70b_versatile.py
│   ├── meta_llama_4_scout.py
│   ├── meta_llama_prompt_guard_22m.py
│   ├── meta_llama_prompt_guard_86m.py
│   ├── qwen3_32b.py
│   ├── qwen3_6_27b.py
│   ├── groq_compound.py
│   ├── groq_compound_mini.py
│   ├── whisper_large_v3.py
│   ├── whisper_large_v3_turbo.py
│   ├── openai_gpt_oss_20b.py
│   ├── openai_gpt_oss_120b.py
│   ├── openai_gpt_oss_safeguard_20b.py
│   ├── allam_2_7b.py
│   ├── canopylabs_orpheus_v1_english.py
│   └── canopylabs_orpheus_arabic_saudi.py
│
├── tracking/                      # Usage, token, quota, account tracking
│   ├── rolling_window.py          # RollingWindowCounter — sliding-window primitive
│   ├── usage_tracker.py           # Lifetime totals + DailyBucket history
│   ├── token_tracker.py           # Rolling TPM/TPD + calendar-day per model
│   ├── quota_tracker.py           # Provider-reported x-ratelimit-remaining-* snapshots
│   ├── account_tracker.py         # Per-key rolling 24h usage for rotation scoring
│   ├── request_tracker.py         # Per-key RPM/RPH/RPD enforcement
│   ├── models.py                  # DailyBucket, TokenUsage value types
│   ├── persistence.py             # JSONPersistence (JSONL), SQLitePersistence
│   └── manager.py                 # TrackingManager — single facade for all trackers
│
├── schemas/                       # Pydantic v2 service-boundary contracts
│   ├── runtime.py                 # APIKeyStatus, FinishReason, ChatMessage, …
│   ├── chat.py                    # ChatRequest, ChatResponse
│   ├── structured.py              # StructuredRequest, StructuredResponse
│   ├── moderation.py              # ModerationRequest, ModerationResponse
│   ├── transcription.py           # TranscriptionRequest, TranscriptionResponse
│   ├── synthesis.py               # SynthesisRequest, SynthesisResponse
│   ├── model_info.py              # ModelInfo, ModelCapabilities, Capability
│   ├── common.py                  # Metadata, FinishReason, utcnow
│   ├── usage.py                   # TokenUsage, UsageSummary
│   ├── keys.py                    # APIKey identity schema
│   ├── context.py                 # RequestContext
│   ├── envelope.py                # Request/response envelopes
│   └── ops.py                     # Operational / health schemas
│
├── exceptions/                    # Typed exception hierarchy
│   ├── base.py                    # GroqServiceError (root)
│   ├── configuration.py           # ConfigurationError, EmptyKeyPoolError, …
│   ├── keys.py                    # APIKeyError, NoAvailableAPIKeyError, …
│   ├── rate_limit.py              # RateLimitExceededError, QuotaExceededError, …
│   ├── request.py                 # InvalidRequestError, MissingPromptError, …
│   ├── response.py                # InvalidResponseError
│   ├── output.py                  # StructuredOutputError
│   ├── transport.py               # TransportError, UpstreamTimeoutError, …
│   └── persistence.py             # PersistenceError
│
├── tests/                         # 53-file pytest suite
│   ├── conftest.py                # Root shared fixtures
│   ├── unit/                      # 35 files — fully offline, no Groq key required
│   ├── integration/               # 5 files — cross-module, mock SDK
│   ├── providers/                 # 6 files — client-layer tests
│   ├── e2e/                       # 7 files — full lifecycle (requires real key)
│   ├── fixtures/
│   └── mocks/
│
├── examples/                      # 13 runnable usage examples
│   ├── basic_chat.py
│   ├── async_chat.py
│   ├── streaming.py
│   ├── tool_calling.py
│   ├── structured_output.py
│   ├── moderation.py
│   ├── transcription.py
│   ├── synthesis.py
│   ├── sessions.py
│   ├── batch.py
│   ├── persistence.py
│   ├── error_handling.py
│   └── custom_scheduling.py
│
├── scripts/
│   └── smoke_test.py              # Quick sanity-check without full test suite
│
├── utils.py                       # SlidingWindowCounter + shared helpers
├── retry.py                       # RetryPolicy built on tenacity
│
├── poolgate_data/                 # ── RUNTIME DATA — never committed ──
│   ├── logs/                      # Structured JSON log files (8 categories)
│   ├── requests/                  # Per-day JSONL request records
│   ├── tracking/                  # usage.json, tokens.json, account.json
│   └── audio/                     # Synthesis audio output
│
├── .env.example                   # Copy → .env and fill in GROQ_API_KEY_*
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- One or more [Groq API keys](https://console.groq.com/keys)

### With uv (recommended)

```bash
git clone https://github.com/GARV-PATEL-11/PoolGate.git

cd PoolGate

# Install all dependencies including dev tools
uv sync --group dev
```

### With pip

```bash
git clone https://github.com/GARV-PATEL-11/PoolGate.git

cd PoolGate

python -m venv .venv && source .venv/bin/activate

pip install -e ".[dev]"
```

---

## Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# ── Required ─────────────────────────────────────────────────────────────────
TOTAL_GROQ_KEYS=3
GROQ_API_KEY_01=gsk_...
GROQ_API_KEY_02=gsk_...
GROQ_API_KEY_03=gsk_...

# ── Key pool behaviour (optional) ────────────────────────────────────────────
GROQ_MAX_RPM_PER_KEY=30
GROQ_MAX_ACTIVE_REQUESTS=10
GROQ_COOLDOWN_SECONDS=60
GROQ_FAILURE_THRESHOLD=5
GROQ_LATENCY_PENALTY_THRESHOLD=3.0

# ── Per-model rate-limit overrides (optional) ─────────────────────────────────
# LLAMA_3_3_70B_RPM=30
# LLAMA_3_3_70B_RPD=14400
```

> **Security:** `.env` and `poolgate_data/` are in `.gitignore`. Raw keys never touch logs — `mask_key()` converts them to `gsk_****abcd` before any I/O.

---

## Quick Start

```python
from core.config import GroqConfig
from services import GroqService

# Load config from environment
config = GroqConfig.from_env()

# Instantiate the main service facade
service = GroqService(config)

# Chat completion — key selection, rotation, retry all handled automatically
response = service.chat(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Explain sliding window rate limiting."}],
)
print(response.content)
```

See the [`examples/`](examples/) directory for 13 runnable scripts covering streaming, tool calling, structured output, moderation, transcription, batch requests, and more.

---

## Scheduling Strategies

Six strategies are available; configure via `GroqConfig` or pass directly to `RequestScheduler`:

| Strategy | Best for |
|---|---|
| `round_robin` | Even distribution, equal-capacity keys |
| `least_used` | Minimise per-key total request count |
| `weighted_round_robin` | Keys with different quotas / tiers |
| `least_remaining_capacity` | Avoid hot keys near their limit |
| `priority_failover` | Designate primary + standby keys |
| `health_score` (default) | Composite scoring — latency, error rate, remaining capacity |

```python
from schedulers.scheduling_strategies import SchedulingStrategyType

config.scheduling_strategy = SchedulingStrategyType.LEAST_REMAINING_CAPACITY
```

---

## Supported Models

PoolGate ships per-model `ModelRateLimitConfig` subclasses for all current Groq models. Each has plan-appropriate defaults and can be overridden at runtime via env vars.

| Family | Models |
|---|---|
| **Llama** | `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `meta-llama-4-scout` |
| **Llama Guard** | `meta-llama-prompt-guard-22m`, `meta-llama-prompt-guard-86m` |
| **Qwen** | `qwen3-32b`, `qwen3-6.27b` |
| **Groq Compound** | `groq-compound`, `groq-compound-mini` |
| **Whisper** | `whisper-large-v3`, `whisper-large-v3-turbo` |
| **OpenAI OSS** | `openai-gpt-oss-20b`, `openai-gpt-oss-120b`, `openai-gpt-oss-safeguard-20b` |
| **Allam** | `allam-2-7b` |
| **Orpheus** | `canopylabs-orpheus-v1-english`, `canopylabs-orpheus-arabic-saudi` |

---

## Tracking System

PoolGate tracks usage across four orthogonal dimensions:

| Tracker | Question answered | Time model |
|---|---|---|
| `UsageTracker` | "What did we send overall / today?" | Calendar day (UTC midnight) |
| `TokenTracker` | "How many tokens in the last minute / day?" | Rolling window (anchored to *now*) |
| `QuotaTracker` | "What does Groq say remains?" | Snapshot — last response header wins |
| `AccountTracker` | "Which key is least loaded right now?" | Rolling 24 h per key |
| `RequestTracker` | "Is this key within its RPM/RPH/RPD limits?" | Sliding window per key |

All state is held in memory by default. `PersistenceService` writes/reads `poolgate_data/tracking/*.json` via the pluggable `JSONPersistence` or `SQLitePersistence` backend.

---

## Examples

```bash
# Basic chat
uv run python examples/basic_chat.py

# Async with streaming
uv run python examples/async_chat.py

# Tool calling
uv run python examples/tool_calling.py

# JSON-mode structured output
uv run python examples/structured_output.py

# Custom scheduling strategy
uv run python examples/custom_scheduling.py
```

---

## Testing

```bash
# Full suite
uv run pytest

# With coverage report
uv run pytest --cov=. --cov-report=term-missing

# Unit tests only (no Groq key required)
uv run pytest tests/unit/ -v

# Client-layer tests
uv run pytest tests/providers/ -v

# Integration tests (mock SDK — no real key)
uv run pytest tests/integration/ -v

# E2E tests (requires GROQ_API_KEY_* in .env)
uv run pytest tests/e2e/ -v -m e2e

# Quick sanity check
uv run python scripts/smoke_test.py
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide — dev setup, coding conventions, branch naming, commit format, pre-commit hooks, and the PR checklist.

---

## License

[MIT](LICENSE) © 2026 Garv Patel