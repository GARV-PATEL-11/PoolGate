"""Shared mock factories for PoolGate tests.

Import these in any test file that needs a mock SDK or response:

    from tests.mocks import mock_completion, mock_stream, mock_async_sdk, MockSdk

All factories return configured MagicMock / AsyncMock objects that satisfy
the interfaces expected by the capability clients and GroqService.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


def mock_completion(
    text: str = "mocked answer",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    finish_reason: str = "stop",
) -> MagicMock:
    """Return a mock completion object matching the Groq SDK shape."""
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=text), finish_reason=finish_reason)]
    completion.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return completion


def mock_stream(texts: list[str]) -> MagicMock:
    """Return a context-manager mock that yields one chunk per text element."""
    chunks = [_make_chunk(t) for t in texts]
    # Final chunk carries the x_groq usage header
    final = MagicMock()
    final.choices = [MagicMock(delta=MagicMock(content=""))]
    final.x_groq = MagicMock()
    final.x_groq.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunks.append(final)

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=iter(chunks))
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _make_chunk(text: str) -> MagicMock:
    chunk = MagicMock(x_groq=None)
    chunk.choices = [MagicMock(delta=MagicMock(content=text))]
    return chunk


# ---------------------------------------------------------------------------
# SDK factories
# ---------------------------------------------------------------------------


class MockSdk:
    """Sync SDK stand-in that returns configurable completions."""

    def __init__(self, completion: MagicMock | None = None) -> None:
        self._completion = completion or mock_completion()
        self.chat = MagicMock()
        self.chat.completions.create = MagicMock(return_value=self._completion)

    def set_completion(self, completion: MagicMock) -> None:
        self._completion = completion
        self.chat.completions.create.return_value = completion

    def set_side_effect(self, exc: Exception) -> None:
        self.chat.completions.create.side_effect = exc


class MockAsyncSdk:
    """Async SDK stand-in that returns configurable completions."""

    def __init__(self, completion: MagicMock | None = None) -> None:
        self._completion = completion or mock_completion()
        self.chat = MagicMock()
        self.chat.completions.create = AsyncMock(return_value=self._completion)

    def set_completion(self, completion: MagicMock) -> None:
        self._completion = completion
        self.chat.completions.create.return_value = completion

    def set_side_effect(self, exc: Exception) -> None:
        self.chat.completions.create.side_effect = exc


def mock_sync_sdk(
    text: str = "mocked answer",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
) -> MagicMock:
    """Return a plain MagicMock sync SDK suitable for monkeypatching _sync_sdk."""
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = mock_completion(
        text,
        prompt_tokens,
        completion_tokens,
    )
    return sdk


def mock_async_sdk(
    text: str = "mocked answer",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
) -> MagicMock:
    """Return a MagicMock with AsyncMock create() suitable for monkeypatching _async_sdk."""
    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(
        return_value=mock_completion(text, prompt_tokens, completion_tokens),
    )
    return sdk
