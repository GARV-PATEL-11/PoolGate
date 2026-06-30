from __future__ import annotations


class GroqServiceError(Exception):
    """Base exception for all PoolGate errors. Never exposes raw API keys."""

    def __init__(self, message: str, request_id: str | None = None) -> None:
        self.request_id = request_id
        super().__init__(message)

    def __repr__(self) -> str:  # pragma: no cover
        cls = type(self).__name__
        rid = f", request_id={self.request_id!r}" if self.request_id else ""
        return f"{cls}({str(self)!r}{rid})"
