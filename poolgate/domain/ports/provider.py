"""IProviderClient — DIP contract for all provider adapters."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Generator, Protocol, runtime_checkable

from poolgate.domain.models.request import ProviderRequest
from poolgate.domain.models.response import ProviderResponse


@runtime_checkable
class IProviderClient(Protocol):
    def invoke(self, request: ProviderRequest, api_key: str) -> ProviderResponse: ...

    async def async_invoke(self, request: ProviderRequest, api_key: str) -> ProviderResponse: ...

    def stream(self, request: ProviderRequest, api_key: str) -> Generator[str, None, None]: ...

    async def async_stream(self, request: ProviderRequest, api_key: str) -> AsyncGenerator[str, None]: ...

    def invoke_tools(self, request: ProviderRequest, api_key: str, tools: list[dict[str, Any]]) -> ProviderResponse: ...

    def invoke_structured(self, request: ProviderRequest, api_key: str, schema: type) -> Any: ...
