"""
chat.py — Chat completion schemas, including tool-calling primitives.

ToolDefinition and ToolCall live in this file rather than getting their own
request/response pair: tool calling rides inside ChatRequest.tools (what the
model may call) and ChatResponse.message.tool_calls (what it chose to call),
mirroring how clients/tool_client.py already reuses the chat completions
endpoint rather than a separate one.

request_type / response_type are fixed Literal discriminators used by
PoolGateRequest/PoolGateResponse (envelope.py) to resolve which concrete
schema a payload is at runtime. See envelope.py's module docstring for the
caveat this implies when parsing raw JSON.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.common import FinishReason, Metadata, utcnow
from schemas.usage import TokenUsage


class ToolDefinition(BaseModel):
    """A single function the model may choose to call, OpenAI function-calling shape."""

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(
        ...,
        description="JSON Schema object describing the function's arguments.",
    )


class ToolCall(BaseModel):
    """
    A single tool invocation the model requested.

    Flattened relative to the raw OpenAI/Groq wire shape (which nests
    name/arguments under a `function` object) — this is PoolGate's internal
    representation, not required to mirror the SDK's JSON 1:1.
    """

    id: str
    type: Literal["function"] = "function"
    name: str
    arguments: str = Field(..., description="Raw JSON string, exactly as returned by the model.")

    @property
    def parsed_arguments(self) -> dict[str, Any]:
        """Parse `arguments` into a dict. Raises json.JSONDecodeError if the model emitted malformed JSON."""
        return json.loads(self.arguments)


class ChatMessage(BaseModel):
    """A single turn in a chat conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = Field(default=None, description="Optional participant name.")
    tool_call_id: str | None = Field(
        default=None,
        description="Required when role='tool' — links back to the ToolCall.id being answered.",
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="Set when role='assistant' and the model requested tool calls.",
    )

    @model_validator(mode="after")
    def _validate_role_consistency(self) -> ChatMessage:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("role='tool' messages must set tool_call_id.")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("tool_calls may only be set on role='assistant' messages.")
        if self.content is None and not self.tool_calls:
            raise ValueError("ChatMessage requires content, tool_calls, or both.")
        return self

    def to_payload(self) -> dict[str, Any]:
        """Convert to the raw dict shape clients/chat_client.py expects, dropping unset fields."""
        return self.model_dump(exclude_none=True)


class ChatRequest(BaseModel):
    """
    Request body for a chat completion, with optional tool calling attached.

    request_type is a fixed discriminator — see PoolGateRequest in envelope.py.
    """

    model_config = ConfigDict(extra="forbid")

    request_type: Literal["chat"] = "chat"

    model: str
    messages: list[ChatMessage] = Field(..., min_length=1)

    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    stop: list[str] | str | None = None
    stream: bool = False
    timeout: float | None = Field(default=None, gt=0)

    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["auto", "none", "required"] | dict[str, Any] | None = None

    metadata: Metadata | None = None

    @model_validator(mode="after")
    def _validate_tool_choice(self) -> ChatRequest:
        if self.tool_choice not in (None, "none") and not self.tools:
            raise ValueError("tool_choice requires at least one entry in `tools`.")
        return self


class ChatResponse(BaseModel):
    """
    Response body for a chat completion.

    response_type is a fixed discriminator — see PoolGateResponse in envelope.py.
    """

    response_type: Literal["chat"] = "chat"

    id: str = Field(..., description="The originating request_id.")
    model: str
    message: ChatMessage
    finish_reason: FinishReason
    usage: TokenUsage
    latency_ms: float = Field(..., ge=0)
    created_at: datetime = Field(default_factory=utcnow)
