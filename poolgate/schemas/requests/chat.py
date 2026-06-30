from __future__ import annotations

import json
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from poolgate.schemas.common import Metadata


class ToolDefinition(BaseModel):
    """A single function the model may choose to call."""

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(..., description="JSON Schema object describing the function's arguments.")


class ToolCall(BaseModel):
    """A single tool invocation the model requested."""

    id: str
    type: Literal["function"] = "function"
    name: str
    arguments: str = Field(..., description="Raw JSON string, exactly as returned by the model.")

    @property
    def parsed_arguments(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.arguments))


class ChatMessage(BaseModel):
    """A single turn in a chat conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def _validate_role_consistency(self) -> "ChatMessage":
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("role='tool' messages must set tool_call_id.")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("tool_calls may only be set on role='assistant' messages.")
        if self.content is None and not self.tool_calls:
            raise ValueError("ChatMessage requires content, tool_calls, or both.")
        return self

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class ChatRequest(BaseModel):
    """Request body for a chat completion, with optional tool calling."""

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
    def _validate_tool_choice(self) -> "ChatRequest":
        if self.tool_choice not in (None, "none") and not self.tools:
            raise ValueError("tool_choice requires at least one entry in `tools`.")
        return self
