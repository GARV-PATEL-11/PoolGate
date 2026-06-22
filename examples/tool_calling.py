"""Tool calling — invoke with a function tool and inspect the tool call response."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

from dotenv import load_dotenv

from exceptions.base import GroqServiceError
from services.provider_service import GroqService


load_dotenv()

_MODEL = "llama-3.3-70b-versatile"

TOOLS: list[dict] = [
	{
		"type": "function",
		"function": {
			"name": "get_weather",
			"description": "Get the current weather for a location.",
			"parameters": {
				"type": "object",
				"properties": {
					"location": {"type": "string", "description": "City name, e.g. 'Paris'"},
					"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
					},
				"required": ["location"],
				},
			},
		},
	]

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., dict]] = {}


def _register(name: str) -> Callable:
	"""Decorator — register a function as a dispatchable tool."""

	def decorator(fn: Callable) -> Callable:
		_REGISTRY[name] = fn
		return fn

	return decorator


@_register("get_weather")
def get_weather(location: str, unit: str = "celsius") -> dict:
	"""Stub — replace with a real weather API call."""
	return {
		"location": location,
		"temperature": 22 if unit == "celsius" else 72,
		"unit": unit,
		"condition": "Sunny",
		}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(message) -> dict:
	"""Coerce a ChatCompletionMessage (or any Pydantic model) to a plain dict.

	_validate_messages expects every entry in the messages list to be a dict
	and calls .get() on each one.  Passing the raw SDK object causes an
	AttributeError because Pydantic models don't expose .get().
	"""
	if isinstance(message, dict):
		return message
	# Pydantic v2 — strip None fields so the schema validator doesn't choke
	# on optional SDK fields that have no meaning to Groq's chat endpoint.
	if hasattr(message, "model_dump"):
		return message.model_dump(exclude_none=True)
	# Pydantic v1 fallback
	if hasattr(message, "dict"):
		return message.dict(exclude_none=True)
	raise TypeError(f"Cannot convert {type(message)} to a message dict")


def _dispatch(name: str, args: dict) -> str:
	"""Call the named tool and return its JSON-serialised result.

	Raises GroqServiceError for unknown tool names so the caller always gets
	a bounded result variable — no UnboundLocalError risk.
	"""
	fn = _REGISTRY.get(name)
	if fn is None:
		raise GroqServiceError(f"Model requested unknown tool: {name!r}")
	return json.dumps(fn(**args))


def _collect_tool_results(choice) -> list[dict]:
	"""Execute every tool call in *choice* and return the result messages."""
	results = []
	for tc in choice.message.tool_calls or []:
		args = json.loads(tc.function.arguments)
		print(f"  Tool called : {tc.function.name}({args})")
		results.append({
			"role": "tool",
			"tool_call_id": tc.id,
			"name": tc.function.name,
			"content": _dispatch(tc.function.name, args),
			},
			)
	return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
	service = GroqService()
	messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

	# --- Turn 1: let the model decide whether to call a tool ---
	response = service.invoke_tools(
		messages=messages,
		tools=TOOLS,
		model=_MODEL,
		tool_choice="auto",
		)
	print(f"Finish reason : {response.finish_reason}")
	print(f"Response text : {response.text!r}")

	if not (response.raw_response and response.raw_response.choices):
		return

	choice = response.raw_response.choices[0]
	if not choice.message.tool_calls:
		return  # model answered directly — nothing left to do

	# --- Execute tools, then send results back for a final answer ---
	tool_results = _collect_tool_results(choice)
	if not tool_results:
		return

	messages.append(_to_dict(choice.message))  # ← was: choice.message (raw object)
	messages.extend(tool_results)

	final = service.invoke_tools(
		messages=messages,
		tools=TOOLS,
		model=_MODEL,
		tool_choice="auto",
		)
	print("\nFinal Answer:")
	print(final.text)


if __name__ == "__main__":
	try:
		main()
	except GroqServiceError as exc:
		print(f"GroqServiceError: {exc}", file=sys.stderr)
		sys.exit(1)
