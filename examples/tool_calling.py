"""
Tool calling — invoke with a function tool and inspect tool call behavior
across many prompts.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

from dotenv import load_dotenv

from poolgate.exceptions.base import GroqServiceError
from poolgate.services.provider import GroqService

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
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Paris'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    },
]

PROMPTS: list[str] = [  # Basic tool calls
    "What's the weather in Tokyo?",
    "What's the weather in Paris?",
    "Tell me the current weather in London.",
    "How's the weather in New York today?",
    "Weather in Berlin?",
    # Unit selection
    "What's the weather in Chicago in Fahrenheit?",
    "Give me the weather for Toronto using Celsius.",
    "How warm is Miami in degrees Fahrenheit?",
    "Show the weather in Sydney in metric units.",
    "Use Celsius and tell me the weather in Amsterdam.",
    # Natural language variations
    "Do I need a jacket in Tokyo?",
    "Will it be warm in Paris?",
    "Is it sunny in Rome right now?",
    "Can you check the weather for Madrid?",
    "I am traveling to Dubai. What's the weather like there?",
    # Location extraction
    "What's the weather in Ahmedabad, India?",
    "Tell me the weather for San Francisco, California.",
    "Weather at Tokyo, Japan.",
    "How's the weather in Los Angeles?",
    "Check weather for Mumbai.",
    # Multiple locations
    "Compare the weather in Tokyo and Paris.",
    "What's the weather in London, Berlin, and Madrid?",
    "Tell me the temperatures in New York and Chicago.",
    "Which city is warmer right now: Tokyo or Seoul?",
    # Indirect requests
    "I'm flying to Tokyo tomorrow. Should I pack light clothes?",
    "I'm visiting Paris. What weather should I expect?",
    "Would I need an umbrella in London?",
    "Is it a good day to go outside in Berlin?",
    # Parameter inference
    "Give me the weather in Boston using the American temperature scale.",
    "What's the weather in Dallas in metric?",
    "I prefer Fahrenheit. How's the weather in Phoenix?",
    "Use Celsius for Tokyo weather.",
    # Ambiguous locations
    "What's the weather in Springfield?",
    "Weather in Cambridge.",
    "Tell me the weather in Victoria.",
    "What's it like in Washington?",
    # Should not require tool calls
    "What is weather?",
    "Explain how weather forecasts work.",
    "What causes rain?",
    "Define humidity.",
    "How do meteorologists predict storms?",
    # Structured extraction
    "Location: Tokyo\nUnit: Celsius",
    "City=Paris\nTemperature Unit=fahrenheit",
    '{"location":"Berlin","unit":"celsius"}',
    "Weather request:\n- Location: London\n- Unit: fahrenheit",
]

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., dict]] = {}


def _register(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return decorator


@_register("get_weather")
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Stub weather tool."""
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
    if isinstance(message, dict):
        return message

    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)

    if hasattr(message, "dict"):
        return message.dict(exclude_none=True)

    raise TypeError(f"Cannot convert {type(message)} to a message dict")


def _dispatch(name: str, args: dict) -> str:
    fn = _REGISTRY.get(name)

    if fn is None:
        raise GroqServiceError(f"Model requested unknown tool: {name!r}")

    return json.dumps(fn(**args))


def _collect_tool_results(choice) -> list[dict]:
    results: list[dict] = []

    for tool_call in choice.message.tool_calls or []:
        args = json.loads(tool_call.function.arguments)

        print(
            f"  Tool called: " f"{tool_call.function.name}({args})",
        )

        results.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": _dispatch(tool_call.function.name, args),
            },
        )

    return results


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def run_prompt(service: GroqService, prompt: str) -> None:
    print("\n" + "=" * 100)
    print(f"PROMPT: {prompt}")
    print("=" * 100)

    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = service.invoke_tools(
        messages=messages,
        tools=TOOLS,
        model=_MODEL,
        tool_choice="auto",
    )

    print(f"Finish reason: {response.finish_reason}")
    print(f"Response text: {response.text!r}")

    if not response.raw_response or not response.raw_response.choices:
        print("No response choices returned.")
        return

    choice = response.raw_response.choices[0]

    if not choice.message.tool_calls:
        print("\nFinal Answer:")
        print(response.text)
        return

    tool_results = _collect_tool_results(choice)

    if not tool_results:
        print("No tool results generated.")
        return

    messages.append(_to_dict(choice.message))
    messages.extend(tool_results)

    final_response = service.invoke_tools(
        messages=messages,
        tools=TOOLS,
        model=_MODEL,
        tool_choice="auto",
    )

    print("\nFinal Answer:")
    print(final_response.text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    service = GroqService()

    print(f"Running {len(PROMPTS)} tool-calling test prompts...\n")

    for index, prompt in enumerate(PROMPTS, start=1):
        print(f"\n[{index}/{len(PROMPTS)}]")
        run_prompt(service, prompt)

    service.flush_tracking()

    if service._config.paths.base_dir:
        print(
            f"\nData saved to " f"{service._config.paths.base_dir}/",
        )


if __name__ == "__main__":
    try:
        main()

    except GroqServiceError as exc:
        print(
            f"GroqServiceError: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
