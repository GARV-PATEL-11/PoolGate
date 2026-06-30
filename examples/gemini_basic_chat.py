"""Basic Gemini chat example.

Prerequisites:
    pip install poolgate[gemini]

    export TOTAL_GEMINI_KEYS=1
    export GEMINI_API_KEY_01=your-google-ai-studio-key

Models available:
    gemini-2.5-flash  (RPM=5, TPM=250K, RPD=20, Free Tier)
    gemini-3.5-flash  (RPM=5, TPM=250K, RPD=20, Free Tier)
"""

import asyncio

from poolgate.services.gemini_provider import GeminiService


def basic_chat() -> None:
    service = GeminiService()

    # Simple single-turn chat
    response = service.chat(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        model="gemini-2.5-flash",
        temperature=0.7,
    )
    print(f"Response: {response.text}")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.total_tokens}")
    print(f"Latency: {response.latency:.2f}s")


def multi_turn_chat() -> None:
    service = GeminiService()

    messages = [
        {"role": "system", "content": "You are a concise travel guide."},
        {"role": "user", "content": "Tell me about Paris."},
    ]
    response = service.chat(messages=messages, model="gemini-2.5-flash")
    print(f"Guide: {response.text}")

    messages += [
        {"role": "assistant", "content": response.text},
        {"role": "user", "content": "What is the best time to visit?"},
    ]
    follow_up = service.chat(messages=messages, model="gemini-2.5-flash")
    print(f"Follow-up: {follow_up.text}")


def streaming_example() -> None:
    service = GeminiService()

    print("Streaming: ", end="", flush=True)
    for chunk in service.stream(
        prompt="Write a haiku about the ocean.",
        model="gemini-2.5-flash",
    ):
        print(chunk, end="", flush=True)
    print()


def structured_output_example() -> None:
    from pydantic import BaseModel

    class City(BaseModel):
        name: str
        country: str
        population: int
        fun_fact: str

    service = GeminiService()
    city = service.structured(
        prompt="Give me details about Tokyo.",
        schema=City,
        model="gemini-2.5-flash",
    )
    print(f"City: {city.name}, {city.country}")
    print(f"Population: {city.population:,}")
    print(f"Fun fact: {city.fun_fact}")


def tool_calling_example() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    service = GeminiService()
    response = service.invoke_tools(
        messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
        tools=tools,
        model="gemini-2.5-flash",
    )
    tool_calls = response.metadata.get("tool_calls", [])
    if tool_calls:
        print(f"Tool called: {tool_calls[0]['function']['name']}")
        print(f"Arguments: {tool_calls[0]['function']['arguments']}")
    else:
        print(f"Response: {response.text}")


async def async_batch_example() -> None:
    service = GeminiService()

    prompts = [
        "What is 2+2?",
        "Name the planets in our solar system.",
        "What is the boiling point of water?",
    ]
    summary = await service.async_batch_chat(
        prompts=prompts,
        model="gemini-2.5-flash",
        concurrency=2,
    )
    print(f"Batch: {summary.succeeded}/{summary.total} succeeded")
    for result in summary.results:
        if result.success and result.response:
            print(f"  [{result.index}] {result.response.text[:60]}...")


if __name__ == "__main__":
    print("=== Basic Chat ===")
    basic_chat()

    print("\n=== Multi-turn Chat ===")
    multi_turn_chat()

    print("\n=== Streaming ===")
    streaming_example()

    print("\n=== Structured Output ===")
    structured_output_example()

    print("\n=== Tool Calling ===")
    tool_calling_example()

    print("\n=== Async Batch ===")
    asyncio.run(async_batch_example())
