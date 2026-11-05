"""The language model that writes the answers, behind a small interface.

Anything with a `model` name and a `chat(system, user, on_token)` method can be
plugged in: today Ollama running locally, tomorrow a hosted API, without
touching the rest of the code.

An agent needs one more thing: a conversation in which the model may ask to run
tools (`ToolLLM.converse`). The model only names the tool and its arguments; the
program runs it and sends the result back in the next message.

Two clients: `OllamaLLM` (Ollama's native API, the default) and
`OpenAICompatibleLLM` (any server speaking the OpenAI API: the LiteLLM proxy,
vLLM, hosted providers). `connect()` picks one from the environment.
"""

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

from cra_rag.embed import OLLAMA_URL

CHAT_MODEL = "ministral-3:3b"  # Mistral AI, Apache 2.0, 3 GB: runs on a laptop CPU


@dataclass(frozen=True)
class ToolCall:
    """The model asks the program to run a tool: its name and its arguments."""

    name: str
    arguments: dict


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int = 0
    prompt_seconds: float = 0.0  # time spent reading the prompt
    output_tokens: int = 0
    output_seconds: float = 0.0  # time spent writing the answer
    tool_calls: tuple[ToolCall, ...] = ()  # empty unless tools were offered


class LLMError(RuntimeError):
    """The model could not answer; the message says how to fix it."""


class LLM(Protocol):
    model: str

    def chat(
        self, system: str, user: str, on_token: Callable[[str], None] | None = None
    ) -> Completion: ...


class ToolLLM(LLM, Protocol):
    """A model that can also hold a conversation and call tools (for the agent)."""

    def converse(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> Completion: ...


class OllamaLLM:
    """A chat model served by Ollama. Answers are streamed: `on_token` receives
    each piece of text as soon as it is written, so a slow computer shows progress."""

    def __init__(
        self,
        model: str = CHAT_MODEL,
        url: str = OLLAMA_URL,
        client: httpx.Client | None = None,
        max_output_tokens: int = 300,
        context_tokens: int = 8192,
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.client = client or httpx.Client(timeout=900.0, trust_env=False)
        self.options = {
            # As repeatable as possible, which evaluation needs. Not fully: from one
            # machine to another, answers can still differ (see "Agent mode" in the README).
            "temperature": 0,
            "seed": 0,
            "num_predict": max_output_tokens,
            "num_ctx": context_tokens,
        }

    def chat(
        self,
        system: str,
        user: str,
        on_token: Callable[[str], None] | None = None,
        schema: dict | None = None,
    ) -> Completion:
        """`schema`: a JSON schema the reply must follow (Ollama structured outputs)."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._post(messages, on_token, schema=schema)

    def converse(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> Completion:
        """One reply in a conversation. `messages` follow Ollama's format (system,
        user, assistant and tool roles); `tools` describe the functions the model
        may ask for, and its requests come back in `Completion.tool_calls`."""
        return self._post(messages, on_token, tools=tools)

    def _post(
        self,
        messages: list[dict],
        on_token: Callable[[str], None] | None,
        schema: dict | None = None,
        tools: list[dict] | None = None,
    ) -> Completion:
        request = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self.options,
        }
        if schema:
            request["format"] = schema
        if tools:
            request["tools"] = tools
        pieces: list[str] = []
        calls: list[ToolCall] = []
        final: dict = {}
        try:
            with self.client.stream("POST", f"{self.url}/api/chat", json=request) as response:
                if response.status_code == 404:
                    raise LLMError(
                        f"The model {self.model!r} is not installed. Run: ollama pull {self.model}"
                    )
                if response.status_code != 200:
                    detail = response.read().decode(errors="replace")[:200]
                    raise LLMError(f"Ollama answered HTTP {response.status_code}: {detail}")
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    message = json.loads(line)
                    if "error" in message:
                        raise LLMError(f"Ollama: {message['error']}")
                    piece = message.get("message", {}).get("content", "")
                    if piece:
                        pieces.append(piece)
                        if on_token:
                            on_token(piece)
                    for call in message.get("message", {}).get("tool_calls", []):
                        function = call.get("function", {})
                        arguments = function.get("arguments") or {}
                        if isinstance(arguments, str):  # some models send JSON text
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {}
                        calls.append(ToolCall(function.get("name", ""), arguments))
                    if message.get("done"):
                        final = message
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Ollama does not answer at {self.url}. Start the Ollama application, then retry."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError("The model took too long to answer.") from exc

        return Completion(
            text="".join(pieces),
            prompt_tokens=final.get("prompt_eval_count", 0),
            prompt_seconds=final.get("prompt_eval_duration", 0) / 1e9,
            output_tokens=final.get("eval_count", 0),
            output_seconds=final.get("eval_duration", 0) / 1e9,
            tool_calls=tuple(calls),
        )


class OpenAICompatibleLLM:
    """A chat model behind an OpenAI-compatible API: the LiteLLM proxy, a vLLM
    server, Ollama's own /v1 endpoint, or a hosted provider.

    The OpenAI chat format is the common language of these servers, so the same
    client reaches all of them; only `base_url` and the model name change. The
    differences with Ollama's native API are handled here: tool calls carry ids
    and JSON text arguments, streamed pieces arrive as Server-Sent Events, and
    timings are measured by the client (time to the first word, then the rest).
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        client: httpx.Client | None = None,
        max_output_tokens: int = 300,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=900.0, trust_env=False)
        self.max_output_tokens = max_output_tokens

    def chat(
        self,
        system: str,
        user: str,
        on_token: Callable[[str], None] | None = None,
        schema: dict | None = None,
    ) -> Completion:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._post(messages, on_token, schema=schema)

    def converse(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> Completion:
        return self._post(to_openai_messages(messages), on_token, tools=tools)

    def _post(
        self,
        messages: list[dict],
        on_token: Callable[[str], None] | None,
        schema: dict | None = None,
        tools: list[dict] | None = None,
    ) -> Completion:
        request = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "seed": 0,
            "max_tokens": self.max_output_tokens,
        }
        if schema:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": schema},
            }
        if tools:
            request["tools"] = tools
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        pieces: list[str] = []
        calls: dict[int, dict] = {}  # tool calls arrive in pieces, by index
        usage: dict = {}
        start = time.monotonic()
        first_word: float | None = None
        url = f"{self.base_url}/chat/completions"
        try:
            with self.client.stream("POST", url, json=request, headers=headers) as response:
                if response.status_code in (401, 403):
                    raise LLMError(f"{self.base_url} refused the key: set CRA_RAG_LLM_API_KEY.")
                if response.status_code == 404:
                    raise LLMError(f"No model {self.model!r} (or no API) at {self.base_url}.")
                if response.status_code != 200:
                    detail = response.read().decode(errors="replace")[:200]
                    if not self.api_key:  # some servers (LiteLLM) answer 500 to a missing key
                        detail += " (no key was sent: set CRA_RAG_LLM_API_KEY if it needs one)"
                    raise LLMError(
                        f"{self.base_url} answered HTTP {response.status_code}: {detail}"
                    )
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue  # blank lines and keep-alive comments
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        piece = delta.get("content") or ""
                        if piece:
                            first_word = first_word or time.monotonic()
                            pieces.append(piece)
                            if on_token:
                                on_token(piece)
                        for call in delta.get("tool_calls") or []:
                            first_word = first_word or time.monotonic()
                            known = calls.setdefault(call.get("index", 0), {"name": "", "args": ""})
                            function = call.get("function", {})
                            known["name"] += function.get("name") or ""
                            known["args"] += function.get("arguments") or ""
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Nothing answers at {self.base_url}: start the server (LiteLLM, vLLM...)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError("The model took too long to answer.") from exc

        end = time.monotonic()
        first_word = first_word or end
        tool_calls = []
        for _, call in sorted(calls.items()):
            try:
                arguments = json.loads(call["args"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(call["name"], arguments if isinstance(arguments, dict) else {})
            )
        return Completion(
            text="".join(pieces),
            prompt_tokens=usage.get("prompt_tokens", 0),
            prompt_seconds=first_word - start,
            output_tokens=usage.get("completion_tokens", 0),
            output_seconds=end - first_word,
            tool_calls=tuple(tool_calls),
        )


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Ollama's conversation format to OpenAI's: the agent writes the first one.

    OpenAI ties each tool result to its request with an id, and wants the arguments
    as JSON text; Ollama names the tool instead. Ids are given in order, and each
    tool result takes the id of the oldest request still waiting for its result.
    """
    converted: list[dict] = []
    waiting: list[str] = []
    count = 0
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                call_id, count = f"call_{count}", count + 1
                function = call["function"]
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "arguments": json.dumps(function.get("arguments", {})),
                        },
                    }
                )
                waiting.append(call_id)
            converted.append(
                {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}
            )
        elif message.get("role") == "tool":
            call_id = waiting.pop(0) if waiting else "call_unknown"
            converted.append(
                {"role": "tool", "tool_call_id": call_id, "content": message.get("content", "")}
            )
        else:
            converted.append({"role": message["role"], "content": message.get("content", "")})
    return converted


def connect(
    model: str,
    ollama_url: str = OLLAMA_URL,
    max_output_tokens: int = 300,
    context_tokens: int = 8192,
) -> "OllamaLLM | OpenAICompatibleLLM":
    """The chat model to use: Ollama by default, or any OpenAI-compatible server
    (LiteLLM proxy, vLLM...) when CRA_RAG_LLM_URL is set, e.g. http://localhost:4000/v1.
    Nothing else in the code depends on the choice.

    `context_tokens` only reaches Ollama directly: behind an OpenAI-compatible API the
    context window is a setting of the server (num_ctx in the LiteLLM configuration,
    --max-model-len for vLLM)."""
    base_url = os.environ.get("CRA_RAG_LLM_URL", "").strip()
    if base_url:
        api_key = os.environ.get("CRA_RAG_LLM_API_KEY", "")
        return OpenAICompatibleLLM(model, base_url, api_key, max_output_tokens=max_output_tokens)
    return OllamaLLM(
        model, ollama_url, max_output_tokens=max_output_tokens, context_tokens=context_tokens
    )
