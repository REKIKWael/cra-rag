"""The language model that writes the answers, behind a small interface.

Anything with a `model` name and a `chat(system, user, on_token)` method can be
plugged in: today Ollama running locally, tomorrow a hosted API, without
touching the rest of the code.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

from cra_rag.embed import OLLAMA_URL

CHAT_MODEL = "ministral-3:3b"  # Mistral AI, Apache 2.0, 3 GB: runs on a laptop CPU


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int = 0
    prompt_seconds: float = 0.0  # time spent reading the prompt
    output_tokens: int = 0
    output_seconds: float = 0.0  # time spent writing the answer


class LLMError(RuntimeError):
    """The model could not answer; the message says how to fix it."""


class LLM(Protocol):
    model: str

    def chat(
        self, system: str, user: str, on_token: Callable[[str], None] | None = None
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
            "temperature": 0,  # same question, same answer: needed to evaluate
            "seed": 0,
            "num_predict": max_output_tokens,
            "num_ctx": context_tokens,
        }

    def chat(
        self, system: str, user: str, on_token: Callable[[str], None] | None = None
    ) -> Completion:
        request = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": True,
            "options": self.options,
        }
        pieces: list[str] = []
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
        )
