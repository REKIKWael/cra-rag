"""Turn texts into vectors (embeddings) with a model served locally by Ollama.

Two texts that mean the same thing get close vectors, even when they share no
words: "who do we warn when a flaw is exploited?" lands near Article 14. That is
what keyword search (BM25) cannot do.
"""

import httpx

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"  # multilingual (French and English), 1024 dimensions


class EmbeddingError(RuntimeError):
    """Ollama could not produce the embeddings; the message says how to fix it."""


class OllamaEmbedder:
    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        url: str = OLLAMA_URL,
        client: httpx.Client | None = None,
        batch_size: int = 8,
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.batch_size = batch_size
        # Embedding on a laptop CPU can be slow: allow up to 10 minutes per batch.
        self.client = client or httpx.Client(timeout=600.0, trust_env=False)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the same order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._post(batch))
        return vectors

    def _post(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self.client.post(
                f"{self.url}/api/embed", json={"model": self.model, "input": batch}
            )
        except httpx.ConnectError as exc:
            raise EmbeddingError(
                f"Ollama does not answer at {self.url}. Start the Ollama application, then retry."
            ) from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingError("Ollama took too long to answer. Retry with fewer texts.") from exc
        if response.status_code == 404:
            raise EmbeddingError(
                f"The model {self.model!r} is not installed. Run: ollama pull {self.model}"
            )
        if response.status_code != 200:
            detail = response.text[:200]
            raise EmbeddingError(f"Ollama answered HTTP {response.status_code}: {detail}")
        vectors = response.json()["embeddings"]
        if len(vectors) != len(batch):
            raise EmbeddingError(f"Ollama returned {len(vectors)} vectors for {len(batch)} texts.")
        return vectors
