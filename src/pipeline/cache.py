"""Cache em 2 niveis: exact-match (SHA256) + semantic (cosine similarity).

TODO 5 (implementado): SemanticCache.get(). Embeddings LOCAIS via sentence-transformers
(mesmo modelo multilingue do RAG) — Groq nao expoe endpoint de embeddings.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

DEFAULT_EMBED = "paraphrase-multilingual-MiniLM-L12-v2"


class ExactCache:
    """Cache por hash SHA256 da query. Captura replays exatos (~10-15% das queries)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> str | None:
        return self._store.get(self._key(query))

    def put(self, query: str, answer: str) -> None:
        self._store[self._key(query)] = answer

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store)}


class SemanticCache:
    """Cache por similaridade de embedding. Captura parafrases (~20% adicional)."""

    def __init__(self, threshold: float = 0.90) -> None:
        self.threshold = threshold
        self._queries: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._answers: list[str] = []
        self._model = None  # carregado preguicosamente (lazy)
        self._embed_model_name = os.environ.get("EMBED_MODEL", DEFAULT_EMBED)

    def _embed(self, text: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._embed_model_name)
        return np.asarray(self._model.encode(text), dtype=np.float32)

    # ------------------------------------------------------------------ TODO 5
    def get(self, query: str) -> str | None:
        """Retorna resposta cacheada se a query for similar a alguma anterior, ou None."""
        if not self._queries:
            return None

        e = self._embed(query)
        sims = [
            float(np.dot(e, em) / (np.linalg.norm(e) * np.linalg.norm(em) + 1e-9))
            for em in self._embeddings
        ]
        idx = int(np.argmax(sims))
        if sims[idx] >= self.threshold:
            return self._answers[idx]
        return None

    def put(self, query: str, answer: str) -> None:
        self._queries.append(query)
        self._embeddings.append(self._embed(query))
        self._answers.append(answer)

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._queries), "threshold": self.threshold}
