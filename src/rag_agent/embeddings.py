from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

from rag_agent.config import get_settings


EXPECTED_EMBEDDING_DIMENSION = 384


class Embedder:
    def __init__(self) -> None:
        settings = get_settings()
        self.device = self._select_device()
        self.model = SentenceTransformer(
            settings.embedding_model,
            device=self.device,
        )
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding model must output {EXPECTED_EMBEDDING_DIMENSION} dimensions "
                f"to match the PostgreSQL vector column; got {dimension}"
            )

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()