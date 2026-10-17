"""
Local embedding backends for ThermalOps RAG experiments.

Preferred neural model: sentence-transformers/all-MiniLM-L6-v2
License: Apache-2.0 (upstream sentence-transformers MiniLM)

On Python 3.14 Windows, stock PyPI often lacks a working ``torch`` wheel, so
the default neural backend uses ``fastembed`` + ``onnxruntime`` (same model
family, local ONNX weights, no paid API).
"""

from __future__ import annotations

from typing import Sequence

# Public model id used by both documentation and the FastEmbed loader.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_DIMENSION = 384


class FastEmbedMiniLM:
    """
    Local ONNX MiniLM embedder via the fastembed package.

    Downloads public model weights on first use into the local cache.
    Does not send document text to an external embedding API.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - env specific
            raise ImportError(
                "fastembed is required for the local MiniLM experiment. "
                "Install backend/requirements-rag.txt extras."
            ) from exc

        self.model_name = model_name
        self.dimension = DEFAULT_MODEL_DIMENSION
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # fastembed yields numpy arrays; convert to plain Python lists.
        vectors = list(self._model.embed(list(texts)))
        return [vector.tolist() for vector in vectors]


def try_create_minilm_embedder(
    model_name: str = DEFAULT_MODEL_NAME,
) -> FastEmbedMiniLM:
    """Factory used by the demo script; raises a clear ImportError if missing."""
    return FastEmbedMiniLM(model_name=model_name)
