import os
from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbeddings(Embeddings):
    """LangChain Embeddings over FastEmbed (ONNX Runtime): no PyTorch, ~70 MB model, 384 dims."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 8, cache_dir: str = None):
        from fastembed import TextEmbedding

        self.model_name = model_name
        # Attention memory grows with batch x seq_len^2; FastEmbed's default batch of 256
        # peaks at several GB, which would crash a 1-2 GB hosted container.
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache(maxsize=2)
def get_embeddings(model_name: str = None):
    return FastEmbedEmbeddings(
        model_name=model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
        cache_dir=os.getenv("FASTEMBED_CACHE_PATH"),
    )
