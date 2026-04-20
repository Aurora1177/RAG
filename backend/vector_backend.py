"""Vector store: local ChromaDB by default (same idea as rag/main.ipynb PersistentClient). Optional Milvus when VECTOR_BACKEND=milvus."""
import os

_chroma_singleton = None


def get_vector_backend() -> str:
    v = (os.getenv("VECTOR_BACKEND") or "chroma").strip().lower()
    if v in ("milvus", "chroma"):
        return v
    return "chroma"


def is_chroma() -> bool:
    return get_vector_backend() == "chroma"


def is_milvus() -> bool:
    return get_vector_backend() == "milvus"


def get_chroma_store():
    global _chroma_singleton
    if _chroma_singleton is None:
        from .chroma_store import ChromaVectorStore

        _chroma_singleton = ChromaVectorStore()
    return _chroma_singleton
