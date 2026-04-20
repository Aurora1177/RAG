"""
Local vector storage using Chroma PersistentClient (same idea as rag/main.ipynb).

Uses the same dense embeddings as Milvus mode (embedding_service); no sparse/BM25 in Chroma path.
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


class ChromaVectorStore:
    """ChromaDB persistent store; API shaped for rag_utils / milvus_writer."""

    def __init__(self) -> None:
        default_path = _ROOT / "data" / "chroma_db"
        raw = (os.getenv("CHROMA_PERSIST_PATH") or "").strip()
        self.persist_path = str(Path(raw).expanduser().resolve()) if raw else str(default_path.resolve())
        self.collection_name = (os.getenv("CHROMA_COLLECTION") or os.getenv("MILVUS_COLLECTION") or "rag_new_kb").strip()
        self._client = None
        self._collection = None

    def _collection_or_create(self):
        if self._collection is None:
            import chromadb

            Path(self.persist_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_path)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "ip"},
            )
        return self._collection

    def init_collection(self) -> None:
        self._collection_or_create()

    def insert(self, rows: list[dict]) -> None:
        """rows: dense_embedding + text + metadata fields (same keys as Milvus insert)."""
        if not rows:
            return
        col = self._collection_or_create()
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for r in rows:
            cid = (r.get("chunk_id") or "").strip() or f"auto_{hash(r.get('text', ''))}"
            ids.append(cid[:512])
            embeddings.append(r["dense_embedding"])
            documents.append(r.get("text") or "")
            metadatas.append(
                {
                    "filename": str(r.get("filename", ""))[:250],
                    "file_type": str(r.get("file_type", ""))[:50],
                    "file_path": str(r.get("file_path", ""))[:500],
                    "page_number": int(r.get("page_number", 0) or 0),
                    "chunk_idx": int(r.get("chunk_idx", 0) or 0),
                    "chunk_id": str(r.get("chunk_id", ""))[:500],
                    "parent_chunk_id": str(r.get("parent_chunk_id", ""))[:500],
                    "root_chunk_id": str(r.get("root_chunk_id", ""))[:500],
                    "chunk_level": int(r.get("chunk_level", 0) or 0),
                }
            )
        col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def delete_by_filename(self, filename: str) -> None:
        if not filename:
            return
        col = self._collection_or_create()
        col.delete(where={"filename": {"$eq": filename}})

    def query_dense(
        self,
        query_embedding: list[float],
        top_k: int,
        leaf_level: int,
    ) -> list[dict]:
        col = self._collection_or_create()
        res = col.query(
            query_embeddings=[query_embedding],
            n_results=max(1, top_k),
            where={"chunk_level": {"$eq": int(leaf_level)}},
            include=["documents", "metadatas", "distances"],
        )
        out: list[dict] = []
        ids_batch = res.get("ids") or []
        docs_batch = res.get("documents") or []
        meta_batch = res.get("metadatas") or []
        dist_batch = res.get("distances") or []
        if not ids_batch or not ids_batch[0]:
            return out
        for i, _cid in enumerate(ids_batch[0]):
            meta = (meta_batch[0][i] if meta_batch and meta_batch[0] else {}) or {}
            dist = None
            if dist_batch and dist_batch[0] and i < len(dist_batch[0]):
                dist = dist_batch[0][i]
            text = ""
            if docs_batch and docs_batch[0] and i < len(docs_batch[0]):
                text = docs_batch[0][i] or ""
            # IP: higher similarity = lower distance in Chroma; use negative distance as score for sorting
            score = float(-(dist if dist is not None else 0.0))
            out.append(
                {
                    "text": text,
                    "filename": meta.get("filename", ""),
                    "file_type": meta.get("file_type", ""),
                    "page_number": int(meta.get("page_number", 0) or 0),
                    "chunk_id": meta.get("chunk_id", ""),
                    "parent_chunk_id": meta.get("parent_chunk_id", ""),
                    "root_chunk_id": meta.get("root_chunk_id", ""),
                    "chunk_level": int(meta.get("chunk_level", 0) or 0),
                    "chunk_idx": int(meta.get("chunk_idx", 0) or 0),
                    "score": score,
                }
            )
        return out

    def count_by_filename(self, filename: str) -> int:
        if not filename:
            return 0
        col = self._collection_or_create()
        res = col.get(where={"filename": {"$eq": filename}}, limit=100000, include=[])
        ids = res.get("ids") or []
        return len(ids)

    def get_texts_by_filename(self, filename: str) -> list[str]:
        col = self._collection_or_create()
        res = col.get(where={"filename": {"$eq": filename}}, limit=100000, include=["documents"])
        docs = res.get("documents") or []
        return [d or "" for d in docs]
