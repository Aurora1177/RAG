"""文档向量化并写入 Milvus 或本地 Chroma - 支持密集+稀疏向量（Milvus）或仅密集（Chroma）"""
from .embedding import EmbeddingService, embedding_service as _default_embedding_service
from .milvus_client import MilvusManager
from .vector_backend import get_chroma_store, is_chroma


class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusManager = None):
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_manager = milvus_manager or MilvusManager()

    def write_documents(self, documents: list[dict], batch_size: int = 50):
        """
        批量写入文档到 Milvus（同时生成密集和稀疏向量）
        :param documents: 文档列表
        :param batch_size: 批次大小
        """
        if not documents:
            return

        if is_chroma():
            self._write_chroma(documents, batch_size)
            return

        self.milvus_manager.init_collection()

        all_texts = [doc["text"] for doc in documents]
        self.embedding_service.increment_add_documents(all_texts)

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]

            # 同时生成密集向量和稀疏向量
            dense_embeddings, sparse_embeddings = self.embedding_service.get_all_embeddings(texts)

            insert_data = [
                {
                    "dense_embedding": dense_emb,
                    "sparse_embedding": sparse_emb,
                    "text": doc["text"],
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                }
                for doc, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings)
            ]

            self.milvus_manager.insert(insert_data)

    def _write_chroma(self, documents: list[dict], batch_size: int) -> None:
        """Local Chroma (rag/main.ipynb style): dense vectors only; BM25 state not updated."""
        store = get_chroma_store()
        store.init_collection()
        total = len(documents)
        n_batches = (total + batch_size - 1) // batch_size
        print(f"[ingest/chroma] {total} leaf chunks in {n_batches} embedding batches (dense only, no BM25)")
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc["text"] for doc in batch]
            # Dense only — get_all_embeddings also builds sparse BM25 and is much slower; Chroma does not use sparse.
            dense_embeddings = self.embedding_service.get_embeddings(texts)
            bi = i // batch_size + 1
            print(f"[ingest/chroma] batch {bi}/{n_batches} ({len(batch)} chunks)")
            insert_data = [
                {
                    "dense_embedding": dense_emb,
                    "text": doc["text"],
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                }
                for doc, dense_emb in zip(batch, dense_embeddings)
            ]
            store.insert(insert_data)
