"""Minimal FastAPI service: ingest knowledge doc and run RAG graph."""
from __future__ import annotations

import os
import re
import socket
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
_RAG_FRONTEND = _ROOT.parent / "rag" / "frontend"
_UPLOAD_DIR = _ROOT / "data" / "uploads"
load_dotenv(_ROOT / ".env", override=True)

from .database import init_db  # noqa: E402
from .document_loader import DocumentLoader  # noqa: E402
from .embedding import embedding_service  # noqa: E402
from .milvus_client import MilvusManager  # noqa: E402
from .milvus_writer import MilvusWriter  # noqa: E402
from .parent_chunk_store import ParentChunkStore  # noqa: E402
from .vector_backend import get_chroma_store, is_chroma  # noqa: E402


def _default_kb_path() -> Path:
    override = (os.getenv("KNOWLEDGE_DOC_PATH") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (_ROOT.parent / "rag" / "doc.md").resolve()


def _safe_filename(name: str) -> str:
    base = Path(name).name
    if not base:
        return "upload.txt"
    base = re.sub(r"[^\w\.\-\u4e00-\u9fff]", "_", base)
    return base[:255] or "upload.txt"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="rag_new", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_loader = DocumentLoader()
_parent_chunk_store = ParentChunkStore()
_milvus_manager = MilvusManager()
_milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=_milvus_manager)


def _remove_bm25_stats_for_filename(filename: str) -> None:
    if is_chroma():
        return
    rows = _milvus_manager.query_all(
        filter_expr=f'filename == "{filename}"',
        output_fields=["text"],
    )
    texts = [r.get("text") or "" for r in rows]
    embedding_service.increment_remove_documents(texts)


def _load_chunks_for_path(src: Path, filename: str) -> list[dict]:
    lower = filename.lower()
    if lower.endswith((".md", ".markdown")):
        return _loader.load_markdown_file(str(src), filename=filename)
    if lower.endswith((".txt", ".text")):
        return _loader.load_text_file(str(src), filename=filename)
    raise HTTPException(
        status_code=400,
        detail="Unsupported file type (use .txt, .text, .md, .markdown). Same chunking pipeline as default KB.",
    )


def _ingest_pipeline(src: Path, filename: str, replace: bool) -> IngestResponse:
    if is_chroma():
        cs = get_chroma_store()
        cs.init_collection()
        if replace:
            try:
                cs.delete_by_filename(filename)
            except Exception:
                pass
            try:
                _parent_chunk_store.delete_by_filename(filename)
            except Exception:
                pass
    else:
        _milvus_manager.init_collection()
        if replace:
            try:
                _remove_bm25_stats_for_filename(filename)
            except Exception:
                pass
            try:
                _milvus_manager.delete(f'filename == "{filename}"')
            except Exception:
                pass
            try:
                _parent_chunk_store.delete_by_filename(filename)
            except Exception:
                pass

    new_docs = _load_chunks_for_path(src, filename)
    if not new_docs:
        raise HTTPException(status_code=400, detail="No content extracted from file")

    parent_docs = [d for d in new_docs if int(d.get("chunk_level", 0) or 0) in (1, 2)]
    leaf_docs = [d for d in new_docs if int(d.get("chunk_level", 0) or 0) == 3]
    if not leaf_docs:
        raise HTTPException(status_code=400, detail="No leaf (L3) chunks produced")

    print(
        f"[ingest] {filename}: {len(parent_docs)} parent + {len(leaf_docs)} leaf chunks, "
        f"vector_backend={'chroma' if is_chroma() else 'milvus'}"
    )
    _parent_chunk_store.upsert_documents(parent_docs)
    _milvus_writer.write_documents(leaf_docs)

    return IngestResponse(
        filename=filename,
        leaf_chunks=len(leaf_docs),
        parent_chunks=len(parent_docs),
        source_path=str(src),
        message=f"Ingested {filename}: {len(leaf_docs)} leaf chunks, {len(parent_docs)} parent chunks (SQLite).",
    )


class IngestRequest(BaseModel):
    path: str | None = Field(None, description="Absolute or relative path to Markdown; default: KNOWLEDGE_DOC_PATH or ../rag/doc.md")
    filename: str | None = Field(None, description="Logical filename for chunk ids / Milvus metadata (default: basename of path)")
    replace: bool = Field(True, description="Delete existing vectors + parent rows for this filename before insert")


class IngestResponse(BaseModel):
    filename: str
    leaf_chunks: int
    parent_chunks: int
    source_path: str
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    chunk_count: int


_MILVUS_HINT_ZH = (
    "未检测到 Milvus：在 rag_new 目录执行 docker compose up -d，等待约 1–2 分钟；"
    "确认本机 19530 端口可访问。也可在 .env 设置 MILVUS_URI 指向云端实例。"
    "若已装 Docker 仍失败，可尝试将 MILVUS_HOST=127.0.0.1（避免 localhost 走 IPv6）。"
)

_CHROMA_HINT_ZH = (
    "本地 Chroma 向量库（VECTOR_BACKEND=chroma），数据目录可写即可，无需 Docker / Milvus。"
)


def _milvus_reachable_quick() -> tuple[bool, str | None]:
    """Fast TCP check — tries 127.0.0.1 when host is localhost (Windows IPv6 quirks)."""
    uri = _milvus_manager.uri
    if uri.startswith("http://"):
        rest = uri.removeprefix("http://")
        host, _, port_s = rest.partition(":")
        port = int(port_s or "19530")
        hosts = [host]
        if host.lower() == "localhost":
            hosts = ["127.0.0.1", "::1", "localhost"]
        last_err: OSError | None = None
        timeout = float(os.getenv("MILVUS_HEALTH_TIMEOUT", "3"))
        for h in hosts:
            try:
                with socket.create_connection((h, port), timeout=timeout):
                    return True, None
            except OSError as e:
                last_err = e
                continue
        return False, str(last_err) if last_err else "connection failed"
    if Path(uri).suffix.lower() in (".db",):
        return Path(uri).parent.is_dir(), None
    return False, "unknown_uri_scheme"


def _chroma_persist_ok() -> tuple[bool, str | None]:
    cs = get_chroma_store()
    p = Path(cs.persist_path)
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".health_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, None
    except OSError as e:
        return False, str(e)


@app.get("/health")
async def health():
    if is_chroma():
        cs = get_chroma_store()
        milvus_ok, milvus_error = _chroma_persist_ok()
        return {
            "status": "ok",
            "vector_backend": "chroma",
            "milvus_uri": f"chroma:{cs.persist_path}",
            "collection": cs.collection_name,
            "milvus_ok": milvus_ok,
            "milvus_error": milvus_error,
            "milvus_hint": None if milvus_ok else (_CHROMA_HINT_ZH + (f" 错误：{milvus_error}" if milvus_error else "")),
        }
    milvus_ok, milvus_error = _milvus_reachable_quick()
    return {
        "status": "ok",
        "vector_backend": "milvus",
        "milvus_uri": _milvus_manager.uri,
        "collection": _milvus_manager.collection_name,
        "milvus_ok": milvus_ok,
        "milvus_error": milvus_error,
        "milvus_hint": None if milvus_ok else _MILVUS_HINT_ZH,
    }


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List ingested documents with leaf chunk counts (Chroma or Milvus)."""
    rows = _parent_chunk_store.list_distinct_filenames()
    out: list[DocumentInfo] = []
    for row in rows:
        fn = row["filename"]
        try:
            if is_chroma():
                n = get_chroma_store().count_by_filename(fn)
            else:
                n = _milvus_manager.count_entities_by_filename(fn)
        except Exception:
            n = 0
        out.append(DocumentInfo(filename=fn, file_type=row.get("file_type") or "", chunk_count=n))
    return out


@app.delete("/documents")
async def delete_document(filename: str = Query(..., min_length=1, description="Logical filename to remove from vector store + SQLite (+ BM25 if Milvus mode)")):
    try:
        if is_chroma():
            cs = get_chroma_store()
            cs.init_collection()
            try:
                cs.delete_by_filename(filename)
            except Exception:
                pass
        else:
            _milvus_manager.init_collection()
            try:
                _remove_bm25_stats_for_filename(filename)
            except Exception:
                pass
            try:
                _milvus_manager.delete(f'filename == "{filename}"')
            except Exception:
                pass
        deleted_parents = _parent_chunk_store.delete_by_filename(filename)
        return {"filename": filename, "parent_rows_deleted": deleted_parents, "message": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ingest/doc", response_model=IngestResponse)
async def ingest_doc(body: IngestRequest | None = None):
    body = body or IngestRequest()
    src = Path(body.path).expanduser().resolve() if body.path else _default_kb_path()
    if not src.is_file():
        raise HTTPException(status_code=404, detail=f"Knowledge file not found: {src}")

    filename = (body.filename or src.name).strip() or src.name

    try:
        return _ingest_pipeline(src, filename, body.replace)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    replace: bool = Query(True),
):
    """
    Upload a text or Markdown file; same three-level chunking + dense/sparse vectors as /ingest/doc.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    filename = _safe_filename(file.filename)
    lower = filename.lower()
    if not (lower.endswith((".txt", ".text", ".md", ".markdown"))):
        raise HTTPException(status_code=400, detail="Only .txt, .text, .md, .markdown are allowed")

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / filename

    try:
        raw = await file.read()
        text = raw.decode("utf-8", errors="replace")
        dest.write_text(text, encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}") from e

    try:
        return _ingest_pipeline(dest, filename, replace)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/rag/query")
async def rag_query(body: QueryRequest):
    from .rag_pipeline import run_rag_graph  # noqa: E402

    try:
        return run_rag_graph(body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/rag/complete")
async def rag_complete(body: QueryRequest):
    """Run retrieval graph, then synthesize an answer with the chat model when API key is set."""
    from langchain.chat_models import init_chat_model  # noqa: E402

    from .llm_env import API_KEY, MODEL, BASE_URL, OPENAI_COMPAT_EXTRA_BODY  # noqa: E402
    from .rag_pipeline import run_rag_graph  # noqa: E402

    try:
        state = run_rag_graph(body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rag_graph: {e}") from e

    context = (state.get("context") or "").strip()
    rag_trace = state.get("rag_trace") or {}
    docs = state.get("docs") or []

    if not context:
        return {
            "answer": "",
            "context": "",
            "docs": docs,
            "rag_trace": rag_trace,
            "message": "No context retrieved. Ingest documents and check /health (Chroma persist path or Milvus).",
        }

    if not API_KEY:
        return {
            "answer": "",
            "context": context,
            "docs": docs,
            "rag_trace": rag_trace,
            "message": "Set DASHSCOPE_API_KEY in .env to generate answers. Context is returned below.",
        }

    try:
        model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.3,
            stream_usage=True,
            extra_body=OPENAI_COMPAT_EXTRA_BODY,
        )
        prompt = (
            "You are a helpful assistant. Answer the user question using ONLY the provided context. "
            "If the context is insufficient, say so clearly. Prefer answering in the same language as the question.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{body.question}"
        )
        resp = model.invoke([{"role": "user", "content": prompt}])
        answer = (resp.content or "").strip()
        return {
            "answer": answer,
            "context": context,
            "docs": docs,
            "rag_trace": rag_trace,
            "message": "ok",
        }
    except Exception as e:
        return {
            "answer": "",
            "context": context,
            "docs": docs,
            "rag_trace": rag_trace,
            "message": f"LLM error: {e}",
        }


if _RAG_FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=str(_RAG_FRONTEND), html=True), name="frontend")
