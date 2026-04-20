"""Stub for rag_pipeline.emit_rag_step (optional telemetry hooks; no-op by default)."""

_RAG_STEP_QUEUE = None
_RAG_STEP_LOOP = None


def set_rag_step_queue(queue):
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    try:
        import asyncio
        _RAG_STEP_LOOP = asyncio.get_running_loop()
    except RuntimeError:
        _RAG_STEP_LOOP = None


def emit_rag_step(icon: str, label: str, detail: str | None = None):
    """No-op for CLI/API; could log if needed."""
    pass
