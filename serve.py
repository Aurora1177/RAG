"""Run uvicorn with backend on PYTHONPATH."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    # ProactorEventLoop on Windows can surface ConnectionResetError in asyncio callbacks when
    # clients close the socket first; SelectorEventLoop avoids most of that (common uvicorn tip).
    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    reload = os.getenv("RELOAD", "").lower() in ("1", "true", "yes")
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info").lower()

    uvicorn.run(
        "backend.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
