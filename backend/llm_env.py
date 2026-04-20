"""Bailian (DashScope) OpenAI-compatible API settings for LangChain chat models."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# Optional for ingest-only workflows; grading / query rewrite in rag_pipeline need a key at runtime.
DASHSCOPE_API_KEY = (os.getenv("DASHSCOPE_API_KEY") or "").strip()

# Beijing region compatible-mode endpoint; use dashscope-intl.aliyuncs.com for Singapore.
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"

def _env_or_default(key: str, default: str) -> str:
    raw = (os.getenv(key) or "").strip()
    return raw if raw else default


BASE_URL = _env_or_default("BASE_URL", DEFAULT_BASE_URL)
MODEL = _env_or_default("MODEL", DEFAULT_MODEL)
GRADE_MODEL = _env_or_default("GRADE_MODEL", DEFAULT_MODEL)

# Names used by existing backend modules
API_KEY = DASHSCOPE_API_KEY
ARK_API_KEY = DASHSCOPE_API_KEY

# DashScope OpenAI-compatible API (Qwen3): non-streaming invoke() requires enable_thinking=false or HTTP 400.
OPENAI_COMPAT_EXTRA_BODY = {"enable_thinking": False}
