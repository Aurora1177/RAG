"""In-process cache (replaces Redis for rag_new)."""
import json
from typing import Any, Optional


class _MemoryCache:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get_json(self, key: str) -> Optional[Any]:
        raw = self._data.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self._data[key] = json.dumps(value, ensure_ascii=False)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)


cache = _MemoryCache()
