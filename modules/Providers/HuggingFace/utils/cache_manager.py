# modules/Providers/HuggingFace/utils/cache_manager.py

import os
import hashlib
import json
from typing import Any, Dict, Iterable, List

class CacheManager:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        if not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def get(self, key: str) -> str:
        return self.cache.get(key)

    def set(self, key: str, value: str):
        self.cache[key] = value
        self.save_cache()

    def _sanitize_for_cache(self, data: Any) -> Any:
        """Normalize data to a JSON-serializable structure for cache hashing."""
        if isinstance(data, dict):
            # Preserve key order deterministically by sorting keys
            return {key: self._sanitize_for_cache(data[key]) for key in sorted(data.keys())}
        if isinstance(data, list):
            return [self._sanitize_for_cache(item) for item in data]
        if isinstance(data, tuple):
            return [self._sanitize_for_cache(item) for item in data]
        if isinstance(data, set):
            return [self._sanitize_for_cache(item) for item in sorted(data, key=str)]
        if isinstance(data, bytes):
            return f"<binary:{len(data)}bytes>"
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data
        if isinstance(data, Iterable):
            return [self._sanitize_for_cache(item) for item in data]
        return str(data)

    def generate_cache_key(self, messages: List[Dict[str, Any]], model: str, settings: Dict) -> str:
        sanitized_messages = self._sanitize_for_cache(messages)
        sanitized_settings = self._sanitize_for_cache(settings)
        cache_payload = {"messages": sanitized_messages, "model": model, "settings": sanitized_settings}
        cache_data = json.dumps(cache_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.md5(cache_data.encode()).hexdigest()
