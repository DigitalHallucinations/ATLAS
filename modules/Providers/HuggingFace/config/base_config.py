# modules/Providers/HuggingFace/config/base_config.py

import logging
import os
from typing import Any, Dict, Mapping


class BaseConfig:
    """Holds generation defaults for Hugging Face models."""

    DEFAULT_MODEL_SETTINGS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 50,
        "max_tokens": 100,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "early_stopping": False,
        "do_sample": False,
    }

    def __init__(self, config_manager):
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self.model_cache_dir = self.config_manager.get_model_cache_dir()
        os.makedirs(self.model_cache_dir, exist_ok=True)

        persisted: Mapping[str, Any] = {}
        getter = getattr(self.config_manager, "get_huggingface_generation_settings", None)
        if callable(getter):
            try:
                candidate = getter()
                if isinstance(candidate, Mapping):
                    persisted = dict(candidate)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning(
                    "Failed to load persisted HuggingFace generation settings: %s", exc
                )

        self.model_settings: Dict[str, Any] = dict(self.DEFAULT_MODEL_SETTINGS)
        for key, value in persisted.items():
            if key in self.DEFAULT_MODEL_SETTINGS:
                try:
                    self.model_settings[key] = self._validate_setting(key, value)
                except ValueError:
                    continue

    def _validate_setting(self, key: str, value: Any) -> Any:
        """Normalise and validate a single generation setting."""

        if value is None:
            raise ValueError(f"Setting '{key}' cannot be None")

        if key == "temperature":
            normalized = float(value)
            if not 0.0 <= normalized <= 2.0:
                raise ValueError("Temperature must be between 0.0 and 2.0")
            return normalized
        if key == "top_p":
            normalized = float(value)
            if not 0.0 <= normalized <= 1.0:
                raise ValueError("Top-p must be between 0.0 and 1.0")
            return normalized
        if key == "top_k":
            normalized = int(value)
            if normalized < 0:
                raise ValueError("Top-k must be greater than or equal to 0")
            return normalized
        if key == "max_tokens":
            normalized = int(value)
            if normalized <= 0:
                raise ValueError("Max tokens must be a positive integer")
            return normalized
        if key in {"presence_penalty", "frequency_penalty"}:
            normalized = float(value)
            if not -2.0 <= normalized <= 2.0:
                raise ValueError(f"{key.replace('_', ' ').title()} must be between -2.0 and 2.0")
            return normalized
        if key == "repetition_penalty":
            normalized = float(value)
            if normalized <= 0:
                raise ValueError("Repetition penalty must be greater than 0")
            return normalized
        if key == "length_penalty":
            normalized = float(value)
            if normalized < 0:
                raise ValueError("Length penalty must be non-negative")
            return normalized
        if key in {"early_stopping", "do_sample"}:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "on"}:
                    return True
                if normalized in {"false", "0", "no", "off"}:
                    return False
            return bool(value)

        raise ValueError(f"Unsupported setting '{key}'")

    def update_model_settings(self, new_settings: Mapping[str, Any]):
        """Update generation defaults with validation and persistence."""

        if not isinstance(new_settings, Mapping):
            raise ValueError("Model settings must be provided as a mapping")

        updated: Dict[str, Any] = {}
        for key in self.DEFAULT_MODEL_SETTINGS:
            if key not in new_settings:
                continue
            try:
                updated[key] = self._validate_setting(key, new_settings[key])
            except ValueError as exc:
                raise ValueError(f"Invalid value for '{key}': {exc}") from exc

        if not updated:
            return

        self.model_settings.update(updated)

        setter = getattr(self.config_manager, "set_huggingface_generation_settings", None)
        if callable(setter):
            try:
                setter(self.model_settings)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning(
                    "Failed to persist HuggingFace generation settings: %s", exc
                )
