"""Provider-focused configuration sections for :mod:`ATLAS.config`."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

from modules.Providers.Google.settings_resolver import GoogleSettingsResolver

PROVIDER_UNSET = object()

if False:  # pragma: no cover - circular import protection for type checkers
    from .config_manager import ConfigManager


class ProviderConfigSections:
    """Container managing provider-specific configuration helpers."""

    def __init__(self, manager: "ConfigManager") -> None:
        self.manager = manager
        self.logger = manager.logger
        self._env_keys: Dict[str, str] = {
            "OpenAI": "OPENAI_API_KEY",
            "Mistral": "MISTRAL_API_KEY",
            "Google": "GOOGLE_API_KEY",
            "HuggingFace": "HUGGINGFACE_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Grok": "GROK_API_KEY",
            "ElevenLabs": "XI_API_KEY",
        }
        self._provider_env_lookup: Dict[str, str] = {
            env_key: provider for provider, env_key in self._env_keys.items()
        }
        self._pending_provider_warnings: Dict[str, str] = {}

        self.openai = OpenAIProviderConfig(manager=manager, registry=self)
        self.google = GoogleProviderConfig(manager=manager, registry=self)
        self.mistral = MistralProviderConfig(manager=manager, registry=self)

    def apply(self) -> None:
        """Apply provider-specific post-load normalization."""

        self.openai.apply()

    # ------------------------------------------------------------------
    # Credential helpers
    # ------------------------------------------------------------------
    def get_env_keys(self) -> Dict[str, str]:
        """Return the mapping between provider display names and env keys."""

        return dict(self._env_keys)

    def sync_provider_warning(self, env_key: str, value: Optional[str]) -> None:
        """Refresh pending provider warnings when credential values change."""

        provider_name = self._provider_env_lookup.get(env_key)
        if not provider_name:
            for provider, candidate in self._env_keys.items():
                if candidate == env_key:
                    provider_name = provider
                    self._provider_env_lookup[env_key] = provider
                    break

        if not provider_name:
            return

        if value:
            self._pending_provider_warnings.pop(provider_name, None)
            return

        warning_message = (
            f"API key for provider '{provider_name}' is not configured. "
            "Protected features will remain unavailable until a key is provided."
        )
        self._pending_provider_warnings[provider_name] = warning_message

    def initialize_pending_warnings(self) -> None:
        """Prime deferred credential warnings for the default provider."""

        default_provider = self.manager.config.get("DEFAULT_PROVIDER", "OpenAI")
        if not self.is_api_key_set(default_provider):
            warning_message = (
                f"API key for provider '{default_provider}' is not configured. "
                "Protected features will remain unavailable until a key is provided."
            )
            self.logger.warning(warning_message)
            self._pending_provider_warnings[default_provider] = warning_message

    def get_pending_provider_warnings(self) -> Dict[str, str]:
        """Return provider credential warnings that should be surfaced."""

        return dict(self._pending_provider_warnings)

    def is_default_provider_ready(self) -> bool:
        """Return ``True`` when the configured default provider has credentials."""

        default_provider = self.manager.get_default_provider()
        if not default_provider:
            return True
        return default_provider not in self._pending_provider_warnings

    def is_api_key_set(self, provider_name: str) -> bool:
        """Return ``True`` if an API key is configured for the provider."""

        env_key = self._env_keys.get(provider_name)
        if not env_key:
            return False

        api_key = self.manager.get_config(env_key)
        return bool(api_key)

    def update_api_key(self, provider_name: str, new_api_key: str) -> None:
        """Update the API key for a provider and refresh environment state."""

        env_key = self._env_keys.get(provider_name)
        if not env_key:
            raise ValueError(
                f"No API key mapping found for provider '{provider_name}'."
            )

        self.manager._persist_env_value(env_key, new_api_key)
        self.logger.info("API key for %s updated successfully.", provider_name)

    def has_provider_api_key(self, provider_name: str) -> bool:
        """Determine whether an API key is configured for the provider."""

        return self.is_api_key_set(provider_name)

    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Return metadata describing configured providers without secrets."""

        providers: Dict[str, Dict[str, Any]] = {}

        for provider, env_key in self._env_keys.items():
            value = self.manager.get_config(env_key)
            if value is None:
                secret = ""
            elif isinstance(value, str):
                secret = value
            else:
                secret = str(value)

            available = bool(secret)
            length = len(secret) if available else 0
            hint = (
                self.manager._mask_secret_preview(secret) if available else ""
            )

            providers[provider] = {
                "available": available,
                "length": length,
                "hint": hint,
            }

        return providers

    # ------------------------------------------------------------------
    # LLM fallback helpers
    # ------------------------------------------------------------------
    def get_llm_fallback_config(self) -> Dict[str, Any]:
        """Return the configured fallback provider settings with defaults."""

        fallback_block: Dict[str, Any] = {}
        stored = self.manager.get_config("LLM_FALLBACK")
        if isinstance(stored, Mapping):
            fallback_block.update(stored)

        env_provider = self.manager.get_config("LLM_FALLBACK_PROVIDER")
        if isinstance(env_provider, str) and env_provider.strip():
            fallback_block["provider"] = env_provider.strip()

        env_model = self.manager.get_config("LLM_FALLBACK_MODEL")
        if isinstance(env_model, str) and env_model.strip():
            fallback_block["model"] = env_model.strip()

        provider = (
            fallback_block.get("provider")
            or self.manager.get_default_provider()
            or "OpenAI"
        )
        fallback_block["provider"] = provider

        defaults_lookup: Dict[str, Any] = {
            "OpenAI": self.openai.get_llm_settings,
            "Mistral": self.mistral.get_llm_settings,
            "Google": self.google.get_llm_settings,
            "Anthropic": self.manager.get_anthropic_settings,
        }

        getter = defaults_lookup.get(provider)
        provider_defaults = getter() if callable(getter) else {}

        merged: Dict[str, Any] = copy.deepcopy(provider_defaults)
        merged.update(fallback_block)

        if not merged.get("model"):
            default_model = (
                provider_defaults.get("model")
                if isinstance(provider_defaults, Mapping)
                else None
            )
            if not default_model:
                default_model = self.manager.get_default_model()
            merged["model"] = default_model

        if "stream" not in merged and isinstance(provider_defaults, Mapping):
            merged["stream"] = provider_defaults.get("stream", True)

        if "max_tokens" not in merged and isinstance(provider_defaults, Mapping):
            merged["max_tokens"] = provider_defaults.get("max_tokens")

        return merged


class OpenAIProviderConfig:
    """Provider section encapsulating OpenAI settings helpers."""

    def __init__(self, *, manager: "ConfigManager", registry: ProviderConfigSections) -> None:
        self.manager = manager
        self.registry = registry
        self.logger = manager.logger

    def apply(self) -> None:
        """Apply persisted OpenAI speech configuration to the active config."""

        self._synchronize_speech_block()

    def set_speech_config(
        self,
        *,
        api_key: Optional[str] = None,
        stt_provider: Optional[str] = None,
        tts_provider: Optional[str] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> None:
        """Persist OpenAI speech configuration values."""

        if api_key is not None:
            if not api_key:
                raise ValueError("OpenAI API key cannot be empty.")
            self.manager._persist_env_value("OPENAI_API_KEY", api_key)
            self.logger.info("OpenAI API key updated for speech services.")

        config_updates = {
            "OPENAI_STT_PROVIDER": stt_provider,
            "OPENAI_TTS_PROVIDER": tts_provider,
            "OPENAI_LANGUAGE": language,
            "OPENAI_TASK": task,
            "OPENAI_INITIAL_PROMPT": initial_prompt,
        }

        for key, value in config_updates.items():
            if value is not None:
                self.manager.config[key] = value
            elif key in self.manager.config:
                self.manager.config[key] = None

        block: Dict[str, Any] = {}
        existing = self.manager.yaml_config.get("OPENAI_SPEECH")
        if isinstance(existing, Mapping):
            block.update(existing)

        block_updates = {
            "stt_provider": stt_provider,
            "tts_provider": tts_provider,
            "language": language,
            "task": task,
            "initial_prompt": initial_prompt,
        }

        for block_key, value in block_updates.items():
            if value is None:
                block.pop(block_key, None)
            else:
                block[block_key] = value

        if block:
            self.manager.yaml_config["OPENAI_SPEECH"] = block
            self.manager.config["OPENAI_SPEECH"] = dict(block)
        else:
            self.manager.yaml_config.pop("OPENAI_SPEECH", None)
            self.manager.config.pop("OPENAI_SPEECH", None)

        self.manager._write_yaml_config()
        self._synchronize_speech_block()

    def _synchronize_speech_block(self) -> None:
        """Sync the persisted OpenAI speech YAML block into the in-memory config."""

        block = self.manager.yaml_config.get("OPENAI_SPEECH")

        if not isinstance(block, Mapping):
            self.manager.config.pop("OPENAI_SPEECH", None)
            return

        normalized = dict(block)
        self.manager.config["OPENAI_SPEECH"] = normalized

        mapping = {
            "OPENAI_STT_PROVIDER": "stt_provider",
            "OPENAI_TTS_PROVIDER": "tts_provider",
            "OPENAI_LANGUAGE": "language",
            "OPENAI_TASK": "task",
            "OPENAI_INITIAL_PROMPT": "initial_prompt",
        }

        for config_key, block_key in mapping.items():
            if block_key in normalized:
                self.manager.config[config_key] = normalized[block_key]

    def get_llm_settings(self) -> Dict[str, Any]:
        """Return persisted OpenAI LLM defaults merged with environment values."""

        defaults = {
            "model": self.manager.get_config("DEFAULT_MODEL", "gpt-4o"),
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 4000,
            "max_output_tokens": None,
            "stream": True,
            "function_calling": True,
            "parallel_tool_calls": True,
            "tool_choice": None,
            "reasoning_effort": "medium",
            "base_url": self.manager.get_config("OPENAI_BASE_URL"),
            "organization": self.manager.get_config("OPENAI_ORGANIZATION"),
            "json_mode": False,
            "json_schema": None,
            "enable_code_interpreter": False,
            "enable_file_search": False,
            "audio_enabled": False,
            "audio_voice": "alloy",
            "audio_format": "wav",
        }

        stored = self.manager.get_config("OPENAI_LLM")
        if isinstance(stored, dict):
            defaults.update({k: stored.get(k, defaults.get(k)) for k in defaults.keys()})

        return defaults

    def set_llm_settings(
        self,
        *,
        model: Optional[str],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        function_calling: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Optional[str] = None,
        enable_code_interpreter: Optional[bool] = None,
        enable_file_search: Optional[bool] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        json_mode: Optional[Any] = None,
        json_schema: Optional[Any] = None,
        audio_enabled: Optional[bool] = None,
        audio_voice: Optional[str] = None,
        audio_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist OpenAI chat-completion defaults and related metadata."""

        if not model:
            raise ValueError("A default OpenAI model must be provided.")

        normalized_temperature = 0.0 if temperature is None else float(temperature)
        if not 0.0 <= normalized_temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0.")

        normalized_top_p = 1.0 if top_p is None else float(top_p)
        if not 0.0 <= normalized_top_p <= 1.0:
            raise ValueError("Top-p must be between 0.0 and 1.0.")

        normalized_frequency_penalty = (
            0.0 if frequency_penalty is None else float(frequency_penalty)
        )
        if not -2.0 <= normalized_frequency_penalty <= 2.0:
            raise ValueError("Frequency penalty must be between -2.0 and 2.0.")

        normalized_presence_penalty = (
            0.0 if presence_penalty is None else float(presence_penalty)
        )
        if not -2.0 <= normalized_presence_penalty <= 2.0:
            raise ValueError("Presence penalty must be between -2.0 and 2.0.")

        normalized_max_tokens = None if max_tokens is None else int(max_tokens)
        if normalized_max_tokens is not None and normalized_max_tokens <= 0:
            raise ValueError("Max tokens must be greater than zero.")

        normalized_max_output_tokens = (
            None if max_output_tokens in {None, ""} else int(max_output_tokens)
        )
        if (
            normalized_max_output_tokens is not None
            and normalized_max_output_tokens <= 0
        ):
            raise ValueError("Max output tokens must be greater than zero.")

        normalized_stream = (
            bool(stream)
            if stream is not None
            else bool(self.manager.config.get("OPENAI_LLM", {}).get("stream", True))
        )

        normalized_function_calling = (
            bool(function_calling)
            if function_calling is not None
            else bool(
                self.manager.config.get("OPENAI_LLM", {}).get(
                    "function_calling", True
                )
            )
        )

        def _normalize_reasoning_effort(value: Optional[str]) -> str:
            if value is None:
                return str(
                    self.manager.config.get("OPENAI_LLM", {}).get(
                        "reasoning_effort", "medium"
                    )
                )
            cleaned = value.strip().lower()
            if cleaned not in {"low", "medium", "high"}:
                raise ValueError(
                    "Reasoning effort must be one of: low, medium, or high."
                )
            return cleaned

        normalized_reasoning_effort = _normalize_reasoning_effort(reasoning_effort)

        def _normalize_json_mode(value: Any, existing: bool) -> bool:
            if value is None:
                return existing
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if not normalized:
                    return existing
                if normalized in {"1", "true", "yes", "on", "json", "json_object"}:
                    return True
                if normalized in {"0", "false", "no", "off", "text", "none"}:
                    return False
                return existing
            try:
                return bool(value)
            except Exception:
                return existing

        def _normalize_json_schema(
            value: Any, existing: Optional[Dict[str, Any]]
        ) -> Optional[Dict[str, Any]]:
            if value is None:
                return existing

            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8")

            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return None
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON schema must be valid JSON: {exc.msg}"
                    ) from exc

            if value is False:
                return None

            if not isinstance(value, dict):
                raise ValueError(
                    "JSON schema must be provided as an object or JSON string."
                )

            if not value:
                return None

            schema_payload = value.get("schema") if isinstance(value, dict) else None
            schema_name = value.get("name") if isinstance(value, dict) else None

            if schema_payload is None:
                schema_payload = value
                schema_like_keys = {
                    "$schema",
                    "$ref",
                    "type",
                    "properties",
                    "items",
                    "oneOf",
                    "anyOf",
                    "allOf",
                    "definitions",
                    "patternProperties",
                }
                if isinstance(schema_payload, dict) and not (
                    schema_like_keys & set(schema_payload.keys())
                ):
                    raise ValueError(
                        "JSON schema must include a 'schema' object or a valid schema definition."
                    )

            if not isinstance(schema_payload, dict):
                raise ValueError(
                    "The 'schema' entry for the JSON schema must be an object."
                )

            if schema_name is None:
                if isinstance(existing, dict):
                    schema_name = existing.get("name")
                if not schema_name:
                    schema_name = "atlas_response"

            normalized: Dict[str, Any] = {
                "name": str(schema_name).strip() or "atlas_response",
                "schema": schema_payload,
            }

            strict_value = value.get("strict") if isinstance(value, dict) else None
            if strict_value is None and isinstance(existing, dict):
                strict_value = existing.get("strict")

            if strict_value is not None:
                normalized["strict"] = bool(strict_value)

            try:
                return json.loads(json.dumps(normalized))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"JSON schema contains non-serializable content: {exc}"
                ) from exc

        sanitized_base_url = (base_url or "").strip() or None
        sanitized_org = (organization or "").strip() or None

        settings_block: Dict[str, Any] = {}
        existing = self.manager.yaml_config.get("OPENAI_LLM")
        if isinstance(existing, dict):
            settings_block.update(existing)

        previous_json_mode = bool(settings_block.get("json_mode", False))
        normalized_json_mode = _normalize_json_mode(json_mode, previous_json_mode)

        previous_schema = settings_block.get("json_schema")
        if not isinstance(previous_schema, dict):
            previous_schema = None
        normalized_json_schema = _normalize_json_schema(json_schema, previous_schema)

        previous_parallel_tool_calls = bool(
            settings_block.get("parallel_tool_calls", True)
        )
        normalized_parallel_tool_calls = (
            previous_parallel_tool_calls
            if parallel_tool_calls is None
            else bool(parallel_tool_calls)
        )

        previous_code_interpreter = bool(
            settings_block.get("enable_code_interpreter", False)
        )
        normalized_code_interpreter = (
            previous_code_interpreter
            if enable_code_interpreter is None
            else bool(enable_code_interpreter)
        )

        previous_file_search = bool(settings_block.get("enable_file_search", False))
        normalized_file_search = (
            previous_file_search
            if enable_file_search is None
            else bool(enable_file_search)
        )

        previous_audio_enabled = bool(settings_block.get("audio_enabled", False))
        normalized_audio_enabled = (
            previous_audio_enabled
            if audio_enabled is None
            else bool(audio_enabled)
        )

        def _normalize_audio_string(
            value: Optional[str], existing: Optional[str]
        ) -> Optional[str]:
            if value is None:
                return existing

            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    return None
                return cleaned

            return existing

        previous_voice = settings_block.get("audio_voice")
        previous_format = settings_block.get("audio_format")

        normalized_audio_voice = _normalize_audio_string(audio_voice, previous_voice)

        normalized_audio_format = _normalize_audio_string(audio_format, previous_format)
        if isinstance(normalized_audio_format, str):
            normalized_audio_format = normalized_audio_format.lower()

        def _normalize_tool_choice(
            value: Optional[str], existing_value: Optional[str]
        ) -> Optional[str]:
            if value is None:
                return existing_value

            if isinstance(value, str):
                normalized_value = value.strip().lower()
                if not normalized_value:
                    return None
                if normalized_value in {"auto", "none", "required"}:
                    return normalized_value

            return existing_value

        normalized_tool_choice = _normalize_tool_choice(
            tool_choice, settings_block.get("tool_choice")
        )

        if not normalized_function_calling:
            normalized_code_interpreter = False
            normalized_file_search = False

        settings_block.update(
            {
                "model": model,
                "temperature": normalized_temperature,
                "top_p": normalized_top_p,
                "frequency_penalty": normalized_frequency_penalty,
                "presence_penalty": normalized_presence_penalty,
                "max_tokens": normalized_max_tokens,
                "max_output_tokens": normalized_max_output_tokens,
                "stream": normalized_stream,
                "function_calling": normalized_function_calling,
                "parallel_tool_calls": normalized_parallel_tool_calls,
                "tool_choice": normalized_tool_choice,
                "reasoning_effort": normalized_reasoning_effort,
                "base_url": sanitized_base_url,
                "organization": sanitized_org,
                "json_mode": normalized_json_mode,
                "json_schema": normalized_json_schema,
                "enable_code_interpreter": normalized_code_interpreter,
                "enable_file_search": normalized_file_search,
                "audio_enabled": normalized_audio_enabled,
                "audio_voice": normalized_audio_voice,
                "audio_format": normalized_audio_format,
            }
        )

        self.manager.yaml_config["OPENAI_LLM"] = settings_block
        self.manager.config["OPENAI_LLM"] = dict(settings_block)

        self.manager._persist_env_value("DEFAULT_MODEL", model)
        self.manager.config["DEFAULT_MODEL"] = model

        self.manager._persist_env_value("OPENAI_BASE_URL", sanitized_base_url)
        self.manager._persist_env_value("OPENAI_ORGANIZATION", sanitized_org)

        self.manager.env_config["DEFAULT_MODEL"] = model
        self.manager.env_config["OPENAI_BASE_URL"] = sanitized_base_url
        self.manager.env_config["OPENAI_ORGANIZATION"] = sanitized_org

        self.manager._write_yaml_config()

        return dict(settings_block)


class GoogleProviderConfig:
    """Provider section encapsulating Google Gemini helpers."""

    def __init__(self, *, manager: "ConfigManager", registry: ProviderConfigSections) -> None:
        self.manager = manager
        self.registry = registry
        self.logger = manager.logger
        self._unset = manager.UNSET

    # ------------------------------------------------------------------
    # Credential helpers
    # ------------------------------------------------------------------
    def set_credentials(self, credentials_path: str) -> None:
        """Persist Google application credentials and refresh process state."""

        if not credentials_path:
            raise ValueError("Google credentials path cannot be empty.")

        self.manager._persist_env_value("GOOGLE_APPLICATION_CREDENTIALS", credentials_path)
        self.logger.info("Google credentials path updated.")

    # ------------------------------------------------------------------
    # Speech helpers
    # ------------------------------------------------------------------
    def get_speech_settings(self) -> Dict[str, Any]:
        """Return persisted Google speech preferences when available."""

        block = self.manager.yaml_config.get("GOOGLE_SPEECH")
        if not isinstance(block, dict):
            block = {}

        settings = {
            "tts_voice": block.get("tts_voice"),
            "stt_language": block.get("stt_language"),
            "auto_punctuation": block.get("auto_punctuation"),
        }

        if settings["auto_punctuation"] is not None:
            settings["auto_punctuation"] = bool(settings["auto_punctuation"])

        return settings

    def set_speech_settings(
        self,
        *,
        tts_voice: Optional[str] = None,
        stt_language: Optional[str] = None,
        auto_punctuation: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Persist Google speech preferences to the YAML configuration."""

        block: Dict[str, Any] = {}
        existing = self.manager.yaml_config.get("GOOGLE_SPEECH")
        if isinstance(existing, dict):
            block.update(existing)

        if tts_voice is None:
            block.pop("tts_voice", None)
        else:
            block["tts_voice"] = tts_voice

        if stt_language is None:
            block.pop("stt_language", None)
        else:
            block["stt_language"] = stt_language

        if auto_punctuation is None:
            block.pop("auto_punctuation", None)
        else:
            block["auto_punctuation"] = bool(auto_punctuation)

        if block:
            self.manager.yaml_config["GOOGLE_SPEECH"] = block
            self.manager.config["GOOGLE_SPEECH"] = dict(block)
        else:
            self.manager.yaml_config.pop("GOOGLE_SPEECH", None)
            self.manager.config.pop("GOOGLE_SPEECH", None)

        self.manager._write_yaml_config()
        return self.get_speech_settings()

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def get_llm_settings(self) -> Dict[str, Any]:
        """Return the persisted Google LLM defaults, if configured."""

        defaults: Dict[str, Any] = {
            "stream": True,
            "function_calling": True,
            "function_call_mode": "auto",
            "allowed_function_names": [],
            "cached_allowed_function_names": [],
            "response_schema": {},
            "seed": None,
            "response_logprobs": False,
        }

        settings = self.manager.yaml_config.get("GOOGLE_LLM")
        if isinstance(settings, dict):
            normalized = copy.deepcopy(settings)
        else:
            normalized = {}

        merged: Dict[str, Any] = copy.deepcopy(defaults)
        merged.update(normalized)
        resolver = GoogleSettingsResolver(stored=normalized, defaults=defaults)
        merged["stream"] = resolver.resolve_bool("stream", None, default=True)
        merged["function_calling"] = resolver.resolve_bool(
            "function_calling",
            None,
            default=True,
        )
        merged["seed"] = resolver.resolve_seed(None, allow_invalid_stored=True)
        merged["response_logprobs"] = resolver.resolve_bool(
            "response_logprobs",
            None,
            default=False,
        )

        merged["function_call_mode"] = self._coerce_function_call_mode(
            normalized.get("function_call_mode"),
            default="auto",
        )
        try:
            merged["allowed_function_names"] = self._coerce_allowed_function_names(
                normalized.get("allowed_function_names")
            )
        except ValueError:
            merged["allowed_function_names"] = []

        try:
            merged["cached_allowed_function_names"] = self._coerce_allowed_function_names(
                normalized.get("cached_allowed_function_names")
            )
        except ValueError:
            merged["cached_allowed_function_names"] = []

        try:
            merged["response_schema"] = resolver.resolve_response_schema(
                None,
                allow_invalid_stored=True,
            )
        except ValueError as exc:
            self.logger.warning(
                "Ignoring persisted Google response schema due to validation error: %s",
                exc,
            )
            merged["response_schema"] = {}

        return merged

    def set_llm_settings(
        self,
        *,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[Any] = None,
        candidate_count: Optional[Any] = None,
        max_output_tokens: Optional[Any] = None,
        stop_sequences: Optional[Any] = None,
        safety_settings: Optional[Any] = None,
        response_mime_type: Optional[str] = None,
        system_instruction: Optional[str] = None,
        stream: Optional[bool] = None,
        function_calling: Optional[bool] = None,
        function_call_mode: Optional[str] = None,
        allowed_function_names: Optional[Any] = None,
        response_schema: Optional[Any] = None,
        cached_allowed_function_names: Any = PROVIDER_UNSET,
        seed: Optional[Any] = None,
        response_logprobs: Optional[bool] = None,
        base_url: Any = PROVIDER_UNSET,
    ) -> Dict[str, Any]:
        """Persist default configuration for the Google Gemini provider."""

        if not isinstance(model, str) or not model.strip():
            raise ValueError("A default Google model must be provided.")

        defaults = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "candidate_count": 1,
            "max_output_tokens": 32000,
            "stop_sequences": [],
            "safety_settings": [],
            "response_mime_type": None,
            "system_instruction": None,
            "stream": True,
            "function_calling": True,
            "function_call_mode": "auto",
            "allowed_function_names": [],
            "cached_allowed_function_names": [],
            "response_schema": {},
            "seed": None,
            "response_logprobs": False,
        }

        existing_settings: Dict[str, Any] = {}
        stored_block = self.manager.yaml_config.get("GOOGLE_LLM")
        if isinstance(stored_block, dict):
            existing_settings = copy.deepcopy(stored_block)

        settings_block: Dict[str, Any] = copy.deepcopy(defaults)
        settings_block.update(existing_settings)
        settings_block["model"] = model.strip()

        def _normalize_stop_sequences(
            value: Optional[Any],
            previous: Any,
        ) -> List[str]:
            if value is None:
                if isinstance(previous, list):
                    return list(previous)
                if previous in {None, ""}:
                    return []
                return self.manager._coerce_stop_sequences(previous)
            return self.manager._coerce_stop_sequences(value)

        def _coerce_safety_settings(
            value: Optional[Any],
            previous: Any,
        ) -> List[Dict[str, str]]:
            if value is None:
                return copy.deepcopy(previous) if isinstance(previous, list) else []

            if value in ({}, []):
                return []

            normalized: List[Dict[str, str]] = []

            if isinstance(value, Mapping):
                for category, threshold in value.items():
                    cleaned_category = str(category).strip()
                    if not cleaned_category:
                        continue
                    if threshold in {None, ""}:
                        raise ValueError("Safety setting threshold cannot be empty.")
                    normalized.append(
                        {
                            "category": cleaned_category,
                            "threshold": str(threshold).strip(),
                        }
                    )
                return normalized

            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for entry in value:
                    if entry is None:
                        continue
                    if isinstance(entry, str) and not entry.strip():
                        continue
                    if not isinstance(entry, Mapping):
                        raise ValueError(
                            "Safety settings must be provided as mappings or mapping sequences."
                        )
                    category = (
                        entry.get("category")
                        or entry.get("harmCategory")
                        or entry.get("name")
                    )
                    threshold = (
                        entry.get("threshold")
                        or entry.get("thresholdValue")
                        or entry.get("harmBlockThreshold")
                    )
                    if not category:
                        raise ValueError(
                            "Safety setting entries must include a category value."
                        )
                    if threshold in {None, ""}:
                        raise ValueError(
                            "Safety setting entries must include a threshold value."
                        )
                    normalized.append(
                        {
                            "category": str(category).strip(),
                            "threshold": str(threshold).strip(),
                        }
                    )
                return normalized

            raise ValueError(
                "Safety settings must be provided as a mapping or list of mappings."
            )

        def _normalize_optional_text(
            value: Optional[Any],
            previous: Optional[str],
        ) -> Optional[str]:
            if value is None:
                return previous
            cleaned = str(value).strip()
            return cleaned or None

        resolver = GoogleSettingsResolver(
            stored=existing_settings,
            defaults=defaults,
        )

        settings_block["temperature"] = resolver.resolve_float(
            "temperature",
            temperature,
            field="Temperature",
            minimum=0.0,
            maximum=2.0,
        )
        settings_block["top_p"] = resolver.resolve_float(
            "top_p",
            top_p,
            field="Top-p",
            minimum=0.0,
            maximum=1.0,
        )
        settings_block["top_k"] = resolver.resolve_optional_int(
            "top_k",
            top_k,
            field="Top-k",
            minimum=1,
        )
        settings_block["candidate_count"] = resolver.resolve_int(
            "candidate_count",
            candidate_count,
            field="Candidate count",
            minimum=1,
        )

        settings_block["stop_sequences"] = _normalize_stop_sequences(
            stop_sequences,
            settings_block.get("stop_sequences", defaults["stop_sequences"]),
        )
        settings_block["safety_settings"] = _coerce_safety_settings(
            safety_settings,
            settings_block.get("safety_settings", defaults["safety_settings"]),
        )
        settings_block["response_mime_type"] = _normalize_optional_text(
            response_mime_type,
            settings_block.get("response_mime_type", defaults["response_mime_type"]),
        )
        settings_block["system_instruction"] = _normalize_optional_text(
            system_instruction,
            settings_block.get("system_instruction", defaults["system_instruction"]),
        )
        settings_block["stream"] = resolver.resolve_bool(
            "stream",
            stream,
            default=defaults["stream"],
        )
        settings_block["function_calling"] = resolver.resolve_bool(
            "function_calling",
            function_calling,
            default=defaults["function_calling"],
        )
        settings_block["seed"] = resolver.resolve_seed(
            seed,
            allow_invalid_stored=True,
        )
        settings_block["response_logprobs"] = resolver.resolve_bool(
            "response_logprobs",
            response_logprobs,
            default=defaults["response_logprobs"],
        )

        settings_block["function_call_mode"] = self._coerce_function_call_mode(
            function_call_mode
            if function_call_mode is not None
            else settings_block.get("function_call_mode"),
            default="auto",
        )

        def _coerce_allowlist(source: Any) -> List[str]:
            if source is self._unset or source is PROVIDER_UNSET:
                return settings_block.get("allowed_function_names", [])
            if source is None:
                return []
            return self._coerce_allowed_function_names(source)

        current_allowed = _coerce_allowlist(allowed_function_names)
        settings_block["allowed_function_names"] = current_allowed

        try:
            existing_cache = self._coerce_allowed_function_names(
                settings_block.get("cached_allowed_function_names", [])
            )
        except ValueError:
            existing_cache = []

        if (
            cached_allowed_function_names is self._unset
            or cached_allowed_function_names is PROVIDER_UNSET
        ):
            cache_source = current_allowed if current_allowed else existing_cache
        else:
            cache_source = cached_allowed_function_names

        normalized_cache = self._coerce_allowed_function_names(cache_source)
        settings_block["cached_allowed_function_names"] = normalized_cache

        if settings_block["function_calling"] is False:
            settings_block["function_call_mode"] = "none"
            settings_block["allowed_function_names"] = []
        elif settings_block["function_call_mode"] == "none":
            settings_block["allowed_function_names"] = []
        else:
            if (
                not current_allowed
                and normalized_cache
                and (
                    cached_allowed_function_names is self._unset
                    or cached_allowed_function_names is PROVIDER_UNSET
                )
                and allowed_function_names is None
            ):
                settings_block["allowed_function_names"] = list(normalized_cache)

        if response_schema is not None:
            settings_block["response_schema"] = resolver.resolve_response_schema(
                response_schema,
                allow_invalid_stored=False,
            )
            if (
                settings_block["response_schema"]
                and not settings_block["response_mime_type"]
            ):
                settings_block["response_mime_type"] = "application/json"
        elif not settings_block.get("response_schema"):
            settings_block["response_schema"] = {}
        elif (
            settings_block["response_schema"]
            and not settings_block.get("response_mime_type")
        ):
            settings_block["response_mime_type"] = "application/json"

        if (
            settings_block["response_schema"]
            and settings_block["response_mime_type"]
            and settings_block["response_mime_type"].lower() != "application/json"
        ):
            raise ValueError(
                "Response MIME type must be 'application/json' when a schema is provided."
            )

        if settings_block["function_call_mode"] == "none":
            settings_block["allowed_function_names"] = []

        if settings_block["allowed_function_names"] and not settings_block[
            "function_calling"
        ]:
            settings_block["cached_allowed_function_names"] = self._coerce_allowed_function_names(
                settings_block["allowed_function_names"]
            )
            settings_block["allowed_function_names"] = []

        def _normalise_base_url(value: Any, existing: Optional[str]) -> Optional[str]:
            if value is self._unset or value is PROVIDER_UNSET:
                return existing
            if value in {None, ""}:
                return None
            if isinstance(value, (bytes, bytearray)):
                try:
                    value = value.decode("utf-8")
                except Exception as exc:
                    raise ValueError("Base URL must be a valid HTTP(S) URL.") from exc
            if isinstance(value, str):
                candidate = value.strip()
                if not candidate:
                    return None
                parsed = urlparse(candidate)
                if parsed.scheme in {"http", "https"} and parsed.netloc:
                    return candidate
                raise ValueError("Base URL must include an http:// or https:// scheme.")
            raise ValueError("Base URL must be provided as text.")

        settings_block["base_url"] = _normalise_base_url(
            base_url,
            settings_block.get("base_url"),
        )

        max_tokens_value = resolver.resolve_optional_int(
            "max_output_tokens",
            max_output_tokens,
            field="Max output tokens",
            minimum=1,
        )
        settings_block["max_output_tokens"] = max_tokens_value

        self.manager.yaml_config["GOOGLE_LLM"] = copy.deepcopy(settings_block)
        self.manager.config["GOOGLE_LLM"] = copy.deepcopy(settings_block)
        self.manager._write_yaml_config()

        return copy.deepcopy(settings_block)

    @staticmethod
    def _coerce_function_call_mode(value: Any, *, default: str) -> str:
        """Validate Gemini function call mode values against supported options."""

        valid_modes = {"auto", "any", "none", "require"}

        if value in {None, ""}:
            return default

        if isinstance(value, str):
            candidate = value.strip().lower()
            if candidate in valid_modes:
                return candidate
        elif isinstance(value, Mapping):
            candidate = str(value.get("type", "")).strip().lower()
            if candidate in valid_modes:
                return candidate

        raise ValueError("Function call mode must be one of: auto, any, none, require.")

    @staticmethod
    def _coerce_allowed_function_names(value: Any) -> List[str]:
        """Normalise allowed function names as a list of distinct strings."""

        if value in (None, "", []):
            return []

        names: List[str] = []
        if isinstance(value, str):
            tokens = [part.strip() for part in value.split(",") if part and part.strip()]
            names.extend(tokens)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            for item in value:
                if item in {None, ""}:
                    continue
                if not isinstance(item, str):
                    raise ValueError("Allowed function names must be strings.")
                cleaned = item.strip()
                if cleaned:
                    names.append(cleaned)
        else:
            raise ValueError(
                "Allowed function names must be provided as a comma-separated string or sequence of strings."
            )

        seen = set()
        deduped: List[str] = []
        for name in names:
            if name not in seen:
                deduped.append(name)
                seen.add(name)
        return deduped


class MistralProviderConfig:
    """Provider section encapsulating Mistral LLM helpers."""

    def __init__(self, *, manager: "ConfigManager", registry: ProviderConfigSections) -> None:
        self.manager = manager
        self.registry = registry
        self.logger = manager.logger
        self._unset = manager.UNSET

    def get_llm_settings(self) -> Dict[str, Any]:
        """Return persisted defaults for the Mistral chat provider."""

        defaults: Dict[str, Any] = {
            "model": "mistral-large-latest",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": None,
            "safe_prompt": False,
            "stream": True,
            "random_seed": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "stop_sequences": [],
            "json_mode": False,
            "json_schema": None,
            "max_retries": 3,
            "retry_min_seconds": 4,
            "retry_max_seconds": 10,
            "base_url": self.manager.get_config("MISTRAL_BASE_URL"),
            "prompt_mode": None,
        }

        stored = self.manager.get_config("MISTRAL_LLM")
        if isinstance(stored, dict):
            merged = dict(defaults)

            model = stored.get("model")
            if isinstance(model, str) and model.strip():
                merged["model"] = model.strip()

            def _coerce_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
                if value is None:
                    return default
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return default
                if number < minimum or number > maximum:
                    return default
                return number

            def _coerce_int(value: Any, *, allow_zero: bool = False) -> Optional[int]:
                if value is None or value == "":
                    return None
                try:
                    candidate = int(value)
                except (TypeError, ValueError):
                    return None
                if candidate < 0:
                    return None
                if candidate == 0 and not allow_zero:
                    return None
                return candidate

            def _coerce_bool(value: Any, default: bool) -> bool:
                if value is None:
                    return default
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    normalized = value.strip().lower()
                    if normalized in {"1", "true", "yes", "on"}:
                        return True
                    if normalized in {"0", "false", "no", "off"}:
                        return False
                    return default
                try:
                    return bool(value)
                except Exception:
                    return default

            merged["temperature"] = _coerce_float(
                stored.get("temperature"),
                default=defaults["temperature"],
                minimum=0.0,
                maximum=2.0,
            )
            merged["top_p"] = _coerce_float(
                stored.get("top_p"),
                default=defaults["top_p"],
                minimum=0.0,
                maximum=1.0,
            )
            merged["max_tokens"] = _coerce_int(
                stored.get("max_tokens"),
                allow_zero=True,
            )
            merged["safe_prompt"] = _coerce_bool(
                stored.get("safe_prompt"),
                defaults["safe_prompt"],
            )
            merged["stream"] = _coerce_bool(
                stored.get("stream"),
                defaults["stream"],
            )
            merged["random_seed"] = _coerce_int(
                stored.get("random_seed"),
                allow_zero=True,
            )
            merged["frequency_penalty"] = _coerce_float(
                stored.get("frequency_penalty"),
                default=defaults["frequency_penalty"],
                minimum=-2.0,
                maximum=2.0,
            )
            merged["presence_penalty"] = _coerce_float(
                stored.get("presence_penalty"),
                default=defaults["presence_penalty"],
                minimum=-2.0,
                maximum=2.0,
            )
            merged["tool_choice"] = stored.get("tool_choice", defaults["tool_choice"])
            merged["parallel_tool_calls"] = _coerce_bool(
                stored.get("parallel_tool_calls"),
                defaults["parallel_tool_calls"],
            )
            merged["stop_sequences"] = self.manager._coerce_stop_sequences(
                stored.get("stop_sequences", defaults["stop_sequences"])
            )
            merged["json_mode"] = _coerce_bool(
                stored.get("json_mode"),
                defaults["json_mode"],
            )
            merged["json_schema"] = stored.get("json_schema", defaults["json_schema"])
            merged["max_retries"] = _coerce_int(
                stored.get("max_retries"),
                allow_zero=False,
            ) or defaults["max_retries"]
            merged["retry_min_seconds"] = _coerce_int(
                stored.get("retry_min_seconds"),
                allow_zero=False,
            ) or defaults["retry_min_seconds"]
            merged["retry_max_seconds"] = _coerce_int(
                stored.get("retry_max_seconds"),
                allow_zero=False,
            ) or defaults["retry_max_seconds"]
            merged["base_url"] = stored.get("base_url", defaults["base_url"])
            merged["prompt_mode"] = stored.get("prompt_mode", defaults["prompt_mode"])

            defaults = merged

        return defaults

    def set_llm_settings(
        self,
        *,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        safe_prompt: Optional[bool] = None,
        stream: Optional[bool] = None,
        random_seed: Optional[Any] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
        stop_sequences: Any = PROVIDER_UNSET,
        json_mode: Optional[Any] = None,
        json_schema: Optional[Any] = None,
        max_retries: Optional[int] = None,
        retry_min_seconds: Optional[int] = None,
        retry_max_seconds: Optional[int] = None,
        base_url: Any = PROVIDER_UNSET,
        prompt_mode: Any = PROVIDER_UNSET,
    ) -> Dict[str, Any]:
        """Persist default configuration for the Mistral chat provider."""

        if not isinstance(model, str) or not model.strip():
            raise ValueError("A default Mistral model must be provided.")

        current = self.get_llm_settings()
        settings = dict(current)
        settings["model"] = model.strip()

        def _normalize_float(
            value: Optional[Any], *, field: str, default: float, minimum: float, maximum: float
        ) -> float:
            if value is None:
                return default
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be a number.") from exc
            if numeric < minimum or numeric > maximum:
                raise ValueError(
                    f"{field} must be between {minimum} and {maximum}."
                )
            return numeric

        def _normalize_positive_int(
            value: Optional[Any],
            field: str,
            *,
            allow_zero: bool = False,
            zero_means_none: bool = False,
        ) -> Optional[int]:
            if value is None or value == "":
                return None
            try:
                numeric = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be an integer.") from exc
            if numeric < 0:
                raise ValueError(f"{field} must be a non-negative integer.")
            if numeric == 0:
                if zero_means_none:
                    return None
                if not allow_zero:
                    raise ValueError(f"{field} must be a positive integer.")
            return numeric

        def _normalize_bool(value: Optional[Any]) -> Optional[bool]:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    return True
                if normalized in {"0", "false", "no", "off"}:
                    return False
                raise ValueError("Boolean fields must be provided as a boolean or yes/no string.")
            return bool(value)

        def _normalize_tool_choice(value: Optional[Any]) -> Any:
            if value is None:
                return settings.get("tool_choice")
            if isinstance(value, Mapping):
                return dict(value)
            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    return None
                return cleaned
            raise ValueError(
                "Tool choice must be a mapping, string, or None."
            )

        def _normalize_json_mode(value: Any, existing: bool) -> bool:
            if value is None:
                return existing
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if not normalized:
                    return existing
                if normalized in {"1", "true", "yes", "on", "json", "json_object"}:
                    return True
                if normalized in {"0", "false", "no", "off", "text", "none"}:
                    return False
                return existing
            try:
                return bool(value)
            except Exception:
                return existing

        def _normalize_json_schema(
            value: Any, existing: Optional[Dict[str, Any]]
        ) -> Optional[Dict[str, Any]]:
            if value is None:
                return existing

            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8")

            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return None
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON schema must be valid JSON: {exc.msg}"
                    ) from exc

            if value is False:
                return None

            if not isinstance(value, dict):
                raise ValueError(
                    "JSON schema must be provided as an object or JSON string."
                )

            if not value:
                return None

            schema_payload = value.get("schema") if isinstance(value, dict) else None
            schema_name = value.get("name") if isinstance(value, dict) else None

            if schema_payload is None:
                schema_payload = value
                schema_like_keys = {
                    "$schema",
                    "$ref",
                    "type",
                    "properties",
                    "items",
                    "oneOf",
                    "anyOf",
                    "allOf",
                    "definitions",
                    "patternProperties",
                }
                if isinstance(schema_payload, dict) and not (
                    schema_like_keys & set(schema_payload.keys())
                ):
                    raise ValueError(
                        "JSON schema must include a 'schema' object or a valid schema definition."
                    )

            if not isinstance(schema_payload, dict):
                raise ValueError(
                    "The 'schema' entry for the JSON schema must be an object."
                )

            if schema_name is None:
                if isinstance(existing, dict):
                    schema_name = existing.get("name")
                if not schema_name:
                    schema_name = "atlas_response"

            normalized: Dict[str, Any] = {
                "name": str(schema_name).strip() or "atlas_response",
                "schema": schema_payload,
            }

            strict_value = value.get("strict") if isinstance(value, dict) else None
            if strict_value is None and isinstance(existing, dict):
                strict_value = existing.get("strict")

            if strict_value is not None:
                normalized["strict"] = bool(strict_value)

            try:
                return json.loads(json.dumps(normalized))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"JSON schema contains non-serializable content: {exc}"
                ) from exc

        settings["temperature"] = _normalize_float(
            temperature,
            field="Temperature",
            default=float(settings.get("temperature", 0.0)),
            minimum=0.0,
            maximum=2.0,
        )
        settings["top_p"] = _normalize_float(
            top_p,
            field="Top-p",
            default=float(settings.get("top_p", 1.0)),
            minimum=0.0,
            maximum=1.0,
        )
        tokens = _normalize_positive_int(
            max_tokens,
            "Max tokens",
            zero_means_none=True,
        )
        settings["max_tokens"] = tokens

        normalized_safe_prompt = _normalize_bool(safe_prompt)
        if normalized_safe_prompt is not None:
            settings["safe_prompt"] = normalized_safe_prompt

        normalized_stream = _normalize_bool(stream)
        if normalized_stream is not None:
            settings["stream"] = normalized_stream

        normalized_parallel = _normalize_bool(parallel_tool_calls)
        if normalized_parallel is not None:
            settings["parallel_tool_calls"] = normalized_parallel

        seed = (
            _normalize_positive_int(
                random_seed,
                "Random seed",
                allow_zero=True,
            )
            if random_seed not in {None, ""}
            else None
        )
        settings["random_seed"] = seed

        settings["frequency_penalty"] = _normalize_float(
            frequency_penalty,
            field="Frequency penalty",
            default=float(settings.get("frequency_penalty", 0.0)),
            minimum=-2.0,
            maximum=2.0,
        )
        settings["presence_penalty"] = _normalize_float(
            presence_penalty,
            field="Presence penalty",
            default=float(settings.get("presence_penalty", 0.0)),
            minimum=-2.0,
            maximum=2.0,
        )

        settings["tool_choice"] = _normalize_tool_choice(tool_choice)

        if (
            stop_sequences is not self._unset
            and stop_sequences is not PROVIDER_UNSET
        ):
            settings["stop_sequences"] = self.manager._coerce_stop_sequences(
                stop_sequences
            )

        settings["json_mode"] = _normalize_json_mode(
            json_mode,
            bool(settings.get("json_mode", False)),
        )
        settings["json_schema"] = _normalize_json_schema(
            json_schema,
            settings.get("json_schema"),
        )

        def _normalize_prompt_mode(value: Any, existing: Optional[str]) -> Optional[str]:
            if value is self._unset or value is PROVIDER_UNSET:
                return existing
            if value in {None, ""}:
                return None
            if isinstance(value, (bytes, bytearray)):
                try:
                    value = value.decode("utf-8")
                except Exception as exc:
                    raise ValueError("Prompt mode must be text.") from exc
            if isinstance(value, str):
                normalized = value.strip().lower()
                if not normalized:
                    return None
                if normalized not in {"reasoning"}:
                    raise ValueError(
                        "Prompt mode must be one of: reasoning or left unset."
                    )
                return normalized
            raise ValueError("Prompt mode must be provided as text or None.")

        settings["prompt_mode"] = _normalize_prompt_mode(
            prompt_mode,
            settings.get("prompt_mode"),
        )

        def _normalize_base_url(value: Any, existing: Optional[str]) -> Optional[str]:
            if value is self._unset or value is PROVIDER_UNSET:
                return existing
            if value in {None, ""}:
                return None
            if isinstance(value, (bytes, bytearray)):
                try:
                    value = value.decode("utf-8")
                except Exception as exc:
                    raise ValueError("Base URL must be a valid HTTP(S) URL.") from exc
            if isinstance(value, str):
                candidate = value.strip()
                if not candidate:
                    return None
                parsed = urlparse(candidate)
                if parsed.scheme in {"http", "https"} and parsed.netloc:
                    return candidate
                raise ValueError("Base URL must include an http:// or https:// scheme.")
            raise ValueError("Base URL must be provided as text.")

        settings["base_url"] = _normalize_base_url(base_url, settings.get("base_url"))

        if max_retries is not None:
            normalized_retries = _normalize_positive_int(
                max_retries,
                "Max retries",
            )
            if normalized_retries is None:
                raise ValueError("Max retries must be a positive integer.")
            settings["max_retries"] = normalized_retries

        current_retry_min = settings.get("retry_min_seconds", 4)
        if not isinstance(current_retry_min, (int, float)) or current_retry_min <= 0:
            current_retry_min = 4

        if retry_min_seconds is not None:
            normalized_retry_min = _normalize_positive_int(
                retry_min_seconds,
                "Retry minimum wait",
            )
            if normalized_retry_min is None:
                raise ValueError("Retry minimum wait must be a positive integer.")
            current_retry_min = normalized_retry_min
            settings["retry_min_seconds"] = current_retry_min
        else:
            settings["retry_min_seconds"] = int(current_retry_min)

        current_retry_max = settings.get("retry_max_seconds", max(current_retry_min, 10))
        if not isinstance(current_retry_max, (int, float)) or current_retry_max <= 0:
            current_retry_max = max(current_retry_min, 10)

        if retry_max_seconds is not None:
            normalized_retry_max = _normalize_positive_int(
                retry_max_seconds,
                "Retry maximum wait",
            )
            if normalized_retry_max is None:
                raise ValueError("Retry maximum wait must be a positive integer.")
            current_retry_max = normalized_retry_max

        if current_retry_max < current_retry_min:
            raise ValueError("Retry maximum wait must be greater than or equal to the minimum wait.")

        settings["retry_max_seconds"] = int(current_retry_max)

        self.manager.yaml_config["MISTRAL_LLM"] = copy.deepcopy(settings)
        self.manager.config["MISTRAL_LLM"] = copy.deepcopy(settings)
        self.manager.config["MISTRAL_BASE_URL"] = settings.get("base_url")
        self.manager.env_config["MISTRAL_BASE_URL"] = settings.get("base_url")

        self.manager._write_yaml_config()

        return copy.deepcopy(settings)

