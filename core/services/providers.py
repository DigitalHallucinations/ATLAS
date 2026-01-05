"""Provider-facing service facade used by the ATLAS application."""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from modules.background_tasks import run_async_in_thread


class ProviderService:
    """Centralise provider related helpers for the ATLAS facade."""

    def __init__(
        self,
        *,
        provider_manager,
        config_manager,
        logger,
        chat_session,
        speech_manager=None,
    ) -> None:
        self._provider_manager = provider_manager
        self._config_manager = config_manager
        self._logger = logger
        self._chat_session = chat_session
        self._speech_manager = speech_manager
        self._provider_change_listeners: List[Callable[[Dict[str, str]], None]] = []

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _require_provider_manager(self):
        if self._provider_manager is None:
            raise RuntimeError("Provider manager is not initialized.")
        return self._provider_manager

    def _notify_provider_change_listeners(self) -> None:
        summary = self.get_chat_status_summary()
        for listener in list(self._provider_change_listeners):
            try:
                listener(summary)
            except Exception as exc:  # pragma: no cover - log defensive guard
                self._logger.error(
                    "Provider change listener %s failed: %s",
                    listener,
                    exc,
                    exc_info=True,
                )

    def notify_provider_change_listeners(self) -> None:
        """Expose notification dispatching for facade consumers."""

        self._notify_provider_change_listeners()

    # ------------------------------------------------------------------
    # Background task helpers
    # ------------------------------------------------------------------
    def run_in_background(
        self,
        coroutine_factory: Callable[[], Awaitable[Any]],
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        thread_name: Optional[str] = None,
    ) -> Future:
        return run_async_in_thread(
            coroutine_factory,
            on_success=on_success,
            on_error=on_error,
            logger=self._logger,
            thread_name=thread_name,
        )

    def run_provider_manager_task(
        self,
        coroutine_factory: Callable[[], Awaitable[Any]],
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        thread_name: Optional[str] = None,
    ) -> Future:
        return self.run_in_background(
            coroutine_factory,
            on_success=on_success,
            on_error=on_error,
            thread_name=thread_name,
        )

    def set_current_provider_in_background(
        self,
        provider: str,
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> Future:
        thread_name = f"set-provider-{provider}" if provider else None
        return self.run_in_background(
            lambda: self.set_current_provider(provider),
            on_success=on_success,
            on_error=on_error,
            thread_name=thread_name,
        )

    def update_provider_api_key_in_background(
        self,
        provider_name: str,
        new_api_key: Optional[str],
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> Future:
        thread_name = f"update-api-key-{provider_name}" if provider_name else None
        return self.run_in_background(
            lambda: self.update_provider_api_key(provider_name, new_api_key),
            on_success=on_success,
            on_error=on_error,
            thread_name=thread_name,
        )

    # ------------------------------------------------------------------
    # Provider selection helpers
    # ------------------------------------------------------------------
    async def refresh_current_provider(
        self, provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        manager = self._require_provider_manager()
        active_provider = manager.get_current_provider()
        target_provider = provider_name or active_provider

        if not target_provider:
            return {"success": False, "error": "No active provider is configured."}

        if target_provider != active_provider:
            return {
                "success": False,
                "error": f"Provider '{target_provider}' is not the active provider.",
                "active_provider": active_provider,
            }

        await manager.set_current_provider(active_provider)
        return {
            "success": True,
            "message": f"Provider {active_provider} refreshed.",
            "provider": active_provider,
        }

    async def set_current_provider(self, provider: str) -> None:
        manager = self._require_provider_manager()
        try:
            await manager.set_current_provider(provider)
        except Exception as exc:
            self._logger.error(
                "Failed to set provider %s: %s", provider, exc, exc_info=True
            )
            raise

        if self._chat_session is not None:
            try:
                self._chat_session.set_provider(provider)
            except Exception as exc:  # pragma: no cover - UI guard
                self._logger.error(
                    "Failed updating chat session provider: %s", exc, exc_info=True
                )

        current_model = manager.get_current_model()
        if self._chat_session is not None:
            try:
                self._chat_session.set_model(current_model)
            except Exception as exc:  # pragma: no cover - UI guard
                self._logger.error(
                    "Failed updating chat session model: %s", exc, exc_info=True
                )

        self._logger.debug(
            "Current provider set to %s with model %s", provider, current_model
        )
        self._notify_provider_change_listeners()

    # ------------------------------------------------------------------
    # Provider metadata accessors
    # ------------------------------------------------------------------
    def get_available_providers(self) -> List[str]:
        manager = self._require_provider_manager()
        providers = manager.get_available_providers()
        return list(providers) if isinstance(providers, Iterable) else providers

    async def test_huggingface_token(
        self, token: Optional[str] = None
    ) -> Dict[str, Any]:
        return await self._require_provider_manager().test_huggingface_token(token)

    def list_hf_models(self) -> Dict[str, Any]:
        return self._require_provider_manager().list_hf_models()

    async def load_hf_model(
        self, model_name: str, force_download: bool = False
    ) -> Dict[str, Any]:
        return await self._require_provider_manager().load_hf_model(
            model_name, force_download=force_download
        )

    async def unload_hf_model(self) -> Dict[str, Any]:
        return await self._require_provider_manager().unload_hf_model()

    async def remove_hf_model(self, model_name: str) -> Dict[str, Any]:
        return await self._require_provider_manager().remove_hf_model(model_name)

    async def download_hf_model(
        self, model_id: str, force: bool = False
    ) -> Dict[str, Any]:
        return await self._require_provider_manager().download_huggingface_model(
            model_id, force=force
        )

    async def search_hf_models(
        self,
        search_query: str,
        filters: Optional[Dict[str, Any]] = None,
        *,
        limit: int = 10,
    ) -> Dict[str, Any]:
        return await self._require_provider_manager().search_huggingface_models(
            search_query, filters, limit=limit
        )

    def update_hf_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        return self._require_provider_manager().update_huggingface_settings(settings)

    def clear_hf_cache(self) -> Dict[str, Any]:
        return self._require_provider_manager().clear_huggingface_cache()

    def save_hf_token(self, token: Optional[str]) -> Dict[str, Any]:
        return self._require_provider_manager().save_huggingface_token(token)

    def get_provider_api_key_status(self, provider_name: str) -> Dict[str, Any]:
        return self._require_provider_manager().get_provider_api_key_status(provider_name)

    async def update_provider_api_key(
        self, provider_name: str, new_api_key: Optional[str]
    ) -> Dict[str, Any]:
        return await self._require_provider_manager().update_provider_api_key(
            provider_name, new_api_key
        )

    def ensure_huggingface_ready(self) -> Dict[str, Any]:
        return self._require_provider_manager().ensure_huggingface_ready()

    def get_default_provider(self) -> Optional[str]:
        manager = self._require_provider_manager()
        return manager.get_current_provider()

    def get_default_model(self) -> Optional[str]:
        manager = self._require_provider_manager()
        return manager.get_current_model()

    def get_openai_llm_settings(self) -> Dict[str, Any]:
        settings = self._require_provider_manager().get_openai_llm_settings()
        return dict(settings)

    def get_google_llm_settings(self) -> Dict[str, Any]:
        settings = self._require_provider_manager().get_google_llm_settings()
        return dict(settings)

    def get_anthropic_settings(self) -> Dict[str, Any]:
        settings = self._require_provider_manager().get_anthropic_settings()
        return dict(settings)

    async def list_openai_models(
        self,
        *,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._require_provider_manager().list_openai_models(
            base_url=base_url, organization=organization
        )

    async def list_anthropic_models(
        self,
        *,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._require_provider_manager().list_anthropic_models(
            base_url=base_url
        )

    async def list_google_models(
        self,
        *,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._require_provider_manager().list_google_models(
            base_url=base_url
        )

    def set_openai_llm_settings(self, **kwargs: Any) -> Dict[str, Any]:
        manager = self._require_provider_manager()
        setter = getattr(manager, "set_openai_llm_settings", None)
        if not callable(setter):
            raise AttributeError(
                "Provider manager does not support OpenAI settings updates."
            )
        return setter(**kwargs)

    def set_google_llm_settings(self, **kwargs: Any) -> Dict[str, Any]:
        manager = self._require_provider_manager()
        setter = getattr(manager, "set_google_llm_settings", None)
        if not callable(setter):
            raise AttributeError(
                "Provider manager does not support Google settings updates."
            )
        return setter(**kwargs)

    def set_anthropic_settings(self, **kwargs: Any) -> Dict[str, Any]:
        manager = self._require_provider_manager()
        setter = getattr(manager, "set_anthropic_settings", None)
        if not callable(setter):
            raise AttributeError(
                "Provider manager does not support Anthropic settings updates."
            )
        return setter(**kwargs)

    def get_models_for_provider(self, provider: str) -> List[str]:
        manager = self._require_provider_manager()
        getter = getattr(manager, "get_models_for_provider", None)
        if not callable(getter):
            return []
        return getter(provider)

    # ------------------------------------------------------------------
    # Listener utilities
    # ------------------------------------------------------------------
    def add_provider_change_listener(
        self, listener: Callable[[Dict[str, str]], None]
    ) -> None:
        if not callable(listener):
            raise TypeError("listener must be callable")
        if listener in self._provider_change_listeners:
            return
        self._provider_change_listeners.append(listener)

    def remove_provider_change_listener(
        self, listener: Callable[[Dict[str, str]], None]
    ) -> None:
        if listener in self._provider_change_listeners:
            self._provider_change_listeners.remove(listener)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def get_chat_status_summary(self) -> Dict[str, str]:
        summary: Dict[str, str] = {
            "llm_provider": "Unknown",
            "llm_model": "No model selected",
            "tts_provider": "None",
            "tts_voice": "Not Set",
        }

        provider_manager = self._provider_manager
        if provider_manager is not None:
            try:
                provider_name = provider_manager.get_current_provider()
                if provider_name:
                    summary["llm_provider"] = provider_name
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.error(
                    "Failed to read current LLM provider: %s", exc, exc_info=True
                )

            try:
                model_name = provider_manager.get_current_model()
                if model_name:
                    summary["llm_model"] = model_name
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.error(
                    "Failed to read current LLM model: %s", exc, exc_info=True
                )

        if self._speech_manager is not None:
            try:
                tts_provider, tts_voice = self._speech_manager.get_active_tts_summary()
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.error(
                    "Failed to read active TTS configuration: %s", exc, exc_info=True
                )
            else:
                summary["tts_provider"] = tts_provider or summary["tts_provider"]
                summary["tts_voice"] = tts_voice or summary["tts_voice"]

        if self._config_manager is not None:
            try:
                warnings = self._config_manager.get_pending_provider_warnings()
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.error(
                    "Failed to read pending provider warnings: %s",
                    exc,
                    exc_info=True,
                )
            else:
                provider_warning = warnings.get(summary.get("llm_provider"))
                if not provider_warning:
                    default_provider = self._config_manager.get_default_provider()
                    provider_warning = warnings.get(default_provider)
                    if provider_warning and (
                        not summary.get("llm_provider")
                        or summary.get("llm_provider") == "Unknown"
                    ):
                        summary["llm_provider"] = f"{default_provider} (Not Configured)"

                if provider_warning:
                    summary["llm_warning"] = provider_warning
                    if summary.get("llm_model") in (None, "No model selected"):
                        summary["llm_model"] = "Unavailable"

        return summary

    def format_chat_status(
        self, status_summary: Optional[Dict[str, str]] = None
    ) -> str:
        if status_summary is None:
            try:
                summary = self.get_chat_status_summary()
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.error(
                    "Failed to obtain chat status summary: %s", exc, exc_info=True
                )
                summary = {}
        else:
            summary = status_summary

        llm_provider = summary.get("llm_provider") or "Unknown"
        llm_model = summary.get("llm_model") or "No model selected"
        tts_provider = summary.get("tts_provider") or "None"
        tts_voice = summary.get("tts_voice") or "Not Set"
        status_text = (
            f"LLM: {llm_provider} • Model: {llm_model} • "
            f"TTS: {tts_provider} (Voice: {tts_voice})"
        )

        llm_warning = summary.get("llm_warning")
        if llm_warning:
            status_text = f"{status_text} • Warning: {llm_warning}"

        return status_text

