"""
Provider configuration service.

Manages provider configurations, credentials, and state.
Acts as the central source of truth for provider settings.

Author: ATLAS Team
Date: Jan 11, 2026
"""

import logging
from typing import Any, Dict, List, Optional

from core.services.common import Actor
from core.services.providers.types import (
    ProviderConfig,
    ProviderConfigEvent,
    ProviderStateEvent,
    ProviderType,
)
from core.services.providers.permissions import ProviderPermissionChecker


logger = logging.getLogger(__name__)


class ProviderConfigService:
    """
    Service for managing provider configurations.
    """

    def __init__(
        self, 
        config_manager: Any, 
        message_bus: Any,
        permission_checker: Optional[ProviderPermissionChecker] = None
    ) -> None:
        self._config = config_manager
        self._bus = message_bus
        self._permissions = permission_checker or ProviderPermissionChecker()

    def list_providers(self, actor: Actor) -> List[ProviderConfig]:
        """
        List all available providers.
        """
        self._permissions.check_read_permission(actor)
        
        # Currently, ConfigManager is the source of truth.
        # We need to adapt ConfigManager's internal format to ProviderConfig objects.
        # This is a bit tricky without knowing exactly what ConfigManager returns, 
        # but we iterate known providers.
        
        providers = []
        # Hardcoded list from ProviderManager.AVAILABLE_PROVIDERS for now
        # ideally this comes from dynamic discovery or config
        known_providers = ["OpenAI", "Mistral", "Google", "HuggingFace", "Anthropic", "Grok"]
        
        for name in known_providers:
            provider_id = name.lower()
            # Fetch config from ConfigManager
            # e.g., self._config.get_provider_settings(name)
            
            # Mocking the adaptation for the skeleton
            providers.append(ProviderConfig(
                provider_id=provider_id,
                name=name,
                provider_type=ProviderType.LLM,
                enabled=True, # Need to check actual config
                description=f"{name} Provider"
            ))
            
        return providers

    def get_provider(self, actor: Actor, provider_id: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider."""
        self._permissions.check_read_permission(actor)
        
        # Mock implementation
        # Real impl would fetch from config_manager
        return ProviderConfig(
            provider_id=provider_id,
            name=provider_id.capitalize(),
            provider_type=ProviderType.LLM,
            enabled=True
        )

    def configure_provider(
        self, 
        actor: Actor, 
        provider_id: str, 
        config_updates: Dict[str, Any]
    ) -> ProviderConfig:
        """Update provider configuration."""
        self._permissions.check_write_permission(actor)
        
        # Update config via config_manager
        self._config.update_provider(provider_id, config_updates)
        
        # Emit event
        event = ProviderConfigEvent(
            provider_id=provider_id,
            tenant_id=actor.tenant_id,
            actor_id=actor.id,
            changed_fields=list(config_updates.keys())
        )
        self._bus.publish(event)
        
        return self.get_provider(actor, provider_id) # type: ignore

    def enable_provider(self, actor: Actor, provider_id: str) -> None:
        """Enable a provider."""
        self._permissions.check_write_permission(actor)
        
        # self._config.set_provider_enabled(provider_id, True)
        
        event = ProviderStateEvent(
            provider_id=provider_id,
            tenant_id=actor.tenant_id,
            actor_id=actor.id,
            enabled=True,
            event_type="provider.enabled"
        )
        self._bus.publish(event)

    def disable_provider(self, actor: Actor, provider_id: str) -> None:
        """Disable a provider."""
        self._permissions.check_write_permission(actor)
        
        # self._config.set_provider_enabled(provider_id, False)
        
        event = ProviderStateEvent(
            provider_id=provider_id,
            tenant_id=actor.tenant_id,
            actor_id=actor.id,
            enabled=False,
            event_type="provider.disabled"
        )
        self._bus.publish(event)

    def set_credentials(self, actor: Actor, provider_id: str, credentials: Dict[str, str]) -> None:
        """Set credentials for a provider."""
        self._permissions.check_credential_access(actor)
        
        # self._config.set_provider_credentials(provider_id, credentials)
        
        # Intentionally do not emit credentials in event
        event = ProviderConfigEvent(
            provider_id=provider_id,
            tenant_id=actor.tenant_id,
            actor_id=actor.id,
            changed_fields=["credentials"]
        )
        self._bus.publish(event)
