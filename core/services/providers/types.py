"""
Provider service types and domain events.

Defines DTOs for service operations and domain events for
integration with the ATLAS messaging system.

Author: ATLAS Team
Date: Jan 11, 2026
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def _generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


# =============================================================================
# Enums
# =============================================================================


class ProviderStatus(str, Enum):
    """Execution status of a provider."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    UNKNOWN = "unknown"


class ProviderType(str, Enum):
    """Type of provider service."""
    LLM = "llm"
    SPEECH = "speech"
    IMAGE = "image"
    EMBEDDING = "embedding"
    OTHER = "other"


# =============================================================================
# Data Transfer Objects
# =============================================================================


@dataclass
class ProviderConfig:
    """Configuration for a service provider."""
    provider_id: str
    name: str # Display name
    provider_type: ProviderType
    enabled: bool = False
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    default_model: Optional[str] = None
    models: List[str] = field(default_factory=list)
    description: Optional[str] = None
    # Credentials are typically stored separately or masked
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "name": self.name,
            "provider_type": self.provider_type.value,
            "enabled": self.enabled,
            "base_url": self.base_url,
            "api_version": self.api_version,
            "default_model": self.default_model,
            "models": self.models,
            "description": self.description,
        }


@dataclass
class ProviderHealth:
    """Health status of a provider."""
    provider_id: str
    status: ProviderStatus
    last_check: datetime
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "error_message": self.error_message,
            "latency_ms": self.latency_ms,
        }


# =============================================================================
# Domain Events
# =============================================================================


@dataclass(frozen=True)
class ProviderConfigEvent:
    """Emitted when a provider is configured/updated."""
    provider_id: str
    tenant_id: str
    actor_id: str
    changed_fields: List[str]
    actor_type: str = "user"
    event_type: str = "provider.configured"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "provider_id": self.provider_id,
            "changed_fields": self.changed_fields,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ProviderStateEvent:
    """Emitted when a provider is enabled or disabled."""
    provider_id: str
    tenant_id: str
    actor_id: str
    enabled: bool
    actor_type: str = "user"
    event_type: str = "provider.state_changed" # Will be overridden by 'enabled'/'disabled'
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)

    def __post_init__(self):
        # Allow event_type to be set dynamically if not passed explicitly?
        # Actually frozen=True makes it hard to change after init.
        # We'll rely on the specific event_type string being passed or inferred
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "provider_id": self.provider_id,
            "enabled": self.enabled,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ProviderHealthEvent:
    """Emitted when a provider's health status changes."""
    provider_id: str
    old_status: ProviderStatus
    new_status: ProviderStatus
    latency_ms: Optional[float] = None
    event_type: str = "provider.health_changed"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "provider_id": self.provider_id,
            "old_status": self.old_status.value,
            "new_status": self.new_status.value,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ProviderErrorEvent:
    """Emitted when a provider encounters a functional error."""
    provider_id: str
    error_code: str
    error_message: str
    context: Optional[Dict[str, Any]] = None
    event_type: str = "provider.error"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "provider_id": self.provider_id,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }
