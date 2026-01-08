"""
Common types and utilities for ATLAS services.

Provides the foundation types, exceptions, and protocols that all
services should use for consistency and interoperability.

Author: ATLAS Team  
Date: Jan 7, 2026
"""

# Core types
from .types import Actor, DomainEvent, OperationResult

# Exception hierarchy
from .exceptions import (
    BusinessRuleError,
    ConfigurationError,
    ConflictError,
    ExternalServiceError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceError,
    ValidationError,
)

# Permission system
from .permissions import (
    InMemoryPermissionProvider,
    PermissionChecker,
    PermissionProvider,
)

# Protocol interfaces
from .protocols import (
    AuditLogger,
    BackgroundTaskService,
    CacheableService,
    EventPublisher,
    EventSubscriber,
    MetricsService,
    Repository,
    SearchableRepository,
    Service,
    TenantService,
)

# Messaging integration
from .messaging import (
    create_domain_event_publisher,
    DomainEventPublisher,
    DomainEventSubscriber,
    get_event_channel_mappings,
    register_domain_event_channel,
)

__all__ = [
    # Core types
    "Actor",
    "DomainEvent", 
    "OperationResult",
    
    # Exceptions
    "ServiceError",
    "ValidationError",
    "NotFoundError",
    "ConflictError",
    "PermissionDeniedError",
    "ConfigurationError",
    "ExternalServiceError",
    "RateLimitError",
    "BusinessRuleError",
    
    # Permission system
    "PermissionChecker",
    "PermissionProvider",
    "InMemoryPermissionProvider",
    
    # Protocols
    "Service",
    "Repository",
    "SearchableRepository",
    "EventPublisher",
    "EventSubscriber",
    "AuditLogger",
    "TenantService",
    "CacheableService",
    "BackgroundTaskService",
    "MetricsService",
    
    # Messaging
    "DomainEventPublisher",
    "DomainEventSubscriber", 
    "create_domain_event_publisher",
    "get_event_channel_mappings",
    "register_domain_event_channel",
]