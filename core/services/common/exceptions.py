"""
Standard exception hierarchy for ATLAS services.

Provides a clean separation between service-layer errors and
implementation details (e.g., database errors, HTTP errors).
All service methods should raise these exceptions rather than
leaking infrastructure-specific errors.

Author: ATLAS Team
Date: Jan 7, 2026
"""

from typing import Any, Dict, Optional


class ServiceError(Exception):
    """
    Base exception for all service-layer errors.
    
    Should be subclassed for specific error types.
    Provides a clean interface that doesn't leak implementation details.
    
    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error code
        details: Additional error context (for debugging)
        
    Example:
        raise ServiceError("Database connection failed", "DB_CONNECTION_ERROR")
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}
        
    def __str__(self) -> str:
        if self.error_code:
            return f"{self.error_code}: {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ValidationError(ServiceError):
    """
    Raised when input validation fails.
    
    Used for invalid parameters, malformed data, business rule violations, etc.
    Should include specific details about what validation failed.
    
    Example:
        raise ValidationError(
            "Email format is invalid",
            "INVALID_EMAIL", 
            {"field": "email", "value": "not-an-email"}
        )
    """
    pass


class NotFoundError(ServiceError):
    """
    Raised when a requested entity cannot be found.
    
    Should be used instead of returning None or empty results
    when the absence of an entity is an error condition.
    
    Example:
        raise NotFoundError(
            "User not found",
            "USER_NOT_FOUND",
            {"user_id": user_id}
        )
    """
    pass


class ConflictError(ServiceError):
    """
    Raised when an operation conflicts with current state.
    
    Common scenarios:
    - Trying to create entity that already exists
    - Concurrent modification conflicts
    - Business rule conflicts
    
    Example:
        raise ConflictError(
            "Username already taken",
            "USERNAME_CONFLICT",
            {"username": username}
        )
    """
    pass


class PermissionDeniedError(ServiceError):
    """
    Raised when an actor lacks required permissions.
    
    Should include information about what permission was required
    and who was attempting the operation (without leaking sensitive data).
    
    Example:
        raise PermissionDeniedError(
            "Insufficient permissions to delete conversation",
            "INSUFFICIENT_PERMISSIONS",
            {"required_permission": "conversations:delete", "actor_type": "user"}
        )
    """
    pass


class ConfigurationError(ServiceError):
    """
    Raised when service configuration is invalid or missing.
    
    Should be used during service initialization or when
    configuration-dependent operations fail.
    
    Example:
        raise ConfigurationError(
            "Database connection string not configured",
            "MISSING_DB_CONFIG"
        )
    """
    pass


class ExternalServiceError(ServiceError):
    """
    Raised when an external service dependency fails.
    
    Should be used to wrap errors from external APIs, databases,
    message queues, etc. while hiding implementation details.
    
    Example:
        raise ExternalServiceError(
            "Failed to connect to OpenAI API",
            "EXTERNAL_API_ERROR",
            {"service": "openai", "status_code": 503}
        )
    """
    pass


class RateLimitError(ServiceError):
    """
    Raised when rate limiting is triggered.
    
    Should include information about when the operation can be retried.
    
    Example:
        raise RateLimitError(
            "API rate limit exceeded",
            "RATE_LIMIT_EXCEEDED",
            {"retry_after": 60}
        )
    """
    pass


class BusinessRuleError(ServiceError):
    """
    Raised when a business rule is violated.
    
    Distinct from ValidationError in that the data may be well-formed
    but violates domain-specific business logic.
    
    Example:
        raise BusinessRuleError(
            "Cannot delete conversation with active agents",
            "ACTIVE_AGENTS_PREVENT_DELETE",
            {"conversation_id": conv_id, "active_agents": 2}
        )
    """
    pass