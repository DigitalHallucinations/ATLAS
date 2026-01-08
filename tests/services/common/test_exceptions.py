"""
Tests for service exceptions.

Tests the standard service exception hierarchy to ensure
proper error handling and serialization.

Author: ATLAS Team  
Date: Jan 7, 2026
"""

import pytest

from core.services.common.exceptions import (
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


class TestServiceError:
    """Test base ServiceError class."""
    
    def test_basic_service_error(self):
        """Test basic service error creation."""
        error = ServiceError("Something went wrong")
        
        assert str(error) == "SERVICEERROR: Something went wrong"
        assert error.message == "Something went wrong"
        assert error.error_code == "SERVICEERROR"
        assert error.details == {}
    
    def test_service_error_with_code(self):
        """Test service error with custom error code."""
        error = ServiceError(
            "Database connection failed", 
            "DB_CONNECTION_ERROR"
        )
        
        assert str(error) == "DB_CONNECTION_ERROR: Database connection failed"
        assert error.error_code == "DB_CONNECTION_ERROR"
    
    def test_service_error_with_details(self):
        """Test service error with additional details."""
        details = {"host": "localhost", "port": 5432}
        error = ServiceError(
            "Connection failed",
            "CONNECTION_ERROR", 
            details
        )
        
        assert error.details == details
    
    def test_service_error_to_dict(self):
        """Test serializing service error to dictionary."""
        error = ServiceError(
            "Test error",
            "TEST_ERROR",
            {"key": "value"}
        )
        
        data = error.to_dict()
        
        expected = {
            "error_type": "ServiceError",
            "message": "Test error",
            "error_code": "TEST_ERROR",
            "details": {"key": "value"}
        }
        
        assert data == expected


class TestValidationError:
    """Test ValidationError exception."""
    
    def test_validation_error(self):
        """Test validation error creation."""
        error = ValidationError(
            "Email format is invalid",
            "INVALID_EMAIL",
            {"field": "email", "value": "not-an-email"}
        )
        
        assert isinstance(error, ServiceError)
        assert error.message == "Email format is invalid"
        assert error.error_code == "INVALID_EMAIL"
        assert error.details["field"] == "email"


class TestNotFoundError:
    """Test NotFoundError exception."""
    
    def test_not_found_error(self):
        """Test not found error creation."""
        error = NotFoundError(
            "User not found",
            "USER_NOT_FOUND",
            {"user_id": "123"}
        )
        
        assert isinstance(error, ServiceError)
        assert error.message == "User not found"
        assert error.error_code == "USER_NOT_FOUND"


class TestConflictError:
    """Test ConflictError exception."""
    
    def test_conflict_error(self):
        """Test conflict error creation."""
        error = ConflictError(
            "Username already taken",
            "USERNAME_CONFLICT",
            {"username": "johndoe"}
        )
        
        assert isinstance(error, ServiceError)
        assert error.message == "Username already taken"
        assert error.error_code == "USERNAME_CONFLICT"


class TestPermissionDeniedError:
    """Test PermissionDeniedError exception."""
    
    def test_permission_denied_error(self):
        """Test permission denied error creation."""
        error = PermissionDeniedError(
            "Insufficient permissions to delete conversation",
            "INSUFFICIENT_PERMISSIONS",
            {
                "required_permission": "conversations:delete",
                "actor_type": "user"
            }
        )
        
        assert isinstance(error, ServiceError)
        assert error.message == "Insufficient permissions to delete conversation"
        assert error.error_code == "INSUFFICIENT_PERMISSIONS"
        assert error.details["required_permission"] == "conversations:delete"


class TestConfigurationError:
    """Test ConfigurationError exception."""
    
    def test_configuration_error(self):
        """Test configuration error creation."""
        error = ConfigurationError(
            "Database connection string not configured",
            "MISSING_DB_CONFIG"
        )
        
        assert isinstance(error, ServiceError)
        assert error.message == "Database connection string not configured"
        assert error.error_code == "MISSING_DB_CONFIG"


class TestExternalServiceError:
    """Test ExternalServiceError exception."""
    
    def test_external_service_error(self):
        """Test external service error creation."""
        error = ExternalServiceError(
            "Failed to connect to OpenAI API",
            "EXTERNAL_API_ERROR",
            {"service": "openai", "status_code": 503}
        )
        
        assert isinstance(error, ServiceError)
        assert error.message == "Failed to connect to OpenAI API"
        assert error.error_code == "EXTERNAL_API_ERROR"
        assert error.details["service"] == "openai"
        assert error.details["status_code"] == 503


class TestRateLimitError:
    """Test RateLimitError exception."""
    
    def test_rate_limit_error(self):
        """Test rate limit error creation."""
        error = RateLimitError(
            "API rate limit exceeded",
            "RATE_LIMIT_EXCEEDED",
            {"retry_after": 60}
        )
        
        assert isinstance(error, ServiceError)
        assert error.message == "API rate limit exceeded"
        assert error.error_code == "RATE_LIMIT_EXCEEDED"
        assert error.details["retry_after"] == 60


class TestBusinessRuleError:
    """Test BusinessRuleError exception."""
    
    def test_business_rule_error(self):
        """Test business rule error creation."""
        error = BusinessRuleError(
            "Cannot delete conversation with active agents",
            "ACTIVE_AGENTS_PREVENT_DELETE",
            {"conversation_id": "conv_123", "active_agents": 2}
        )
        
        assert isinstance(error, ServiceError)
        assert error.message == "Cannot delete conversation with active agents"
        assert error.error_code == "ACTIVE_AGENTS_PREVENT_DELETE"
        assert error.details["active_agents"] == 2


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""
    
    def test_all_inherit_from_service_error(self):
        """Test that all custom exceptions inherit from ServiceError."""
        exceptions = [
            ValidationError("test"),
            NotFoundError("test"),
            ConflictError("test"),
            PermissionDeniedError("test"),
            ConfigurationError("test"),
            ExternalServiceError("test"),
            RateLimitError("test"),
            BusinessRuleError("test"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, ServiceError)
            assert isinstance(exc, Exception)
    
    def test_can_catch_as_service_error(self):
        """Test that all exceptions can be caught as ServiceError."""
        with pytest.raises(ServiceError):
            raise ValidationError("test validation error")
        
        with pytest.raises(ServiceError):
            raise NotFoundError("test not found error")
        
        with pytest.raises(ServiceError):
            raise PermissionDeniedError("test permission error")


class TestErrorCodeDefaults:
    """Test default error code generation."""
    
    def test_default_error_codes(self):
        """Test that exceptions get appropriate default error codes."""
        error = ValidationError("test")
        assert error.error_code == "VALIDATIONERROR"
        
        error = NotFoundError("test")
        assert error.error_code == "NOTFOUNDERROR"
        
        error = ConflictError("test")
        assert error.error_code == "CONFLICTERROR"
    
    def test_custom_error_code_overrides_default(self):
        """Test that custom error codes override defaults."""
        error = ValidationError("test", "CUSTOM_VALIDATION_ERROR")
        assert error.error_code == "CUSTOM_VALIDATION_ERROR"