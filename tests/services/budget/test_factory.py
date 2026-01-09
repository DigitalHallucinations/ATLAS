"""
Tests for Budget Service Factory functions.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import pytest

from core.services.budget.factory import (
    reset_services,
)


class TestFactoryFunctions:
    """Test the service factory functions."""
    
    def setup_method(self) -> None:
        """Reset services before each test."""
        reset_services()
    
    def teardown_method(self) -> None:
        """Reset services after each test."""
        reset_services()
    
    def test_reset_services_clears_instances(self) -> None:
        """Test that reset_services clears all cached instances."""
        from core.services.budget import factory
        
        # Set some dummy values
        factory._policy_service = "dummy"  # type: ignore
        factory._tracking_service = "dummy"  # type: ignore
        factory._alert_service = "dummy"  # type: ignore
        factory._initialized = True
        
        # Reset
        reset_services()
        
        # Verify all are None
        assert factory._policy_service is None
        assert factory._tracking_service is None
        assert factory._alert_service is None
        assert factory._initialized is False
    
    def test_imports_are_available(self) -> None:
        """Test that factory functions are importable from package."""
        from core.services.budget import (
            get_policy_service,
            get_tracking_service,
            get_alert_service,
            get_all_services,
            shutdown_services,
            reset_services,
        )
        
        # Just verify they're callable
        assert callable(get_policy_service)
        assert callable(get_tracking_service)
        assert callable(get_alert_service)
        assert callable(get_all_services)
        assert callable(shutdown_services)
        assert callable(reset_services)
