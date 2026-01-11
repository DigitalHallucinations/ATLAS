"""
Pytest configuration for service tests.

Author: ATLAS Team
Date: Jan 10, 2026
"""

import pytest

# Enable asyncio mode for all async tests in this directory
# Note: pytest_plugins should only be defined in top-level conftest.py
# pytest_plugins = ['pytest_asyncio']

# Configure pytest-asyncio to auto-detect async tests
def pytest_configure(config):
    """Configure pytest-asyncio to auto mode."""
    # Use asyncio mode auto for all async tests
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
