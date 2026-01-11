import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from core.services.common import Actor
from core.services.providers.health_service import ProviderHealthService
from core.services.providers.config_service import ProviderConfigService
from core.services.providers.types import ProviderConfig, ProviderHealth, ProviderStatus, ProviderHealthEvent, ProviderType
from core.services.providers.permissions import ProviderPermissionChecker

@pytest.fixture
def mock_bus():
    return Mock()

@pytest.fixture
def mock_config_service():
    return Mock(spec=ProviderConfigService)

@pytest.fixture
def health_service(mock_config_service, mock_bus):
    permissions = Mock(spec=ProviderPermissionChecker)
    return ProviderHealthService(mock_config_service, mock_bus, permission_checker=permissions)

@pytest.fixture
def actor():
    return Actor(type="user", id="test-user", tenant_id="system", permissions={"providers:read"})

@pytest.mark.asyncio
async def test_check_health_provider_not_found(health_service, mock_config_service, actor):
    # Setup
    mock_config_service.get_provider.return_value = None
    
    # Act
    health = await health_service.check_health(actor, "unknown-provider")
    
    # Assert
    assert health.provider_id == "unknown-provider"
    assert health.status == ProviderStatus.UNKNOWN
    assert health.error_message == "Provider not found"

@pytest.mark.asyncio
async def test_check_health_disabled(health_service, mock_config_service, actor):
    # Setup
    config = ProviderConfig(provider_id="disabled-provider", name="Disabled", provider_type=ProviderType.LLM, enabled=False)
    mock_config_service.get_provider.return_value = config
    
    # Act
    health = await health_service.check_health(actor, "disabled-provider")
    
    # Assert
    assert health.status == ProviderStatus.DISABLED

@pytest.mark.asyncio
async def test_check_health_success(health_service, mock_config_service, actor, mock_bus):
    # Setup
    config = ProviderConfig(provider_id="healthy-provider", name="Healthy", provider_type=ProviderType.LLM, enabled=True)
    mock_config_service.get_provider.return_value = config
    
    # Act
    # The current impl of check_health mocks logic and returns ENABLED with latency
    health = await health_service.check_health(actor, "healthy-provider")
    
    # Assert
    assert health.status == ProviderStatus.ENABLED
    assert health.latency_ms is not None
    
    # Check event emitted because status changed from UNKNOWN (initially empty)
    mock_bus.publish.assert_called()
    event = mock_bus.publish.call_args[0][0]
    assert isinstance(event, ProviderHealthEvent)
    assert event.new_status == ProviderStatus.ENABLED

@pytest.mark.asyncio
async def test_check_all_health(health_service, mock_config_service, actor):
    # Setup
    c1 = ProviderConfig(provider_id="p1", name="P1", provider_type=ProviderType.LLM, enabled=True)
    c2 = ProviderConfig(provider_id="p2", name="P2", provider_type=ProviderType.LLM, enabled=False)
    mock_config_service.list_providers.return_value = [c1, c2]
    
    # Mock calls to get_provider since check_health calls it individually
    mock_config_service.get_provider.side_effect = lambda a, pid: c1 if pid == "p1" else c2
    
    # Act
    results = await health_service.check_all_health(actor)
    
    # Assert
    assert len(results) == 2
    status_map = {r.provider_id: r.status for r in results}
    assert status_map["p1"] == ProviderStatus.ENABLED
    assert status_map["p2"] == ProviderStatus.DISABLED

def test_cache_retrieval(health_service, mock_config_service, actor):
    # Setup - Manually inject state
    health = ProviderHealth(provider_id="p1", status=ProviderStatus.ENABLED, last_check=datetime.now())
    health_service._update_health_state("p1", health)
    
    # Act
    retrieved = health_service.get_status(actor, "p1")
    
    # Assert
    assert retrieved == health
    
    all_statuses = health_service.get_all_statuses(actor)
    assert len(all_statuses) == 1
    assert all_statuses[0] == health
