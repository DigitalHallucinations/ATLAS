import pytest
from unittest.mock import Mock, MagicMock
from core.services.common import Actor
from core.services.providers.config_service import ProviderConfigService
from core.services.providers.types import ProviderConfig, ProviderConfigEvent, ProviderType
from core.services.providers.permissions import ProviderPermissionChecker

@pytest.fixture
def mock_bus():
    return Mock()

@pytest.fixture
def config_service(mock_bus):
    # Bypass permission checks for unit tests by default, or mock them
    permissions = Mock(spec=ProviderPermissionChecker)
    config_manager = Mock()
    return ProviderConfigService(config_manager, message_bus=mock_bus, permission_checker=permissions)

@pytest.fixture
def admin_actor():
    return Actor(type="user", id="admin_user", tenant_id="system", permissions={"providers:admin"})

@pytest.fixture
def sample_config():
    return ProviderConfig(
        provider_id="test-provider-1",
        name="Test Provider",
        provider_type=ProviderType.LLM,
        enabled=True
    )

def test_configure_provider_new(config_service, admin_actor, sample_config, mock_bus):
    # Act - use configure_provider(actor, provider_id, config_updates)
    config_updates = {"name": sample_config.name, "enabled": sample_config.enabled}
    config_service.configure_provider(admin_actor, sample_config.provider_id, config_updates)

    # Assert - verify config_manager was called
    config_service._config.update_provider.assert_called_with(
        sample_config.provider_id, config_updates
    )
    
    # Check event published
    mock_bus.publish.assert_called_once()
    event = mock_bus.publish.call_args[0][0]
    assert isinstance(event, ProviderConfigEvent)
    assert event.provider_id == sample_config.provider_id

def test_configure_provider_update(config_service, admin_actor, sample_config, mock_bus):
    # Setup - initial config
    config_service.configure_provider(admin_actor, sample_config.provider_id, {"name": sample_config.name})
    mock_bus.reset_mock()
    
    # Act - update with new name
    config_service.configure_provider(admin_actor, sample_config.provider_id, {"name": "Updated Name"})

    # Assert - verify config_manager was called with update
    config_service._config.update_provider.assert_called_with(
        sample_config.provider_id, {"name": "Updated Name"}
    )
    
    # Check event
    mock_bus.publish.assert_called_once()
    event = mock_bus.publish.call_args[0][0]
    assert "name" in event.changed_fields

def test_delete_provider(config_service, admin_actor, sample_config, mock_bus):
    # Setup
    config_service.configure_provider(admin_actor, sample_config.provider_id, {"name": sample_config.name})
    mock_bus.reset_mock()
    
    # Act - use disable_provider as a proxy for "delete" (actual delete may not exist)
    config_service.disable_provider(admin_actor, sample_config.provider_id)
    
    # Check event
    mock_bus.publish.assert_called_once()
    event = mock_bus.publish.call_args[0][0]
    assert event.enabled is False

def test_list_providers(config_service, admin_actor, sample_config):
    # Act - list_providers returns hardcoded known providers currently
    providers = config_service.list_providers(admin_actor)
    
    # Assert - should return the known providers list
    assert len(providers) > 0
    # Verify structure
    for p in providers:
        assert p.provider_id is not None
        assert p.name is not None

def test_set_credentials(config_service, admin_actor, mock_bus):
    # Act
    config_service.set_credentials(admin_actor, "provider-1", {"api_key": "secret"})
    
    # Assert - verify event was published (credentials are not stored in event for security)
    mock_bus.publish.assert_called()
    event = mock_bus.publish.call_args[0][0]
    assert event.provider_id == "provider-1"
    assert "credentials" in event.changed_fields

def test_permission_enforcement(mock_bus):
    # Setup with a rigorous permission checker
    checker = Mock(spec=ProviderPermissionChecker)
    checker.check_write_permission.side_effect = PermissionError("Denied")
    
    config_manager = Mock()
    service = ProviderConfigService(config_manager, message_bus=mock_bus, permission_checker=checker)
    actor = Actor(type="user", id="guest", tenant_id="system", permissions={"guest"})
    
    # Act & Assert - use correct signature: (actor, provider_id, config_updates)
    with pytest.raises(PermissionError):
        service.configure_provider(actor, "p1", {"name": "test"})
    
    checker.check_write_permission.assert_called_with(actor)
