"""Configuration manager and helper sections for ATLAS."""

from .config_manager import ConfigManager
from .core import (
    ConfigCore,
    _DEFAULT_CONVERSATION_STORE_DSN,
    find_dotenv,
    load_dotenv,
    set_key,
    setup_logger,
)
from .messaging import MessagingConfigSection, setup_message_bus
from .persistence import PersistenceConfigSection, PersistenceConfigMixin, KV_STORE_UNSET
from .providers import ProviderConfigSections, ProviderConfigMixin
from .tooling import ToolingConfigSection
from .ui_config import UIConfig

__all__ = [
    "ConfigManager",
    "ConfigCore",
    "ProviderConfigMixin",
    "ProviderConfigSections",
    "PersistenceConfigMixin",
    "MessagingConfigSection",
    "setup_message_bus",
    "PersistenceConfigSection",
    "ToolingConfigSection",
    "UIConfig",
    "KV_STORE_UNSET",
    "_DEFAULT_CONVERSATION_STORE_DSN",
    "find_dotenv",
    "load_dotenv",
    "set_key",
    "setup_logger",
]
