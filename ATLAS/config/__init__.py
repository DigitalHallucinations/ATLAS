"""Configuration manager and helper sections for ATLAS."""

from .config_manager import ConfigManager
from .core import (
    ConfigCore,
    ConversationStoreBackendOption,
    _DEFAULT_CONVERSATION_STORE_DSN_BY_BACKEND,
    _DEFAULT_CONVERSATION_STORE_BACKENDS,
    find_dotenv,
    load_dotenv,
    set_key,
    setup_logger,
    default_conversation_store_backend_name,
    get_default_conversation_store_backend,
    get_default_conversation_store_backends,
    infer_conversation_store_backend,
)
from .conversation_summary import ConversationSummaryConfigSection
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
    "ConversationSummaryConfigSection",
    "MessagingConfigSection",
    "setup_message_bus",
    "PersistenceConfigSection",
    "ToolingConfigSection",
    "UIConfig",
    "KV_STORE_UNSET",
    "_DEFAULT_CONVERSATION_STORE_DSN_BY_BACKEND",
    "_DEFAULT_CONVERSATION_STORE_BACKENDS",
    "ConversationStoreBackendOption",
    "default_conversation_store_backend_name",
    "get_default_conversation_store_backend",
    "get_default_conversation_store_backends",
    "infer_conversation_store_backend",
    "find_dotenv",
    "load_dotenv",
    "set_key",
    "setup_logger",
]
