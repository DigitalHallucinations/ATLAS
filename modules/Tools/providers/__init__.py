"""Provider abstractions for tool execution."""

from .base import ProviderHealth, ToolProvider, ToolProviderSpec
from .registry import ProviderFactory, ToolProviderRegistry, tool_provider_registry
from . import atlas_runtime  # noqa: F401 - ensure runtime provider registration
from . import policy_reference  # noqa: F401 - ensure policy provider registration
from . import google_cse  # noqa: F401 - ensure Google CSE provider registration
from . import serpapi  # noqa: F401 - ensure SerpAPI provider registration side-effect
from . import internal_http_client  # noqa: F401 - ensure HTTP provider registration
from . import openweathermap  # noqa: F401 - ensure geocoding provider registration
from . import ip_api  # noqa: F401 - ensure IP geolocation provider registration
from . import ncbi_entrez  # noqa: F401 - ensure medical provider registration
from . import task_queue_default  # noqa: F401 - ensure task queue provider registration
from . import debian12_local  # noqa: F401 - ensure calendar provider registration
from . import kv_store  # noqa: F401 - ensure KV store provider registration
from .vector_store import in_memory  # noqa: F401 - ensure vector store providers register

__all__ = [
    "ProviderFactory",
    "ProviderHealth",
    "ToolProvider",
    "ToolProviderRegistry",
    "ToolProviderSpec",
    "tool_provider_registry",
]
