# ATLAS Codebase Architecture Summary for Image Generation & Provider Discussion

## 1. High-Level Architecture Overview

**ATLAS** is a modular AI assistant application built in Python 3.10+, featuring:

- Multiple LLM provider support (OpenAI, Anthropic, Google, Mistral, HuggingFace, Grok)
- Persona-based customization with tool/skill allowlists
- GTK-based UI shell
- RAG (Retrieval-Augmented Generation) capabilities
- Speech services (TTS/STT)
- Background task scheduling
- Job/Task management system

### Core Directory Structure

```Text
ATLAS/
├── ATLAS/                  # Core application logic
│   ├── ATLAS.py           # Main application class
│   ├── provider_manager.py # LLM provider orchestration (2543 lines)
│   ├── model_manager.py    # Model caching and switching
│   ├── persona_manager.py  # Persona loading/switching
│   ├── ToolManager.py      # Tool execution facade
│   ├── SkillManager.py     # Skill orchestration (772 lines)
│   ├── config/             # Configuration management
│   ├── providers/          # Provider adapters (base.py, openai.py, anthropic.py)
│   ├── services/           # Service facades (providers.py, speech.py, tooling.py)
│   └── tools/              # Tool execution (manifests.py, execution.py, streaming.py)
├── modules/
│   ├── Providers/          # Provider implementations
│   │   ├── OpenAI/        # OpenAI generator (2152 lines)
│   │   ├── Anthropic/     # Anthropic generator (1178 lines)
│   │   ├── Google/
│   │   ├── Mistral/
│   │   ├── HuggingFace/
│   │   └── Grok/
│   ├── Personas/           # Persona definitions (ATLAS, Einstein, MEDIC, etc.)
│   ├── Tools/              # Base tools and tool maps
│   ├── Skills/             # Skill definitions and schema
│   └── ...
├── config.yaml             # Global configuration
└── GTKUI/                  # GTK UI components
```

---

## 2. Provider System Architecture

### 2.1 ProviderManager (Singleton Pattern)

**File:** `ATLAS/provider_manager.py`

The `ProviderManager` is the central orchestrator for LLM providers. Key characteristics:

```python
class ProviderManager:
    """Manages interactions with different LLM providers, ensuring only one instance exists."""

    AVAILABLE_PROVIDERS = ["OpenAI", "Mistral", "Google", "HuggingFace", "Anthropic", "Grok"]
    
    _instance = None  # Singleton instance
    _lock: asyncio.Lock | None = None  # Thread-safe creation

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.model_manager = ModelManager(self.config_manager)
        self.current_llm_provider = self.config_manager.get_default_provider()
        self.current_model = None
        self.generate_response_func = None  # Current response generator
        self.providers = {}  # Cached provider callables
        
        # Provider-specific generators
        self.huggingface_generator = None
        self.grok_generator = None
        self.anthropic_generator: Optional[AnthropicGenerator] = None
        self._openai_generator = None
        self._mistral_generator = None
        self._google_generator = None
```

### 2.2 Provider Invoker Pattern

**File:** `ATLAS/providers/base.py`

Each provider can register a custom invocation adapter:

```python
ResultPayload = Dict[str, Any]
ProviderInvoker = Callable[["ProviderManager", Callable[..., Awaitable[Any]], Dict[str, Any]], Awaitable[Any]]

_PROVIDER_INVOKERS: Dict[str, ProviderInvoker] = {}

def register_invoker(name: str, invoker: ProviderInvoker) -> None:
    """Register an invocation strategy for the given provider."""
    _PROVIDER_INVOKERS[name] = invoker

def get_invoker(name: str) -> Optional[ProviderInvoker]:
    """Return the invocation strategy for ``name`` if one is registered."""
    return _PROVIDER_INVOKERS.get(name)

def build_result(success: bool, *, message: str = "", error: str = "", data: Any = None) -> ResultPayload:
    """Create a structured result payload for provider actions."""
    payload: ResultPayload = {"success": success}
    if success:
        if message: payload["message"] = message
        if data is not None: payload["data"] = data
    else:
        payload["error"] = error or message or "Unknown error"
    return payload
```

### 2.3 Provider Generator Pattern

Each provider has a generator class with this structure:

**Example: OpenAIGenerator** (`modules/Providers/OpenAI/OA_gen_response.py`)

```python
class OpenAIGenerator:
    def __init__(self, config_manager: ConfigManager, model_manager: Optional[ModelManager] = None):
        self.config_manager = config_manager
        self.api_key = self.config_manager.get_openai_api_key()
        settings = self.config_manager.get_openai_llm_settings()
        client_kwargs = {"api_key": self.api_key}
        base_url = settings.get("base_url")
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)
        self.model_manager = model_manager or ModelManager(config_manager)

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: Optional[bool] = None,
        current_persona=None,
        functions=None,
        # ... many more parameters
    ) -> Union[str, AsyncIterator[str]]:
        # Implementation with tool calling support
        ...
```

### 2.4 Provider Switching

```python
async def switch_llm_provider(self, llm_provider: str):
    """Switches the current LLM provider to the specified provider."""
    if llm_provider == self.current_llm_provider and self.generate_response_func is not None:
        return  # Already initialized

    if llm_provider not in self.AVAILABLE_PROVIDERS:
        llm_provider = "OpenAI"  # Fallback

    previous_provider = self.current_llm_provider
    
    # Cleanup previous provider
    if previous_provider and previous_provider != llm_provider:
        await self._cleanup_provider_generator(previous_provider)

    self.current_llm_provider = llm_provider

    if llm_provider == "OpenAI":
        openai_generator = self._ensure_openai_generator()
        self.generate_response_func = openai_generator.generate_response
        self.process_streaming_response_func = openai_generator.process_streaming_response
        default_model = self.get_default_model_for_provider("OpenAI")
        await self.set_model(default_model)
    # Similar patterns for other providers...
```

---

## 3. Configuration System

### 3.1 ConfigManager

**File:** `ATLAS/config/config_manager.py`

Inherits from multiple mixins: `ProviderConfigMixin`, `PersistenceConfigMixin`, `ConfigCore`

```python
class ConfigManager(ProviderConfigMixin, PersistenceConfigMixin, ConfigCore):
    def __init__(self):
        super().__init__()
        # Provider-specific sections
        self.providers = ProviderConfigSections(manager=self)
        self.providers.apply()
        
        # UI, tooling, persistence, messaging, storage architecture...
```

### 3.2 Provider Configuration Sections

**File:** `ATLAS/config/providers.py`

```python
class ProviderConfigSections:
    def __init__(self, manager: "ConfigManager") -> None:
        self.manager = manager
        self._env_keys: Dict[str, str] = {
            "OpenAI": "OPENAI_API_KEY",
            "Mistral": "MISTRAL_API_KEY",
            "Google": "GOOGLE_API_KEY",
            "HuggingFace": "HUGGINGFACE_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Grok": "GROK_API_KEY",
        }
        # Provider-specific config classes
        self.openai = OpenAIProviderConfig(manager=manager, registry=self)
        self.google = GoogleProviderConfig(manager=manager, registry=self)
        self.mistral = MistralProviderConfig(manager=manager, registry=self)
```

### 3.3 OpenAI Provider Settings Structure

```python
def get_llm_settings(self) -> Dict[str, Any]:
    defaults = {
        "model": self.manager.get_config("DEFAULT_MODEL", "gpt-4o"),
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4000,
        "max_output_tokens": None,
        "stream": True,
        "function_calling": True,
        "parallel_tool_calls": True,
        "tool_choice": None,
        "reasoning_effort": "medium",
        "base_url": ...,
        "organization": ...,
        "json_mode": False,
        "json_schema": None,
        "audio_enabled": False,
        "audio_voice": "alloy",
        "audio_format": "wav",
    }
    # Merge with stored settings
    ...
```

### 3.4 Configuration Files

**config.yaml** (root):

```yaml
DEFAULT_STT_PROVIDER: whisper
DEFAULT_TTS_PROVIDER: eleven_labs
TTS_ENABLED: false
tools:
  javascript_executor:
    default_timeout: 5.0
    ...
rag:
  embeddings:
    default_provider: huggingface
    openai:
      model: text-embedding-3-small
    ...
```

**ATLAS/config/atlas_config.yaml**:

```yaml
DEFAULT_MODEL: gpt-4o-mini
DEFAULT_PROVIDER: openai
MODEL_CACHE:
  OpenAI:
    - gpt-4o
    - gpt-4o-mini
    - dall-e-2     # Image models already listed!
    - dall-e-3
    ...
```

---

## 4. Tool System Architecture

### 4.1 Tool Manifest Schema

**File:** `modules/Tools/tool_maps/schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ATLAS Tool Manifest",
  "type": "array",
  "items": {
    "type": "object",
    "required": [
      "name", "description", "parameters", "version",
      "side_effects", "default_timeout", "auth",
      "allow_parallel", "idempotency_key"
    ],
    "properties": {
      "name": {"type": "string", "minLength": 1},
      "description": {"type": "string", "minLength": 1},
      "parameters": {
        "type": "object",
        "properties": {
          "type": {"type": "string", "enum": ["object"]},
          "properties": {"type": "object"},
          "required": {"type": "array", "items": {"type": "string"}}
        }
      },
      "version": {"type": "string"},
      "side_effects": {
        "type": "string",
        "enum": ["none", "write", "network", "read_external_service", 
                 "filesystem", "compute", "system", "database"]
      },
      "default_timeout": {"type": "integer", "minimum": 0},
      "auth": {
        "type": "object",
        "required": ["required"],
        "properties": {
          "required": {"type": "boolean"},
          "type": {"type": "string"},
          "env": {"type": "string"}
        }
      },
      "capabilities": {"type": "array", "items": {"type": "string"}},
      "providers": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "priority": {"type": "integer"},
            "health_check_interval": {"type": "number"}
          }
        }
      },
      "cost_per_call": {"type": "number"},
      "cost_unit": {"type": "string"}
    }
  }
}
```

### 4.2 Tool Manifest Example

**File:** `modules/Tools/tool_maps/functions.json`

```json
[
    {
        "name": "google_search",
        "version": "1.0.0",
        "side_effects": "none",
        "default_timeout": 30,
        "auth": {
            "required": true,
            "type": "api_key",
            "envs": {
                "GOOGLE_API_KEY": {"required": true},
                "GOOGLE_CSE_ID": {"required": true}
            }
        },
        "allow_parallel": true,
        "idempotency_key": false,
        "cost_per_call": 0.005,
        "cost_unit": "USD",
        "capabilities": ["web_search", "knowledge_lookup"],
        "providers": [
            {"name": "google_cse", "priority": 0, "health_check_interval": 300},
            {"name": "serpapi", "priority": 10, "health_check_interval": 300}
        ],
        "description": "A Google search result API...",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search content."},
                "k": {"type": "integer", "default": 10, "minimum": 1}
            },
            "required": ["query"]
        }
    }
]
```

### 4.3 Tool Manager Facade

**File:** `ATLAS/ToolManager.py`

Acts as a compatibility facade exposing:

```python
# Manifest helpers
load_default_function_map = _manifests.load_default_function_map
load_function_map_from_current_persona = _manifests.load_function_map_from_current_persona
load_functions_from_json = _manifests.load_functions_from_json

# Execution utilities
use_tool = _execution.use_tool
call_model_with_new_prompt = _execution.call_model_with_new_prompt
ToolPolicyDecision = _execution.ToolPolicyDecision
SandboxedToolRunner = _execution.SandboxedToolRunner
```

### 4.4 Tool Implementation Example

**File:** `modules/Tools/Base_Tools/browser.py`

```python
class BrowserTool:
    """Track virtual navigation requests without hitting the network."""

    def __init__(self) -> None:
        self._history: list[BrowserVisit] = []

    async def run(
        self,
        *,
        url: str,
        instructions: Optional[str] = None,
        annotations: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Mapping[str, object]:
        """Record a new navigation request and return the session snapshot."""
        # Implementation...
        return {
            "visit": asdict(visit),
            "history_length": len(self._history),
            "recent_history": [asdict(entry) for entry in self._history[-5:]],
        }
```

---

## 5. Skill System Architecture

### 5.1 Skill Schema

**File:** `modules/Skills/schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Skill Manifest",
  "type": "array",
  "items": {
    "type": "object",
    "required": [
      "name", "version", "instruction_prompt",
      "required_tools", "required_capabilities", "safety_notes"
    ],
    "properties": {
      "name": {"type": "string"},
      "version": {"type": "string"},
      "instruction_prompt": {"type": "string"},
      "required_tools": {"type": "array", "items": {"type": "string"}},
      "required_capabilities": {"type": "array", "items": {"type": "string"}},
      "safety_notes": {"type": "string"},
      "summary": {"type": "string"},
      "category": {"type": "string"},
      "capability_tags": {"type": "array", "items": {"type": "string"}}
    }
  }
}
```

### 5.2 Skill Definition Example

**File:** `modules/Skills/skills.json`

```json
{
    "name": "ResearchBrief",
    "version": "1.0.0",
    "instruction_prompt": "Run focused web research on the user's topic, extract the most reliable insights...",
    "required_tools": ["google_search", "webpage_fetch"],
    "required_capabilities": ["web_research", "critical_thinking"],
    "safety_notes": "Only cite reputable sources, flag uncertainty...",
    "summary": "Rapid web-research digest...",
    "category": "Knowledge Synthesis",
    "capability_tags": ["web_research", "analysis", "synthesis"]
}
```

### 5.3 SkillManager

**File:** `ATLAS/SkillManager.py`

```python
@dataclass(slots=True)
class SkillExecutionContext:
    conversation_id: str
    conversation_history: Iterable[Mapping[str, Any]]
    persona: Optional[Mapping[str, Any]] = None
    user: Optional[Mapping[str, Any]] = None
    state: Dict[str, Any] = field(default_factory=dict)  # Shared between tool invocations
    blackboard_client: Optional[BlackboardClient] = None

@dataclass(frozen=True)
class SkillRunResult:
    skill_name: str
    tool_results: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: Optional[str] = None
    required_capabilities: tuple[str, ...] = field(default_factory=tuple)
```

---

## 6. Persona System Architecture

### 6.1 Persona Schema

**File:** `modules/Personas/schema.json`

```json
{
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ATLAS Persona Definition",
    "type": "object",
    "required": ["persona"],
    "properties": {
        "persona": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["name", "content"],
                "properties": {
                    "name": {"type": "string"},
                    "meaning": {"type": "string"},
                    "content": {
                        "type": "object",
                        "required": ["start_locked", "editable_content", "end_locked"],
                        "properties": {
                            "start_locked": {"type": "string"},
                            "editable_content": {"type": "string"},
                            "end_locked": {"type": "string"}
                        }
                    },
                    "allowed_tools": {"type": "array", "items": {"type": "string"}},
                    "allowed_skills": {"type": "array", "items": {"type": "string"}},
                    "type": {
                        "type": "object",
                        "properties": {
                            "personal_assistant": {
                                "properties": {
                                    "access_to_calendar": {"type": "boolean"},
                                    "terminal_read_enabled": {"type": "boolean"},
                                    "terminal_write_enabled": {"type": "boolean"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

### 6.2 Persona Definition Example

**File:** `modules/Personas/ATLAS/Persona/ATLAS.json`

```json
{
    "persona": [{
        "name": "ATLAS",
        "meaning": "Adaptive Task Learning and Autonomous Systems.",
        "content": {
            "start_locked": "The name of the user you are speaking to is <<name>>...",
            "editable_content": "As the flagship persona of the ATLAS Personal Assistant...",
            "end_locked": "User Profile: <<Profile>>..."
        },
        "provider": "openai",
        "model": "gpt-4o",
        "user_profile_enabled": "True",
        "type": {
            "Agent": {"enabled": "True"},
            "personal_assistant": {
                "enabled": "True",
                "access_to_calendar": "False",
                "terminal_read_enabled": "False"
            }
        },
        "Speech_provider": "11labs",
        "voice": "jack",
        "allowed_tools": [
            "google_search",
            "get_current_info",
            // ...
        ]
    }]
}
```

### 6.3 Persona Structure on Disk

```Text
modules/Personas/
├── ATLAS/
│   ├── Persona/ATLAS.json      # Persona definition
│   ├── Toolbox/functions.json  # Persona-specific tools
│   ├── Skills/                 # Persona-specific skills
│   ├── Memory/
│   └── Tasks/
├── Einstein/
│   ├── Persona/Einstein.json
│   └── Toolbox/functions.json
├── MEDIC/
└── ...
```

---

## 7. Model Manager

**File:** `ATLAS/model_manager.py`

```python
class ModelManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.current_model = None
        self.current_provider = None
        self.models = {}  # Dict[provider, List[model_names]]
        self.lock = threading.Lock()
        self.load_models()

    def set_model(self, model_name: str, provider: str) -> None:
        with self.lock:
            if provider not in self.models:
                self.models[provider] = []
            if model_name not in self.models[provider]:
                self.models[provider].append(model_name)
            self.current_model = model_name
            self.current_provider = provider

    def update_models_for_provider(self, provider: str, models: List[str]) -> List[str]:
        """Replace the cached models for a provider while preserving known fallbacks."""
        normalized = self._normalize_models(models)
        with self.lock:
            # Preserve the existing default at the head of the list
            existing = self.models.get(provider, [])
            # ... merge logic
            self.models[provider] = normalized
        return normalized
```

---

## 8. Service Layer Pattern

### 8.1 ProviderService Facade

**File:** `ATLAS/services/providers.py`

```python
class ProviderService:
    """Centralise provider related helpers for the ATLAS facade."""

    def __init__(self, *, provider_manager, config_manager, logger, chat_session, speech_manager=None):
        self._provider_manager = provider_manager
        self._config_manager = config_manager
        self._chat_session = chat_session
        self._provider_change_listeners: List[Callable[[Dict[str, str]], None]] = []

    def run_in_background(
        self,
        coroutine_factory: Callable[[], Awaitable[Any]],
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> Future:
        return run_async_in_thread(coroutine_factory, ...)

    def set_current_provider_in_background(self, provider: str, ...) -> Future:
        return self.run_in_background(
            lambda: self.set_current_provider(provider),
            ...
        )
```

---

## 9. Key Extension Points for Image Generation

### 9.1 Where to Add New Capabilities

1. **New Provider Type (e.g., "ImageProvider")**:
   - Add to `AVAILABLE_PROVIDERS` in `ProviderManager`
   - Create generator class in `modules/Providers/ImageGen/`
   - Add config section in `ATLAS/config/providers.py`
   - Register invoker in `ATLAS/providers/base.py`

2. **New Tool (e.g., "generate_image")**:
   - Add entry to `modules/Tools/tool_maps/functions.json`
   - Create implementation in `modules/Tools/Base_Tools/`
   - Tool automatically discovered via manifest loader

3. **New Skill (e.g., "ImageCreation")**:
   - Add entry to `modules/Skills/skills.json`
   - Reference tools in `required_tools`
   - Add to persona's `allowed_skills`

### 9.2 Existing Image Model References

The model cache already includes image models:

```yaml
MODEL_CACHE:
  OpenAI:
    - dall-e-2
    - dall-e-3
    - gpt-image-1
    - gpt-image-1-mini
```

### 9.3 Provider Settings Pattern

Each provider can have specialized settings. For image generation, you'd add:

```python
def set_image_generation_settings(
    self,
    *,
    model: str,        # dall-e-3, etc.
    size: str,         # 1024x1024, 1792x1024, etc.
    quality: str,      # standard, hd
    style: str,        # vivid, natural
    response_format: str,  # url, b64_json
) -> Dict[str, Any]:
    ...
```

---

## 10. Key Design Patterns to Follow

1. **Singleton with async factory**: `ProviderManager.create()`
2. **Generator classes** with `generate_response()` method
3. **Manifest-driven tool discovery**
4. **Persona-scoped tool allowlists**
5. **Configuration via YAML + env vars**
6. **Service facades** for UI/API consumption
7. **Background task execution** via `run_async_in_thread()`
8. **Structured result payloads**: `build_result(success, message, error, data)`

---

## 11. Questions to Consider for Image Generation

1. **Should image generation be a separate provider or a capability within existing providers?**
   - OpenAI DALL-E uses the same API key but different endpoints
   - Could add `generate_image()` method to `OpenAIGenerator`

2. **Tool-based or direct integration?**
   - Tool: LLM decides when to call `generate_image` tool
   - Direct: Separate endpoint/method for explicit image requests

3. **Multi-provider support needed?**
   - OpenAI DALL-E, Stability AI, Midjourney API, local models (SD)?
   - If yes, create `ImageProviderManager` following existing patterns

4. **Storage/delivery of generated images?**
   - Return URL vs base64
   - Save to `assets/` directory
   - Integrate with storage manager

5. **Persona integration?**
   - Add `image_generation_enabled` flag to persona types
   - Control via `allowed_tools`

---

## 12. Image Generation Implementation Plan (2025–2026)

### 12.1 Provider Tiers & Selection

#### Tier A: Core + Reliable (Start Here)

| Provider | Rationale | Key Features |
| -------- | --------- | ------------ |
| **OpenAI Images / GPT Image** | Already have OpenAI plumbing; models in cache (`gpt-image-1`, `dall-e-3`) | Generate + edit, multi-image inputs, Responses API integration |
| **Google Imagen (Vertex AI)** | Enterprise story (auth, quotas, auditing), SynthID watermarking | Text→image, enterprise governance |
| **Stability AI (SD3/3.5)** | Classic multi-style diffusion, lots of knobs, non-OpenAI alternative | Platform API, style presets |

#### Tier B: Enterprise Governance / Provenance

| Provider                      | Rationale                              | Key Features                                 |
|-------------------------------|----------------------------------------|----------------------------------------------|
| **AWS Bedrock – Titan Image** | Enterprise-safe posture, AWS ops story | Content filters, watermarking, C2PA metadata |
| **Adobe Firefly Services**    | Creative workflows, licensing posture  | Background removal, expand/fill, bulk ops    |

#### Tier C: Creative Specialist APIs

| Provider | Rationale | Key Features |
| -------- | --------- | ------------ |
| **Runway API (Gen-4)** | Creative studio quality, stylized outputs | Reference-driven workflows, SDK |
| **Luma Dream Machine** | Async job model | Reference images support |
| **Ideogram** | Strong typography / text in images | Logos, posters, UI graphics |
| **Leonardo AI** | General-purpose + guidance features | Production API |

#### Tier D: Aggregators / Model Routers

| Provider | Rationale | Key Features |
| -------- | --------- | ------------ |
| **fal.ai** | Many generative models (FLUX, Ideogram, Recraft) behind one platform | Avoid writing 10 adapters |
| **Replicate** | Breadth: "run models via API" | Experimentation, fallback routing |
| **Black Forest Labs (FLUX)** | First-class FLUX support | Direct API |

---

### 12.2 Architecture: MediaProvider Subsystem

**Key Principle:** Don't overload the existing `ProviderManager` (LLMs) with image logic. Create a parallel `MediaProvider` subsystem.

#### New Module Layout

```text
modules/
  Providers/
    Media/
      __init__.py
      base.py                  # MediaProvider interface + normalize helpers
      registry.py              # register_provider(name, factory)
      manager.py               # MediaProviderManager (singleton)
      OpenAIImages/
        __init__.py
        provider.py
      Stability/
        __init__.py
        provider.py
      VertexImagen/
        __init__.py
        provider.py
      BedrockTitan/
        __init__.py
        provider.py
      Fal/
        __init__.py
        provider.py            # Aggregator adapter
```

#### MediaProvider Base Interface

```python
# modules/Providers/Media/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

class OutputFormat(Enum):
    URL = "url"
    BASE64 = "b64"
    FILEPATH = "filepath"

@dataclass
class ImageGenerationRequest:
    prompt: str
    model: Optional[str] = None
    n: int = 1
    size: Optional[str] = None           # e.g., "1024x1024"
    aspect_ratio: Optional[str] = None   # e.g., "16:9"
    style_preset: Optional[str] = None
    quality: Optional[str] = None        # "standard", "hd"
    seed: Optional[int] = None
    input_images: Optional[List[str]] = None  # file_id/path/url for img2img
    mask_image: Optional[str] = None          # for inpainting
    output_format: OutputFormat = OutputFormat.FILEPATH
    safety: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None  # trace_id, persona, cost_center

@dataclass
class GeneratedImage:
    id: str
    mime: str
    path: Optional[str] = None
    url: Optional[str] = None
    b64: Optional[str] = None
    seed_used: Optional[int] = None

@dataclass
class ImageGenerationResult:
    success: bool
    images: List[GeneratedImage]
    provider: str
    model: str
    timing_ms: int
    cost_estimate: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class MediaProvider(ABC):
    """Base interface for image generation providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass
    
    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of model identifiers this provider supports."""
        pass
    
    @abstractmethod
    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate image(s) from text prompt."""
        pass
    
    @abstractmethod
    async def edit_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Edit an existing image based on prompt and mask."""
        pass
    
    async def health_check(self) -> bool:
        """Check if provider is available."""
        return True
    
    async def close(self) -> None:
        """Cleanup resources."""
        pass
```

#### MediaProviderManager (Singleton)

```python
# modules/Providers/Media/manager.py

class MediaProviderManager:
    """Manages image generation providers following ProviderManager patterns."""
    
    AVAILABLE_PROVIDERS = [
        "openai_images", "stability", "vertex_imagen", 
        "bedrock_titan", "fal", "runway", "ideogram"
    ]
    
    _instance = None
    _lock: asyncio.Lock | None = None

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)
        self.current_provider: Optional[str] = None
        self._providers: Dict[str, MediaProvider] = {}
        self._artifact_store: Optional[ImageArtifactStore] = None
    
    @classmethod
    async def create(cls, config_manager: ConfigManager) -> "MediaProviderManager":
        """Async factory method for singleton."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        async with cls._lock:
            if cls._instance is None:
                instance = cls(config_manager)
                await instance._initialize()
                cls._instance = instance
            return cls._instance
    
    async def generate_image(
        self,
        request: ImageGenerationRequest,
        *,
        provider_override: Optional[str] = None,
    ) -> ImageGenerationResult:
        """Generate image using configured or specified provider."""
        provider_name = provider_override or self._select_provider(request)
        provider = await self._ensure_provider(provider_name)
        
        start_time = time.monotonic()
        result = await provider.generate_image(request)
        result.timing_ms = int((time.monotonic() - start_time) * 1000)
        
        # Persist artifacts
        if self._artifact_store and result.success:
            await self._artifact_store.save(result, request)
        
        return result
    
    def _select_provider(self, request: ImageGenerationRequest) -> str:
        """Select provider based on request characteristics and routing rules."""
        # Intent-based routing (can be expanded)
        metadata = request.metadata or {}
        
        if metadata.get("intent") == "typography":
            return "ideogram"
        if metadata.get("intent") == "enterprise":
            return "bedrock_titan"
        if metadata.get("intent") == "creative_studio":
            return "runway"
        
        # Default to OpenAI
        return self.current_provider or "openai_images"
```

---

### 12.3 Tool Manifest: `generate_image`

Add to `modules/Tools/tool_maps/functions.json`:

```json
{
    "name": "generate_image",
    "version": "1.0.0",
    "side_effects": "filesystem",
    "default_timeout": 120,
    "auth": {
        "required": true,
        "type": "api_key",
        "envs": {
            "OPENAI_API_KEY": {"required": false, "providers": ["openai_images"]},
            "STABILITY_API_KEY": {"required": false, "providers": ["stability"]},
            "GOOGLE_API_KEY": {"required": false, "providers": ["vertex_imagen"]},
            "FAL_API_KEY": {"required": false, "providers": ["fal"]}
        },
        "docs": "At least one image generation provider API key must be configured."
    },
    "allow_parallel": false,
    "idempotency_key": {
        "required": true,
        "scope": "per-prompt",
        "guidance": "Use hash of prompt + seed + model as idempotency key."
    },
    "cost_per_call": 0.04,
    "cost_unit": "USD",
    "capabilities": ["image_generation", "media", "creative"],
    "providers": [
        {"name": "openai_images", "priority": 0, "health_check_interval": 300},
        {"name": "stability", "priority": 10, "health_check_interval": 300},
        {"name": "vertex_imagen", "priority": 20, "health_check_interval": 300},
        {"name": "bedrock_titan", "priority": 30, "health_check_interval": 300},
        {"name": "fal", "priority": 100, "health_check_interval": 60}
    ],
    "description": "Generate images from text prompts using AI image generation models. Supports multiple providers (OpenAI DALL-E/GPT-Image, Stability AI, Google Imagen, AWS Titan, fal.ai). Returns file paths to generated images.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed text description of the image to generate.",
                "minLength": 1
            },
            "model": {
                "type": "string",
                "description": "Optional model identifier (e.g., 'dall-e-3', 'gpt-image-1', 'sd3-large'). Defaults to provider's best model."
            },
            "n": {
                "type": "integer",
                "description": "Number of images to generate.",
                "default": 1,
                "minimum": 1,
                "maximum": 4
            },
            "size": {
                "type": "string",
                "description": "Image dimensions (e.g., '1024x1024', '1792x1024').",
                "enum": ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Aspect ratio (alternative to size, e.g., '16:9', '1:1', '9:16')."
            },
            "style": {
                "type": "string",
                "description": "Style preset (provider-specific, e.g., 'vivid', 'natural', 'photographic', 'digital-art')."
            },
            "quality": {
                "type": "string",
                "description": "Quality level.",
                "enum": ["draft", "standard", "hd"]
            },
            "provider": {
                "type": "string",
                "description": "Explicitly select provider (openai_images, stability, vertex_imagen, bedrock_titan, fal)."
            },
            "reference_images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "File paths or URLs of reference images for image-to-image generation."
            }
        },
        "required": ["prompt"]
    }
}
```

---

### 12.4 Tool Implementation

```python
# modules/Tools/Base_Tools/generate_image.py

"""Image generation tool using the MediaProviderManager."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional

from modules.Providers.Media.base import (
    ImageGenerationRequest,
    OutputFormat,
)
from modules.Providers.Media.manager import MediaProviderManager
from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


class ImageGenerationTool:
    """Tool for generating images via configured media providers."""

    def __init__(self, media_provider_manager: MediaProviderManager):
        self._manager = media_provider_manager

    async def run(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        n: int = 1,
        size: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        style: Optional[str] = None,
        quality: Optional[str] = None,
        provider: Optional[str] = None,
        reference_images: Optional[List[str]] = None,
        # Context from tool execution
        conversation_id: Optional[str] = None,
        persona: Optional[str] = None,
        user: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Generate images from text prompt."""
        
        if not prompt or not prompt.strip():
            return {
                "success": False,
                "error": "A non-empty prompt is required for image generation.",
            }

        # Build request
        request = ImageGenerationRequest(
            prompt=prompt.strip(),
            model=model,
            n=min(n, 4),
            size=size,
            aspect_ratio=aspect_ratio,
            style_preset=style,
            quality=quality,
            input_images=reference_images,
            output_format=OutputFormat.FILEPATH,
            metadata={
                "conversation_id": conversation_id,
                "persona": persona,
                "user": user,
                "trace_id": self._generate_trace_id(prompt),
            },
        )

        try:
            result = await self._manager.generate_image(
                request, provider_override=provider
            )
        except Exception as exc:
            logger.error("Image generation failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "error": str(exc),
            }

        if not result.success:
            return {
                "success": False,
                "error": result.error or "Image generation failed.",
            }

        return {
            "success": True,
            "data": {
                "images": [
                    {
                        "id": img.id,
                        "mime": img.mime,
                        "path": img.path,
                        "url": img.url,
                    }
                    for img in result.images
                ],
                "provider": result.provider,
                "model": result.model,
                "timing_ms": result.timing_ms,
                "cost_estimate": result.cost_estimate,
            },
        }

    def _generate_trace_id(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]


__all__ = ["ImageGenerationTool"]
```

---

### 12.5 Skill Definition: `ImageCreation`

Add to `modules/Skills/skills.json`:

```json
{
    "name": "ImageCreation",
    "version": "1.0.0",
    "instruction_prompt": "When the user requests an image, illustration, artwork, or visual content, use the generate_image tool. Craft detailed, descriptive prompts that capture the user's intent including subject, style, mood, lighting, composition, and any specific details mentioned. For ambiguous requests, ask clarifying questions about style preferences (photorealistic, illustration, abstract, etc.) and key visual elements before generating.",
    "required_tools": ["generate_image"],
    "required_capabilities": ["image_generation", "creative"],
    "safety_notes": "Do not generate images of real people without explicit consent. Avoid generating violent, explicit, or harmful imagery. Respect copyright by not replicating copyrighted characters or artworks. When uncertain about content appropriateness, ask the user for clarification.",
    "summary": "Generate images from text descriptions using AI image generation models.",
    "category": "Creative",
    "capability_tags": ["image_generation", "creative", "media", "art"]
}
```

---

### 12.6 Image Artifact Store

```python
# modules/Providers/Media/artifact_store.py

"""Persistent storage for generated images with metadata sidecar."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from modules.Providers.Media.base import (
    ImageGenerationRequest,
    ImageGenerationResult,
    GeneratedImage,
)


class ImageArtifactStore:
    """Stores generated images with JSON metadata sidecars."""

    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            xdg_data = os.environ.get("XDG_DATA_HOME", "~/.local/share")
            base_path = Path(xdg_data).expanduser() / "ATLAS" / "assets" / "generated"
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_storage_dir(self) -> Path:
        """Get date-partitioned storage directory."""
        now = datetime.now(timezone.utc)
        dir_path = self.base_path / str(now.year) / f"{now.month:02d}"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    async def save(
        self,
        result: ImageGenerationResult,
        request: ImageGenerationRequest,
    ) -> list[str]:
        """Save images and metadata, return list of saved paths."""
        storage_dir = self._get_storage_dir()
        saved_paths = []

        for image in result.images:
            # Generate unique filename
            artifact_id = image.id or str(uuid.uuid4())
            ext = self._mime_to_ext(image.mime)
            image_filename = f"{artifact_id}{ext}"
            image_path = storage_dir / image_filename

            # Save image data (if b64, decode; if path exists, copy)
            if image.b64:
                import base64
                image_path.write_bytes(base64.b64decode(image.b64))
            elif image.path and Path(image.path).exists():
                import shutil
                shutil.copy2(image.path, image_path)
            
            # Update image path
            image.path = str(image_path)

            # Save metadata sidecar
            metadata = {
                "id": artifact_id,
                "prompt": request.prompt,
                "model": result.model,
                "provider": result.provider,
                "seed": image.seed_used,
                "size": request.size,
                "aspect_ratio": request.aspect_ratio,
                "style_preset": request.style_preset,
                "quality": request.quality,
                "timing_ms": result.timing_ms,
                "cost_estimate": result.cost_estimate,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "request_metadata": request.metadata,
            }
            metadata_path = storage_dir / f"{artifact_id}.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))

            saved_paths.append(str(image_path))

        return saved_paths

    def _mime_to_ext(self, mime: str) -> str:
        mapping = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        return mapping.get(mime, ".png")
```

---

### 12.7 OpenAI Images Provider Implementation

```python
# modules/Providers/Media/OpenAIImages/provider.py

"""OpenAI Images provider (DALL-E, GPT-Image models)."""

from __future__ import annotations

import base64
import uuid
from typing import List, Optional

from openai import AsyncOpenAI

from ATLAS.config import ConfigManager
from modules.Providers.Media.base import (
    MediaProvider,
    ImageGenerationRequest,
    ImageGenerationResult,
    GeneratedImage,
    OutputFormat,
)
from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


class OpenAIImagesProvider(MediaProvider):
    """OpenAI image generation provider (DALL-E 2/3, GPT-Image-1)."""

    SUPPORTED_MODELS = [
        "dall-e-2",
        "dall-e-3",
        "gpt-image-1",
        "gpt-image-1-mini",
    ]

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        api_key = config_manager.get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not configured.")
        
        settings = config_manager.get_openai_llm_settings()
        client_kwargs = {"api_key": api_key}
        if settings.get("base_url"):
            client_kwargs["base_url"] = settings["base_url"]
        
        self.client = AsyncOpenAI(**client_kwargs)

    @property
    def name(self) -> str:
        return "openai_images"

    @property
    def supported_models(self) -> List[str]:
        return self.SUPPORTED_MODELS

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using OpenAI's images API."""
        
        model = request.model or "gpt-image-1"
        
        # Build API request
        api_kwargs = {
            "model": model,
            "prompt": request.prompt,
            "n": request.n,
        }

        # Size handling
        if request.size:
            api_kwargs["size"] = request.size
        elif request.aspect_ratio:
            api_kwargs["size"] = self._aspect_ratio_to_size(request.aspect_ratio, model)
        else:
            api_kwargs["size"] = "1024x1024"

        # Quality (DALL-E 3 / GPT-Image)
        if request.quality and model in ("dall-e-3", "gpt-image-1"):
            quality_map = {"draft": "standard", "standard": "standard", "hd": "hd"}
            api_kwargs["quality"] = quality_map.get(request.quality, "standard")

        # Style (DALL-E 3)
        if request.style_preset and model == "dall-e-3":
            if request.style_preset in ("vivid", "natural"):
                api_kwargs["style"] = request.style_preset

        # Response format
        if request.output_format == OutputFormat.URL:
            api_kwargs["response_format"] = "url"
        else:
            api_kwargs["response_format"] = "b64_json"

        try:
            response = await self.client.images.generate(**api_kwargs)
        except Exception as exc:
            logger.error("OpenAI image generation failed: %s", exc, exc_info=True)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                timing_ms=0,
                error=str(exc),
            )

        # Parse response
        images: List[GeneratedImage] = []
        for item in response.data:
            img = GeneratedImage(
                id=str(uuid.uuid4()),
                mime="image/png",
                url=getattr(item, "url", None),
                b64=getattr(item, "b64_json", None),
                seed_used=None,  # OpenAI doesn't expose seed
            )
            images.append(img)

        return ImageGenerationResult(
            success=True,
            images=images,
            provider=self.name,
            model=model,
            timing_ms=0,  # Filled by manager
            cost_estimate=self._estimate_cost(model, request.n, api_kwargs.get("size")),
        )

    async def edit_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Edit image using OpenAI's images/edits API."""
        
        if not request.input_images:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="dall-e-2",
                timing_ms=0,
                error="Input image required for editing.",
            )

        # Implementation for image editing...
        # Uses self.client.images.edit(...)
        raise NotImplementedError("Image editing coming in Phase 2")

    def _aspect_ratio_to_size(self, aspect_ratio: str, model: str) -> str:
        """Convert aspect ratio to supported size."""
        ratio_map = {
            "1:1": "1024x1024",
            "16:9": "1792x1024",
            "9:16": "1024x1792",
            "4:3": "1024x1024",  # Closest available
            "3:4": "1024x1024",
        }
        return ratio_map.get(aspect_ratio, "1024x1024")

    def _estimate_cost(
        self, model: str, n: int, size: Optional[str]
    ) -> dict:
        """Estimate cost based on model and parameters."""
        # Approximate pricing (update as needed)
        base_costs = {
            "dall-e-2": {"256x256": 0.016, "512x512": 0.018, "1024x1024": 0.020},
            "dall-e-3": {"1024x1024": 0.040, "1024x1792": 0.080, "1792x1024": 0.080},
            "gpt-image-1": {"1024x1024": 0.040, "1024x1792": 0.080, "1792x1024": 0.080},
            "gpt-image-1-mini": {"1024x1024": 0.020},
        }
        
        model_costs = base_costs.get(model, {})
        per_image = model_costs.get(size or "1024x1024", 0.040)
        
        return {
            "estimated_usd": per_image * n,
            "per_image_usd": per_image,
            "model": model,
            "size": size,
        }

    async def close(self) -> None:
        if hasattr(self.client, "close"):
            await self.client.close()
```

---

### 12.8 Configuration Additions

Add to `ATLAS/config/providers.py`:

```python
class MediaProviderConfig:
    """Configuration section for image/media generation providers."""
    
    def __init__(self, manager: "ConfigManager"):
        self.manager = manager
    
    def get_image_generation_settings(self) -> Dict[str, Any]:
        defaults = {
            "default_provider": "openai_images",
            "default_model": "gpt-image-1",
            "default_size": "1024x1024",
            "default_quality": "standard",
            "max_images_per_request": 4,
            "artifact_storage_path": None,  # Uses XDG default
            "cost_tracking_enabled": True,
        }
        stored = self.manager.get_config("IMAGE_GENERATION")
        if isinstance(stored, dict):
            defaults.update(stored)
        return defaults
    
    def set_image_generation_settings(self, **kwargs) -> Dict[str, Any]:
        # Persist to YAML config...
        pass
```

Add to `config.yaml`:

```yaml
# Image Generation Configuration
image_generation:
  default_provider: openai_images
  default_model: gpt-image-1
  default_size: 1024x1024
  default_quality: standard
  max_images_per_request: 4
  cost_tracking_enabled: true
  
  providers:
    openai_images:
      enabled: true
    stability:
      enabled: false
      # api_key loaded from STABILITY_API_KEY env
    vertex_imagen:
      enabled: false
    fal:
      enabled: false
```

---

### 12.9 Persona Integration

Add to persona schema (`modules/Personas/schema.json`):

```json
"image_generation": {
    "type": "object",
    "properties": {
        "enabled": {
            "anyOf": [
                {"type": "boolean"},
                {"type": "string", "enum": ["True", "False", "true", "false"]}
            ]
        },
        "allowed_providers": {
            "type": "array",
            "items": {"type": "string"}
        },
        "max_images_per_session": {
            "type": "integer",
            "minimum": 0
        },
        "style_guidance": {
            "type": "string",
            "description": "Default style hints for this persona's image generation."
        }
    }
}
```

Example in ATLAS persona:

```json
"image_generation": {
    "enabled": "True",
    "allowed_providers": ["openai_images", "stability"],
    "style_guidance": "Modern, clean, professional aesthetic with balanced composition."
}
```

---

### 12.10 Provider Selection Strategy (Intent-Based Routing)

```python
def _select_provider(self, request: ImageGenerationRequest) -> str:
    """Smart provider selection based on request characteristics."""
    
    metadata = request.metadata or {}
    prompt_lower = request.prompt.lower()
    
    # Typography / text-heavy → Ideogram
    if any(kw in prompt_lower for kw in ["logo", "text", "typography", "poster", "sign"]):
        if self._is_available("ideogram"):
            return "ideogram"
    
    # Enterprise / compliance → Bedrock Titan
    if metadata.get("compliance_required") or metadata.get("enterprise"):
        if self._is_available("bedrock_titan"):
            return "bedrock_titan"
    
    # Creative studio / stylized → Runway
    if metadata.get("intent") == "creative_studio" or "artistic" in prompt_lower:
        if self._is_available("runway"):
            return "runway"
    
    # Photo-realistic / edits → OpenAI
    if "photo" in prompt_lower or "realistic" in prompt_lower:
        return "openai_images"
    
    # Cost-sensitive / draft → use cheaper option
    if request.quality == "draft":
        if self._is_available("fal"):
            return "fal"  # Often cheaper for drafts
    
    # Default fallback
    return self.current_provider or "openai_images"
```

---

### 12.11 Phase 1 Implementation Checklist

#### Immediate (Week 1-2)

- [ ] Create `modules/Providers/Media/` directory structure
- [ ] Implement `base.py` with interfaces
- [ ] Implement `OpenAIImagesProvider`
- [ ] Implement `MediaProviderManager` (singleton)
- [ ] Implement `ImageArtifactStore`
- [ ] Add `generate_image` tool manifest
- [ ] Add `ImageCreation` skill

#### Short-term (Week 3-4)

- [ ] Add one non-OpenAI provider (Stability or Bedrock Titan)
- [ ] Implement cost tracking
- [ ] Add persona `image_generation` flag support
- [ ] GTK UI: Display generated images in chat

#### Medium-term (Month 2)

- [ ] Add `edit_image` tool
- [ ] Implement fal.ai aggregator adapter
- [ ] Add image-to-image "continuation" feature
- [ ] Implement prompt compiler for provider-specific formatting

---

### 12.12 Creative ATLAS Features (Future)

1. **Prompt Compiler**: Transform user intent + persona style into provider-optimized prompts
2. **Image Continuations**: Store last generated image as conversation artifact for iterative refinement
3. **Cost-Aware Routing**: Use `cost_per_call` from manifests to choose "draft" vs "final render" providers
4. **Style Memory**: Learn user preferences across sessions for consistent aesthetic
5. **Gallery View**: GTK panel showing generation history with re-generation controls

---

## 13. Architecture Alignment Gaps (January 2026 Review)

The following items were identified as gaps between the image generation plan and the current ATLAS architecture. These should be addressed during implementation to ensure consistency.

### 13.1 Additional Providers Missing from Plan

#### HuggingFace Images

ATLAS already has full `HuggingFaceGenerator` infrastructure with `InferenceClient` support. The `InferenceClient` natively supports `text_to_image()` for models like FLUX, Stable Diffusion, etc.

**Add to Tier A/B:**

| Provider | Rationale | Key Features |
|----------|-----------|--------------|
| **HuggingFace Inference API** | Already integrated; hosts FLUX, SD, etc. | `text_to_image()`, local model support, ONNX |

**Implementation pattern:**

```python
# modules/Providers/Media/HuggingFace/provider.py

from huggingface_hub import InferenceClient

class HuggingFaceImagesProvider(MediaProvider):
    """HuggingFace image generation using existing InferenceClient patterns."""
    
    SUPPORTED_MODELS = [
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-3-medium",
    ]
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        api_key = config_manager.get_huggingface_api_key()
        self.client = InferenceClient(token=api_key)
    
    @property
    def name(self) -> str:
        return "huggingface_images"
    
    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        model = request.model or "black-forest-labs/FLUX.1-schnell"
        
        # InferenceClient.text_to_image returns PIL Image or bytes
        image_bytes = await asyncio.to_thread(
            self.client.text_to_image,
            request.prompt,
            model=model,
        )
        
        # Convert to GeneratedImage...
```

#### xAI Aurora (Grok Image Generation)

xAI launched **Grok Aurora** for image generation. ATLAS already has `GrokGenerator` using `xai_sdk` with API key infrastructure.

**Add to Tier A:**

| Provider | Rationale | Key Features |
|----------|-----------|--------------|
| **xAI Aurora** | Existing Grok integration; real-time generation | High quality, fast inference |

**Implementation pattern:**

```python
# modules/Providers/Media/XAI/provider.py

class XAIAuroraProvider(MediaProvider):
    """xAI Aurora image generation via xai_sdk."""
    
    SUPPORTED_MODELS = ["aurora", "aurora-hd"]
    
    def __init__(self, config_manager: ConfigManager):
        from xai_sdk import Client
        self.config_manager = config_manager
        api_key = config_manager.get_grok_api_key()
        self.client = Client(api_key=api_key)
    
    @property
    def name(self) -> str:
        return "xai_aurora"
```

#### Google Gemini Images (Dual-Path)

The plan mentions Vertex AI Imagen but Google also offers image generation through the Gemini API. ATLAS uses the `genai` SDK in `GG_gen_response.py`. Support both entry points:

| Entry Point | Use Case | Auth Model |
|-------------|----------|------------|
| **Vertex AI Imagen** | Enterprise (quotas, auditing, SynthID) | Service account / ADC |
| **Gemini API Images** | Consumer / simpler auth | API key |

---

### 13.2 StorageManager Integration

The plan proposes a standalone `ImageArtifactStore` with XDG paths, but ATLAS has a sophisticated `StorageManager` singleton (`modules/storage/manager.py`) that manages:

- Multiple backends (SQLite, PostgreSQL, MongoDB)
- `UnitOfWork` patterns
- Vector storage infrastructure
- Health monitoring

**Gap:** `ImageArtifactStore` should integrate with `StorageManager`, not duplicate infrastructure.

**Recommended approach:**

```python
# modules/storage/manager.py - Add to StorageManager

@dataclass
class StorageManager:
    # ... existing stores ...
    _images: Optional["ImageArtifactRepository"] = None
    
    @property
    def images(self) -> "ImageArtifactRepository":
        """Access the image artifact repository."""
        if self._images is None:
            self._images = ImageArtifactRepository(
                settings=self._settings,
                sql_pool=self._sql_pool,
            )
        return self._images
```

```python
# modules/storage/images/__init__.py

class ImageArtifactRepository:
    """Stores generated images with metadata, integrated with StorageManager."""
    
    def __init__(self, settings: StorageSettings, sql_pool: SQLPool):
        self.settings = settings
        self.sql_pool = sql_pool
        self._base_path = self._resolve_artifact_path()
    
    def _resolve_artifact_path(self) -> Path:
        """Use StorageSettings or XDG fallback for artifact storage."""
        if self.settings.image_artifact_path:
            return Path(self.settings.image_artifact_path)
        xdg_data = os.environ.get("XDG_DATA_HOME", "~/.local/share")
        return Path(xdg_data).expanduser() / "ATLAS" / "assets" / "generated"
    
    async def save(
        self,
        result: ImageGenerationResult,
        request: ImageGenerationRequest,
    ) -> List[str]:
        """Save images and metadata, return paths."""
        # ... implementation using UnitOfWork pattern ...
```

---

### 13.3 Provider Environment Keys Registry

`ProviderConfigSections` in `ATLAS/config/providers.py` maintains `_env_keys` for credential lookups. New image providers must be added:

```python
# ATLAS/config/providers.py - Update _env_keys

self._env_keys: Dict[str, str] = {
    # Existing LLM providers
    "OpenAI": "OPENAI_API_KEY",
    "Mistral": "MISTRAL_API_KEY",
    "Google": "GOOGLE_API_KEY",
    "HuggingFace": "HUGGINGFACE_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Grok": "GROK_API_KEY",
    "ElevenLabs": "XI_API_KEY",
    
    # New image generation providers
    "Stability": "STABILITY_API_KEY",
    "Fal": "FAL_API_KEY",
    "Ideogram": "IDEOGRAM_API_KEY",
    "Runway": "RUNWAY_API_KEY",
    "Replicate": "REPLICATE_API_TOKEN",
    "Leonardo": "LEONARDO_API_KEY",
}
```

---

### 13.4 MODEL_CACHE Updates

`ATLAS/config/atlas_config.yaml` lists models per provider. Add image model identifiers:

```yaml
MODEL_CACHE:
  OpenAI:
    # ... existing ...
    - dall-e-2
    - dall-e-3
    - gpt-image-1
    - gpt-image-1-mini
  
  HuggingFace:
    # LLM models...
    # Image models:
    - black-forest-labs/FLUX.1-dev
    - black-forest-labs/FLUX.1-schnell
    - stabilityai/stable-diffusion-xl-base-1.0
    - stabilityai/stable-diffusion-3-medium
    - stabilityai/stable-diffusion-3.5-large
  
  Google:
    # LLM models...
    # Image models:
    - imagen-3.0-generate-001
    - imagen-3.0-fast-generate-001
  
  Grok:
    # LLM models...
    # Image models:
    - aurora
    - aurora-hd
  
  Stability:
    - sd3-large
    - sd3-large-turbo
    - sd3-medium
    - stable-image-ultra
    - stable-image-core
```

---

### 13.5 Invoker Registration Pattern

`ProviderManager` uses `register_invoker()` for custom dispatch. `MediaProviderManager` should follow this pattern rather than inventing a new mechanism:

```python
# modules/Providers/Media/manager.py

from ATLAS.providers.base import register_invoker, get_invoker

# Register media provider invokers
register_invoker("openai_images", _invoke_openai_images)
register_invoker("huggingface_images", _invoke_huggingface_images)
register_invoker("xai_aurora", _invoke_xai_aurora)
register_invoker("stability", _invoke_stability)

async def _invoke_openai_images(
    manager: "MediaProviderManager",
    func: Callable,
    kwargs: Dict[str, Any],
) -> Any:
    """Custom invocation adapter for OpenAI Images."""
    provider = await manager._ensure_provider("openai_images")
    return await func(provider, **kwargs)
```

---

### 13.6 Conversation Schema for Image Messages

The conversation store needs schema updates to handle image content in messages:

```python
# Message content types to support:
{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Here's the image you requested:"},
        {
            "type": "image",
            "image_id": "abc123",
            "path": "/path/to/generated/image.png",
            "url": "https://...",  # Optional CDN URL
            "mime": "image/png",
            "prompt": "Original generation prompt",
            "provider": "openai_images",
            "model": "gpt-image-1",
        }
    ]
}
```

**Schema additions for conversation store:**

```sql
-- Image artifact reference table
CREATE TABLE image_artifacts (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    message_id UUID REFERENCES messages(id),
    file_path TEXT NOT NULL,
    mime_type VARCHAR(64) NOT NULL,
    prompt TEXT,
    provider VARCHAR(64),
    model VARCHAR(128),
    seed BIGINT,
    size VARCHAR(16),
    cost_usd DECIMAL(10, 6),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);
```

---

### 13.7 Cost Tracking Integration

The plan mentions `cost_per_call` in tool manifests. ATLAS has a `budget_limiter` tool. Image generation should integrate:

```python
# modules/Tools/Base_Tools/generate_image.py

class ImageGenerationTool:
    async def run(self, *, prompt: str, ...):
        # Check budget before generation
        if self._budget_limiter:
            estimated_cost = self._estimate_cost(model, n, size)
            allowed = await self._budget_limiter.check_allowance(
                operation="image_generation",
                estimated_cost=estimated_cost,
            )
            if not allowed:
                return {
                    "success": False,
                    "error": "Image generation would exceed budget limit.",
                }
        
        result = await self._manager.generate_image(request)
        
        # Record actual cost
        if self._budget_limiter and result.success:
            await self._budget_limiter.record_expense(
                operation="image_generation",
                cost=result.cost_estimate.get("estimated_usd", 0),
                metadata={"provider": result.provider, "model": result.model},
            )
        
        return result
```

---

### 13.8 Updated Phase 1 Checklist

#### Immediate (Week 1-2)

- [ ] Create `modules/Providers/Media/` directory structure
- [ ] Implement `base.py` with interfaces
- [ ] Implement `OpenAIImagesProvider`
- [ ] **Implement `HuggingFaceImagesProvider`** (uses existing InferenceClient)
- [ ] Implement `MediaProviderManager` (singleton, with `register_invoker()`)
- [ ] **Integrate `ImageArtifactRepository` with `StorageManager`**
- [ ] Add `generate_image` tool manifest
- [ ] Add `ImageCreation` skill
- [ ] **Add new provider env keys to `ProviderConfigSections`**

#### Short-term (Week 3-4)

- [ ] **Add xAI Aurora provider**
- [ ] Add Stability AI provider
- [ ] **Update MODEL_CACHE with image models for all providers**
- [ ] Implement cost tracking integration with `budget_limiter`
- [ ] Add persona `image_generation` flag support
- [ ] **Update conversation schema for image message parts**
- [ ] GTK UI: Display generated images in chat

#### Medium-term (Month 2)

- [ ] Add `edit_image` tool
- [ ] Add Google Gemini images (dual-path: Vertex AI + Gemini API)
- [ ] Implement fal.ai aggregator adapter
- [ ] Add image-to-image "continuation" feature
- [ ] Implement prompt compiler for provider-specific formatting
- [ ] Optional: CLIP embeddings for image search in RAG

---

### 13.9 RAG Integration (Optional Enhancement)

For searchable image generation history, consider:

1. **Image-to-text descriptions**: Store LLM-generated captions for retrieval
2. **CLIP embeddings**: Vector representations for image similarity search
3. **Knowledge base entries**: Auto-create KB entries for generated images

```python
# Optional: modules/storage/images/embeddings.py

class ImageEmbeddingService:
    """Generate CLIP embeddings for image search."""
    
    async def embed_image(self, image_path: str) -> List[float]:
        """Generate CLIP embedding for an image."""
        # Use HuggingFace sentence-transformers/clip-ViT-B-32
        ...
    
    async def search_similar(
        self, query_embedding: List[float], top_k: int = 10
    ) -> List[ImageArtifact]:
        """Find similar images by embedding."""
        ...
```
