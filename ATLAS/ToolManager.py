"""Compatibility façade for tool management utilities."""
from __future__ import annotations

import asyncio
import random
import socket

from modules.Tools.tool_event_system import event_system, publish_bus_event

from ATLAS.config import ConfigManager
from ATLAS.tools import cache as _cache
from ATLAS.tools import execution as _execution
from ATLAS.tools import manifests as _manifests
from ATLAS.tools import streaming as _streaming
from ATLAS.tools.errors import ToolExecutionError, ToolManifestValidationError

# Expose manifest helpers and caches
load_default_function_map = _manifests.load_default_function_map
load_function_map_from_current_persona = _manifests.load_function_map_from_current_persona
load_functions_from_json = _manifests.load_functions_from_json
_function_map_cache = _manifests._function_map_cache
_function_payload_cache = _manifests._function_payload_cache
_function_payload_cache_lock = _manifests._function_payload_cache_lock
_default_function_map_cache = _manifests._default_function_map_cache
_default_function_map_lock = _manifests._default_function_map_lock
_DEFAULT_FUNCTIONS_CACHE_KEY = _manifests._DEFAULT_FUNCTIONS_CACHE_KEY
_tool_manifest_validator = _manifests._tool_manifest_validator
_tool_manifest_validator_lock = _manifests._tool_manifest_validator_lock
_get_tool_manifest_validator = _manifests._get_tool_manifest_validator

# Expose caching/logging utilities
_get_config_manager = _cache.get_config_manager
_get_config_section = _cache.get_config_section
_clone_json_compatible = _cache.clone_json_compatible
_stringify_tool_value = _cache.stringify_tool_value
_record_tool_activity = _cache.record_tool_activity
_record_tool_failure = _cache.record_tool_failure
get_tool_activity_log = _cache.get_tool_activity_log
_get_tool_logging_preferences = _cache.get_tool_logging_preferences
_build_tool_metrics = _cache.build_tool_metrics
_build_public_tool_entry = _cache.build_public_tool_entry
_tool_activity_log = _cache._tool_activity_log
_tool_activity_lock = _cache._tool_activity_lock

# Stream helpers
_collect_async_chunks = _streaming.collect_async_chunks
_stream_tool_iterator = _streaming.stream_tool_iterator
_gather_async_iterator = _streaming.gather_async_iterator
_is_async_stream = _streaming.is_async_stream
ToolStreamCapture = _streaming.ToolStreamCapture

# Execution utilities
ToolPolicyDecision = _execution.ToolPolicyDecision
SandboxedToolRunner = _execution.SandboxedToolRunner
compute_tool_policy_snapshot = _execution.compute_tool_policy_snapshot
use_tool = _execution.use_tool
call_model_with_new_prompt = _execution.call_model_with_new_prompt

_freeze_generation_settings = _execution._freeze_generation_settings
_extract_text_and_audio = _execution._extract_text_and_audio
_store_assistant_message = _execution._store_assistant_message
_proxy_streaming_response = _execution._proxy_streaming_response
_resolve_provider_manager = _execution._resolve_provider_manager
get_required_args = _execution.get_required_args
_freeze_metadata = _execution._freeze_metadata
_extract_persona_name = _execution._extract_persona_name
_normalize_persona_allowlist = _execution._normalize_persona_allowlist
_join_with_and = _execution._join_with_and
_normalize_requires_flags = _execution._normalize_requires_flags
_coerce_persona_flag_value = _execution._coerce_persona_flag_value
_persona_flag_enabled = _execution._persona_flag_enabled
_collect_missing_flag_requirements = _execution._collect_missing_flag_requirements
_format_operation_flag_reason = _execution._format_operation_flag_reason
_format_denied_operations_summary = _execution._format_denied_operations_summary
_build_persona_context_snapshot = _execution._build_persona_context_snapshot
_has_tool_consent = _execution._has_tool_consent
_request_tool_consent = _execution._request_tool_consent
_evaluate_tool_policy = _execution._evaluate_tool_policy
_get_sandbox_runner = _execution._get_sandbox_runner
_resolve_tool_timeout_seconds = _execution._resolve_tool_timeout_seconds
_generate_idempotency_key = _execution._generate_idempotency_key
_is_tool_idempotent = _execution._is_tool_idempotent
_apply_idempotent_retry_backoff = _execution._apply_idempotent_retry_backoff
_run_with_timeout = _execution._run_with_timeout
_resolve_function_callable = _execution._resolve_function_callable

# Re-export frequently used modules/constants for backwards compatibility
asyncio = asyncio
random = random
socket = socket
ConfigManager = ConfigManager
__all__ = [
    "ToolExecutionError",
    "ToolManifestValidationError",
    "ToolPolicyDecision",
    "SandboxedToolRunner",
    "compute_tool_policy_snapshot",
    "use_tool",
    "call_model_with_new_prompt",
    "load_default_function_map",
    "load_function_map_from_current_persona",
    "load_functions_from_json",
    "get_tool_activity_log",
]
