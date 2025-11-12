# ATLAS/ATLAS.py

import base64
import binascii
import copy
from dataclasses import asdict
import json
from datetime import datetime
from concurrent.futures import Future
from pathlib import Path
from typing import (
    Any,
    AsyncIterator as TypingAsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)
from ATLAS import ToolManager as ToolManagerModule
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger
from modules.logging.audit import get_persona_audit_logger
from ATLAS.provider_manager import ProviderManager
from ATLAS.persona_manager import PersonaManager
from modules.Chat.chat_session import ChatHistoryExportError, ChatSession
from modules.conversation_store import ConversationStoreRepository
from modules.Speech_Services.speech_manager import SpeechManager
from modules.Tools.tool_event_system import (
    DualSubscription,
    event_system as _legacy_event_system,
    subscribe_bus_event as _subscribe_bus_event,
)
from modules.background_tasks import run_async_in_thread
from modules.orchestration.blackboard import (
    get_blackboard as _get_blackboard,
    stream_blackboard as _stream_blackboard,
)
from modules.user_accounts.user_account_facade import UserAccountFacade
from modules.user_accounts.user_account_service import PasswordRequirements
from modules.Server import AtlasServer
from modules.orchestration.job_manager import JobManager
from modules.orchestration.job_scheduler import JobScheduler
from modules.orchestration.task_manager import TaskManager
from ATLAS.services.tooling import ToolingService
from ATLAS.services.conversations import ConversationService
from ATLAS.services.providers import ProviderService
from ATLAS.services.speech import SpeechService

class ATLAS:
    """
    The main ATLAS application class that manages configurations, providers, personas, and speech services.
    """

    def __init__(self):
        """
        Initialize the ATLAS instance with synchronous initialization.
        """
        self.config_manager = ConfigManager()
        self.ui_config = self.config_manager.ui_config
        self.message_bus = self.config_manager.configure_message_bus()
        self.logger = setup_logger(__name__)
        self.persona_path = self.config_manager.get_app_root()
        self.current_persona = None
        self.user_account_facade: UserAccountFacade | None = None
        self.provider_manager = None
        self.provider_facade: ProviderService | None = None
        self._persona_manager: PersonaManager | None = None
        self.chat_session = None
        self._default_status_tooltip = "Active LLM provider/model and TTS status"
        self.speech_manager = SpeechManager(self.config_manager)  # Instantiate SpeechManager with ConfigManager
        self.speech_facade: SpeechService | None = SpeechService(
            speech_manager=self.speech_manager,
            logger=self.logger,
            status_summary_getter=self._speech_status_summary,
            default_status_tooltip=self._default_status_tooltip,
        )
        self._initialized = False
        self._pending_provider_change_listeners: List[
            Callable[[Dict[str, str]], None]
        ] = []
        self._persona_change_listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._message_dispatchers: List[Callable[[str, str], None]] = []
        self.message_dispatcher: Optional[Callable[[str, str], None]] = None
        self.server = AtlasServer(config_manager=self.config_manager)
        self.conversation_repository: ConversationStoreRepository | None = None
        self.conversation_service: ConversationService | None = None

        tenant_value = self.config_manager.config.get("tenant_id")
        if tenant_value is None:
            tenant_value = self.config_manager.env_config.get("TENANT_ID")
        tenant_text = str(tenant_value).strip() if tenant_value else "default"
        self.tenant_id = tenant_text or "default"

        self.tooling_service = ToolingService(
            config_manager=self.config_manager,
            tool_manager_module=ToolManagerModule,
            persona_manager=self._persona_manager,
            message_bus=self.message_bus,
            logger=self.logger,
            tenant_id=self.tenant_id,
        )

        try:
            session_factory = self.config_manager.get_conversation_store_session_factory()
        except Exception as exc:  # pragma: no cover - verification issues during startup
            self.logger.error(
                "Conversation store verification failed: %s", exc, exc_info=True
            )
            raise

        if session_factory is None or not self.config_manager.is_conversation_store_verified():
            message = (
                "Conversation store verification sentinel missing; run the standalone setup utility "
                "before launching ATLAS."
            )
            self.logger.error(message)
            raise RuntimeError(message)

        retention = self.config_manager.get_conversation_retention_policies()
        try:
            repository = ConversationStoreRepository(
                session_factory,
                retention=retention,
            )
        except Exception as exc:  # pragma: no cover - repository initialisation issues
            self.logger.error(
                "Conversation store unavailable: %s", exc, exc_info=True
            )
            raise
        else:
            self.conversation_repository = repository
            self.conversation_service = ConversationService(
                repository=self.conversation_repository,
                logger=self.logger,
                tenant_id=self.tenant_id,
                chat_session_getter=self._require_chat_session,
            )
            self.user_account_facade = UserAccountFacade(
                config_manager=self.config_manager,
                conversation_repository=self.conversation_repository,
                logger=self.logger,
            )
        self.user_account_facade.add_active_user_change_listener(
            self._handle_active_user_change
        )

    # ------------------------------------------------------------------
    # Task operations
    # ------------------------------------------------------------------
    def search_tasks(
        self,
        *,
        text: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        status: Optional[str] = None,
        owner_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fetch task listings using the server search or pagination routes."""

        server = getattr(self, "server", None)
        if server is None:
            raise RuntimeError("ATLAS server is not configured.")

        tenant_id = self.tenant_id or "default"
        context: Dict[str, Any] = {"tenant_id": tenant_id}

        base_filters: Dict[str, Any] = {}
        if status:
            base_filters["status"] = status
        if owner_id:
            base_filters["owner_id"] = owner_id
        if conversation_id:
            base_filters["conversation_id"] = conversation_id

        text_query = str(text or "").strip()
        metadata_mapping: Optional[Dict[str, Any]]
        if isinstance(metadata, Mapping):
            metadata_mapping = {
                str(key): value
                for key, value in metadata.items()
                if str(key).strip()
            }
        else:
            metadata_mapping = None

        use_search_route = bool(text_query or metadata_mapping)

        if use_search_route:
            search_method = getattr(server, "search_tasks", None)
            if not callable(search_method):
                raise RuntimeError("Server search_tasks route is unavailable.")

            payload: Dict[str, Any] = dict(base_filters)
            if text_query:
                payload["text"] = text_query
            if metadata_mapping:
                payload["metadata"] = metadata_mapping

            if offset is not None:
                try:
                    offset_value = int(offset)
                except (TypeError, ValueError):
                    offset_value = 0
                payload["offset"] = max(0, offset_value)

            if limit is not None:
                try:
                    limit_value = int(limit)
                except (TypeError, ValueError):
                    limit_value = None
                else:
                    if limit_value > 0:
                        payload["limit"] = limit_value

            response = search_method(payload, context=context)
        else:
            list_method = getattr(server, "list_tasks", None)
            if not callable(list_method):
                raise RuntimeError("Server list_tasks route is unavailable.")

            params: Dict[str, Any] = dict(base_filters)
            if cursor:
                params["cursor"] = cursor

            if limit is not None:
                try:
                    page_size = int(limit)
                except (TypeError, ValueError):
                    page_size = None
                else:
                    if page_size > 0:
                        params["page_size"] = page_size

            response = list_method(params, context=context)

        if isinstance(response, Mapping):
            return dict(response)

        items: List[Any]
        if isinstance(response, Iterable) and not isinstance(response, (str, bytes)):
            items = list(response)
        else:
            items = []

        payload: Dict[str, Any] = {"items": items}
        if use_search_route:
            payload["count"] = len(items)
        else:
            payload["page"] = {"next_cursor": None, "count": len(items)}
        return payload

    def _require_user_account_facade(self) -> UserAccountFacade:
        """Return the initialized user account facade."""

        if self.user_account_facade is None:
            raise RuntimeError("User account facade is not initialized.")

        return self.user_account_facade

    def _resolve_user_identity(self, *, prefer_generic: bool = False) -> Tuple[str, str]:
        return self._require_user_account_facade().resolve_user_identity(
            prefer_generic=prefer_generic
        )

    def _refresh_active_user_identity(
        self, *, prefer_generic: bool = False
    ) -> Tuple[str, str]:
        username, display_name = self._require_user_account_facade().refresh_active_user_identity(
            prefer_generic=prefer_generic
        )
        return username, display_name

    def _ensure_user_identity(self) -> Tuple[str, str]:
        return self._require_user_account_facade().ensure_user_identity()

    def get_user_display_name(self) -> str:
        return self._require_user_account_facade().get_user_display_name()

    @property
    def user(self) -> str:
        return self.get_user_display_name()

    @user.setter
    def user(self, value: str) -> None:
        self._require_user_account_facade().override_user_identity(value)

    @property
    def task_manager(self) -> TaskManager | None:
        return self.tooling_service.task_manager

    @property
    def job_manager(self) -> JobManager | None:
        return self.tooling_service.job_manager

    @property
    def job_scheduler(self) -> JobScheduler | None:
        return self.tooling_service.job_scheduler

    @property
    def job_repository(self):
        return self.tooling_service.job_repository

    @property
    def job_service(self):
        return self.tooling_service.job_service

    def _require_server(self) -> AtlasServer:
        server = getattr(self, "server", None)
        if server is None:
            raise RuntimeError("ATLAS server is not configured.")
        return server

    def link_job_task(
        self,
        job_id: str,
        task_id: Any,
        *,
        relationship_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attach ``task_id`` to ``job_id`` via the server routes."""

        payload: Dict[str, Any] = {"task_id": str(task_id)}

        relationship = str(relationship_type or "").strip()
        if relationship:
            payload["relationship_type"] = relationship

        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata must be a mapping")
            payload["metadata"] = dict(metadata)

        context = {"tenant_id": self.tenant_id or "default"}
        server = self._require_server()
        return server.link_job_task(str(job_id), payload, context=context)

    def unlink_job_task(
        self,
        job_id: str,
        *,
        link_id: Any | None = None,
        task_id: Any | None = None,
    ) -> Dict[str, Any]:
        """Detach a linked task from ``job_id`` using the server routes."""

        if link_id is None and task_id is None:
            raise ValueError("Either link_id or task_id must be provided")

        context = {"tenant_id": self.tenant_id or "default"}
        server = self._require_server()
        return server.unlink_job_task(
            str(job_id),
            context=context,
            link_id=link_id,
            task_id=task_id,
        )

    def create_job(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        personas: Optional[Iterable[Any]] = None,
        schedule: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a job using the active tenant context."""

        normalized_name = self._require_non_empty_text(name, field_name="name")

        payload: Dict[str, Any] = {"name": normalized_name}
        description_text = self._normalize_optional_text(description)
        if description_text is not None:
            payload["description"] = description_text

        metadata_payload: Dict[str, Any] = {}
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata must be a mapping")
            metadata_payload.update(dict(metadata))

        roster = self._normalize_persona_roster(personas)
        if roster:
            metadata_payload.setdefault("personas", roster)

        if schedule is not None:
            metadata_payload["schedule"] = self._normalize_mapping(
                schedule, field_name="schedule"
            )

        if metadata_payload:
            payload["metadata"] = metadata_payload

        context = {"tenant_id": self.tenant_id or "default"}
        server = self._require_server()
        return server.create_job(payload, context=context)

    def _normalize_mapping(
        self, value: Mapping[str, Any] | None, *, field_name: str
    ) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"{field_name} must be a mapping")
        return dict(value)

    def _normalize_persona_roster(
        self, roster: Optional[Iterable[Any]]
    ) -> List[str]:
        if roster is None:
            return []

        if isinstance(roster, (str, bytes)):
            text = str(roster).strip()
            return [text] if text else []

        names: List[str] = []
        for entry in roster:
            value: Optional[str]
            if isinstance(entry, Mapping):
                raw = entry.get("name") or entry.get("persona") or entry.get("id")
                value = str(raw).strip() if raw else None
            elif entry is not None:
                value = str(entry).strip()
            else:
                value = None
            if value:
                names.append(value)
        seen: set[str] = set()
        unique_names: List[str] = []
        for name in names:
            if name not in seen:
                unique_names.append(name)
                seen.add(name)
        return unique_names

    def _require_non_empty_text(self, value: Any, *, field_name: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{field_name} is required")
        return text

    def _normalize_optional_text(self, value: Any | None) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @property
    def persona_manager(self) -> PersonaManager | None:
        return self._persona_manager

    def _speech_status_summary(self) -> Dict[str, Any]:
        try:
            return self.get_chat_status_summary()
        except Exception as exc:  # pragma: no cover - defensive guard for bootstrap
            self.logger.debug(
                "Unable to obtain chat status summary for speech facade: %s", exc
            )
            return {}

    def _require_speech_facade(self) -> SpeechService:
        if self.speech_facade is None:
            raise RuntimeError("Speech facade is not initialized.")
        return self.speech_facade

    @persona_manager.setter
    def persona_manager(self, value: PersonaManager | None) -> None:
        self._persona_manager = value
        if hasattr(self, "tooling_service"):
            self.tooling_service.set_persona_manager(value)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return merged tool metadata with persisted configuration state."""

        return self.tooling_service.list_tools()

    def update_tool_settings(self, tool_name: str, settings: Mapping[str, Any]) -> Dict[str, Any]:
        """Persist tool settings and refresh dependent caches."""

        return self.tooling_service.update_tool_settings(tool_name, settings)

    def update_tool_credentials(
        self,
        tool_name: str,
        credentials: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Persist tool credentials according to manifest metadata."""

        return self.tooling_service.update_tool_credentials(tool_name, credentials)

    def list_skills(self) -> List[Dict[str, Any]]:
        """Return merged skill metadata with persisted configuration state."""

        return self.tooling_service.list_skills()

    def update_skill_settings(self, skill_name: str, settings: Mapping[str, Any]) -> Dict[str, Any]:
        """Persist skill settings and refresh dependent caches."""

        return self.tooling_service.update_skill_settings(skill_name, settings)

    def update_skill_credentials(
        self,
        skill_name: str,
        credentials: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Persist skill credentials according to manifest metadata."""

        return self.tooling_service.update_skill_credentials(skill_name, credentials)

    def add_active_user_change_listener(
        self, listener: Callable[[str, str], None]
    ) -> None:
        """Register a callback notified whenever the active user changes."""

        self._require_user_account_facade().add_active_user_change_listener(listener)

    def remove_active_user_change_listener(
        self, listener: Callable[[str, str], None]
    ) -> None:
        """Remove a previously registered active user listener."""

        self._require_user_account_facade().remove_active_user_change_listener(listener)

    def _update_persona_manager_user(self, username: str) -> None:
        """Propagate active user changes to the persona manager if available."""

        manager = getattr(self, "persona_manager", None)
        if manager is None:
            return

        setter = getattr(manager, "set_user", None)
        if not callable(setter):
            return

        try:
            setter(username)
        except Exception:  # pragma: no cover - persona manager updates should not break flow
            self.logger.error(
                "Persona manager failed to accept active user '%s'", username, exc_info=True
            )
        else:
            self._notify_persona_change_listeners()

    def _handle_active_user_change(self, username: str, _display_name: str) -> None:
        """Internal bridge between the facade event and persona updates."""

        self._update_persona_manager_user(username)

    async def list_user_accounts(self) -> List[Dict[str, object]]:
        """Return stored user accounts without blocking the event loop."""

        return await self._require_user_account_facade().list_user_accounts()

    async def search_user_accounts(self, query_text: str) -> List[Dict[str, object]]:
        """Search stored user accounts using the service layer."""

        return await self._require_user_account_facade().search_user_accounts(query_text)

    async def get_user_account_details(self, username: str) -> Optional[Dict[str, object]]:
        """Retrieve a single account record for display purposes."""

        return await self._require_user_account_facade().get_user_account_details(username)

    async def get_user_account_overview(self) -> Dict[str, object]:
        """Return aggregated statistics for stored user accounts."""

        return await self._require_user_account_facade().get_user_account_overview()

    async def activate_user_account(self, username: str) -> None:
        """Activate an existing user account without credential prompts."""

        await self._require_user_account_facade().activate_user_account(username)

    async def register_user_account(
        self,
        username: str,
        password: str,
        email: str,
        name: Optional[str] = None,
        dob: Optional[str] = None,
    ) -> Dict[str, object]:
        """Register a new user account through the dedicated service."""

        return await self._require_user_account_facade().register_user_account(
            username,
            password,
            email,
            name,
            dob,
        )

    async def update_user_account(
        self,
        username: str,
        *,
        password: Optional[str] = None,
        current_password: Optional[str] = None,
        email: Optional[str] = None,
        name: Optional[str] = None,
        dob: Optional[str] = None,
    ) -> Dict[str, object]:
        """Update an existing user account and refresh cached identity."""

        return await self._require_user_account_facade().update_user_account(
            username,
            password=password,
            current_password=current_password,
            email=email,
            name=name,
            dob=dob,
        )

    def get_user_password_requirements(self) -> PasswordRequirements:
        """Return the password policy enforced for user accounts."""

        return self._require_user_account_facade().get_user_password_requirements()

    def describe_user_password_requirements(self) -> str:
        """Return a human-readable description of the password policy."""

        return self._require_user_account_facade().describe_user_password_requirements()

    async def login_user_account(self, username: str, password: str) -> bool:
        """Validate credentials and mark the account as active."""

        return await self._require_user_account_facade().login_user_account(
            username,
            password,
        )

    async def request_password_reset(self, identifier: str) -> Optional[Dict[str, object]]:
        """Initiate a password reset flow for the supplied identifier."""

        return await self._require_user_account_facade().request_password_reset(identifier)

    async def verify_password_reset_token(self, username: str, token: str) -> bool:
        """Check whether the supplied password reset token remains valid."""

        return await self._require_user_account_facade().verify_password_reset_token(
            username,
            token,
        )

    async def complete_password_reset(
        self, username: str, token: str, new_password: str
    ) -> bool:
        """Finish the password reset process by storing a new password."""

        return await self._require_user_account_facade().complete_password_reset(
            username,
            token,
            new_password,
        )

    async def logout_active_user(self) -> None:
        """Clear any active account selection."""

        await self._require_user_account_facade().logout_active_user()

    async def delete_user_account(self, username: str) -> None:
        """Delete a stored user account via the service layer."""

        await self._require_user_account_facade().delete_user_account(username)

    def _require_provider_manager(self) -> ProviderManager:
        """Return the initialized provider manager or raise an error if unavailable."""

        if self.provider_manager is None:
            raise RuntimeError("Provider manager is not initialized.")

        return self.provider_manager

    def _require_provider_facade(self) -> ProviderService:
        """Return the provider service facade once it has been initialised."""

        if self.provider_facade is None:
            if self.provider_manager is None:
                raise RuntimeError("Provider facade is not initialized.")

            self.provider_facade = ProviderService(
                provider_manager=self.provider_manager,
                config_manager=getattr(self, "config_manager", None),
                logger=getattr(self, "logger", None),
                chat_session=getattr(self, "chat_session", None),
                speech_manager=getattr(self, "speech_manager", None),
            )
            if self._pending_provider_change_listeners:
                for listener in self._pending_provider_change_listeners:
                    self.provider_facade.add_provider_change_listener(listener)
                self._pending_provider_change_listeners.clear()

        return self.provider_facade

    async def initialize(self):
        """
        Asynchronously initialize the ATLAS instance.
        """
        self.provider_manager = await ProviderManager.create(self.config_manager)
        user_identifier, _ = self._ensure_user_identity()
        self.persona_manager = PersonaManager(master=self, user=user_identifier, config_manager=self.config_manager)
        self.chat_session = ChatSession(self)
        self.provider_facade = ProviderService(
            provider_manager=self.provider_manager,
            config_manager=self.config_manager,
            logger=self.logger,
            chat_session=self.chat_session,
            speech_manager=self.speech_manager,
        )
        if self._pending_provider_change_listeners:
            for listener in self._pending_provider_change_listeners:
                self.provider_facade.add_provider_change_listener(listener)
            self._pending_provider_change_listeners.clear()
        try:
            self.provider_manager.set_conversation_manager(self.chat_session)
        except AttributeError:
            self.logger.debug("Active provider manager does not support conversation manager injection.")
        else:
            try:
                self.provider_manager.set_current_conversation_id(self.chat_session.conversation_id)
            except AttributeError:
                self.logger.debug("Active provider manager does not expose conversation ID tracking.")

        default_provider = self.config_manager.get_default_provider()
        if self.config_manager.is_default_provider_ready():
            await self.provider_facade.set_current_provider(default_provider)
            self.logger.debug(
                "Default provider set to: %s",
                self.provider_manager.get_current_provider(),
            )
            self.logger.debug(
                "Default model set to: %s",
                self.provider_manager.get_current_model(),
            )
        else:
            warning_message = self.config_manager.get_pending_provider_warnings().get(
                default_provider,
                f"API key for provider '{default_provider}' is not configured.",
            )
            self.logger.warning(
                "Default provider '%s' could not be activated automatically: %s",
                default_provider,
                warning_message,
            )

        self.logger.info("ATLAS initialized successfully.")
        
        # Initialize SpeechManager
        await self.speech_manager.initialize()  # Ensure SpeechManager is initialized
        self.logger.info("SpeechManager initialized successfully.")
        
        # Load TTS setting from configuration
        tts_enabled = self.config_manager.get_tts_enabled()
        self.speech_manager.set_tts_status(tts_enabled)
        self.logger.debug("TTS enabled: %s", tts_enabled)
        
        # Optionally, set default TTS provider if specified in config.yaml
        default_tts_provider = self.config_manager.get_config('DEFAULT_TTS_PROVIDER')
        if default_tts_provider:
            self.speech_manager.set_default_tts_provider(default_tts_provider)
            self.logger.debug("Default TTS provider set to: %s", default_tts_provider)

        try:
            self.tooling_service.initialize_job_scheduling()
        except Exception as exc:  # pragma: no cover - defensive guard for early adoption
            self.logger.warning("Job scheduling initialization failed: %s", exc, exc_info=True)

        self._initialized = True
        self._notify_persona_change_listeners()

    def is_initialized(self) -> bool:
        """
        Check if ATLAS is fully initialized.

        Returns:
            bool: True if ATLAS is initialized, False otherwise.
        """
        return self._initialized

    def _require_persona_manager(self) -> PersonaManager:
        """Return the initialized persona manager or raise an error if unavailable."""

        if self.persona_manager is None:
            raise RuntimeError("Persona manager is not initialized.")

        return self.persona_manager

    def get_persona_names(self) -> List[str]:
        """
        Retrieve persona names from the PersonaManager.

        Returns:
            List[str]: A list of persona names.
        """
        return self._require_persona_manager().persona_names

    def load_persona(self, persona: str):
        """
        Delegate loading a persona to the PersonaManager.

        Args:
            persona (str): The name of the persona to load.
        """
        self.logger.debug("Loading persona: %s", persona)
        manager = self._require_persona_manager()

        manager.updater(persona)
        self.current_persona = manager.current_persona  # Update the current_persona in ATLAS
        self.logger.debug("Current persona set to: %s", self.current_persona)
        self._notify_persona_change_listeners()

    def get_active_persona_name(self) -> str:
        """Return the human-friendly name of the active persona."""

        manager = self._require_persona_manager()
        persona = getattr(manager, "current_persona", None) or {}

        if isinstance(persona, dict):
            name = persona.get("name")
            if name:
                return str(name)

        default_name = getattr(manager, "default_persona_name", None)
        if default_name:
            return str(default_name)

        return "Assistant"

    def get_current_persona_context(self) -> Dict[str, Any]:
        """Expose the active persona context for UI consumers."""

        manager = getattr(self, "persona_manager", None)
        if manager is None:
            return {
                "system_prompt": "",
                "substitutions": {},
                "persona_name": None,
                "allowed_tools": [],
                "capability_tags": [],
            }

        getter = getattr(manager, "get_current_persona_context", None)
        if callable(getter):
            try:
                context = getter()
            except Exception as exc:  # pragma: no cover - defensive logging only
                self.logger.error("Persona manager context lookup failed: %s", exc, exc_info=True)
            else:
                if isinstance(context, dict):
                    return context

        try:
            prompt = self.get_current_persona_prompt()
        except Exception:  # pragma: no cover - fallback when persona manager unavailable
            prompt = ""

        persona = getattr(manager, "current_persona", None)
        persona_name = None
        if isinstance(persona, dict):
            raw_name = persona.get("name")
            if raw_name:
                persona_name = str(raw_name)

        return {
            "system_prompt": prompt or "",
            "substitutions": {},
            "persona_name": persona_name,
            "allowed_tools": [],
            "capability_tags": [],
        }

    def get_current_persona_tools(self, *, refresh: bool = False) -> Dict[str, Any]:
        """Return a snapshot of the tools declared for the active persona."""

        persona = getattr(self, "current_persona", None)
        if not persona:
            manager = getattr(self, "persona_manager", None)
            persona = getattr(manager, "current_persona", None)

        if not isinstance(persona, dict):
            return {"function_map": {}, "function_payloads": None}

        conversation_manager = getattr(self, "chat_session", None)
        conversation_id = None
        if conversation_manager is not None:
            id_getter = getattr(conversation_manager, "get_conversation_id", None)
            if callable(id_getter):
                try:
                    conversation_id = id_getter()
                except Exception:  # pragma: no cover - defensive logging only
                    conversation_id = getattr(conversation_manager, "conversation_id", None)
            else:
                conversation_id = getattr(conversation_manager, "conversation_id", None)

        try:
            function_map = ToolManagerModule.load_function_map_from_current_persona(
                persona,
                refresh=refresh,
                config_manager=self.config_manager,
            ) or {}
        except Exception as exc:
            self.logger.error("Failed to load persona function map: %s", exc, exc_info=True)
            function_map = {}

        try:
            functions_payload = ToolManagerModule.load_functions_from_json(
                persona,
                refresh=refresh,
                config_manager=self.config_manager,
            )
        except Exception as exc:
            self.logger.error("Failed to load persona function payloads: %s", exc, exc_info=True)
            functions_payload = None

        policy_snapshot = ToolManagerModule.compute_tool_policy_snapshot(
            function_map,
            current_persona=persona,
            conversation_manager=conversation_manager,
            conversation_id=conversation_id,
        )

        map_snapshot: Dict[str, Any] = {}
        if isinstance(function_map, dict):
            for name, entry in function_map.items():
                callable_obj = ToolManagerModule._resolve_function_callable(entry)
                descriptor = None
                if callable(callable_obj):
                    module = getattr(callable_obj, "__module__", None)
                    qualname = getattr(
                        callable_obj, "__qualname__", getattr(callable_obj, "__name__", None)
                    )
                    if module and qualname:
                        descriptor = f"{module}.{qualname}"
                    elif qualname:
                        descriptor = str(qualname)
                entry_snapshot: Dict[str, Any] = {
                    "callable": descriptor or repr(callable_obj)
                }
                metadata_candidate = None
                if isinstance(entry, dict):
                    metadata_candidate = entry.get("metadata")
                if isinstance(metadata_candidate, Mapping):
                    try:
                        entry_snapshot["metadata"] = json.loads(
                            json.dumps(dict(metadata_candidate), ensure_ascii=False)
                        )
                    except (TypeError, ValueError):
                        entry_snapshot["metadata"] = dict(metadata_candidate)
                decision = policy_snapshot.get(name)
                if decision is not None and not getattr(decision, "allowed", True):
                    entry_snapshot["disabled"] = True
                    reason = getattr(decision, "reason", None)
                    if reason:
                        entry_snapshot["disabled_reason"] = reason
                    else:
                        entry_snapshot["disabled_reason"] = (
                            f"Tool '{name}' is blocked by the current policy."
                        )
                map_snapshot[name] = entry_snapshot

        if isinstance(functions_payload, (dict, list)):
            try:
                payload_snapshot = json.loads(json.dumps(functions_payload, ensure_ascii=False))
            except (TypeError, ValueError):
                payload_snapshot = functions_payload
        else:
            payload_snapshot = functions_payload

        return {"function_map": map_snapshot, "function_payloads": payload_snapshot}

    def get_tool_activity_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Expose a copy of the recent tool invocation log for UI use."""

        try:
            return ToolManagerModule.get_tool_activity_log(limit=limit)
        except Exception as exc:
            self.logger.error("Failed to retrieve tool activity log: %s", exc, exc_info=True)
            return []

    def get_persona_audit_history(
        self,
        persona_name: Optional[str],
        offset: int = 0,
        limit: int = 20,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Return serialized persona audit history entries for UI consumption."""

        persona_filter = None
        if persona_name:
            persona_text = str(persona_name).strip()
            persona_filter = persona_text or None

        try:
            offset_value = int(offset)
        except (TypeError, ValueError):
            offset_value = 0
        if offset_value < 0:
            offset_value = 0

        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = 20
        if limit_value <= 0:
            limit_value = 0

        try:
            entries, total = get_persona_audit_logger().get_history(
                persona_name=persona_filter,
                offset=offset_value,
                limit=limit_value,
            )
        except Exception as exc:
            self.logger.error(
                "Failed to retrieve persona audit history: %s", exc, exc_info=True
            )
            return [], 0

        serialised: List[Dict[str, Any]] = []

        for entry in entries:
            if entry is None:
                continue

            try:
                payload = asdict(entry)
            except TypeError:
                payload = {
                    "timestamp": getattr(entry, "timestamp", ""),
                    "persona_name": getattr(entry, "persona_name", ""),
                    "username": getattr(entry, "username", ""),
                    "old_tools": getattr(entry, "old_tools", []),
                    "new_tools": getattr(entry, "new_tools", []),
                    "rationale": getattr(entry, "rationale", ""),
                }

            def _as_tool_list(raw: object) -> List[str]:
                if isinstance(raw, AbcSequence) and not isinstance(raw, (str, bytes)):
                    return [str(item) for item in raw]
                if raw in (None, ""):
                    return []
                return [str(raw)]

            serialised.append(
                {
                    "timestamp": str(payload.get("timestamp") or ""),
                    "persona_name": str(payload.get("persona_name") or ""),
                    "username": str(payload.get("username") or ""),
                    "old_tools": _as_tool_list(payload.get("old_tools")),
                    "new_tools": _as_tool_list(payload.get("new_tools")),
                    "rationale": str(payload.get("rationale") or ""),
                }
            )

        try:
            total_count = int(total)
        except (TypeError, ValueError):
            total_count = len(serialised)

        return serialised, total_count

    def get_current_persona_prompt(self) -> Optional[str]:
        """Return the system prompt for the active persona when available."""

        manager = self._require_persona_manager()
        try:
            return manager.get_current_persona_prompt()
        except AttributeError:
            persona = getattr(manager, "current_persona", None)
            if isinstance(persona, dict):
                return persona.get("system_prompt")
            return None

    def get_persona_editor_state(self, persona_name: str) -> Optional[Dict[str, Any]]:
        """Fetch the structured editor state for the requested persona."""

        return self._require_persona_manager().get_editor_state(persona_name)

    def get_chat_history_snapshot(self) -> List[Dict[str, Any]]:
        """Return a safe copy of the current conversation history."""

        try:
            service = self._get_conversation_service()
        except RuntimeError:
            return []
        return service.get_chat_history_snapshot()

    def get_active_user_roles(self) -> Tuple[str, ...]:
        """Return the configured roles for the active user."""

        return self._resolve_active_user_roles()

    def can_run_conversation_retention(self) -> bool:
        """Return ``True`` when the retention trigger is available to the user."""

        status = self.conversation_retention_status()
        return bool(status.get("available"))

    def conversation_retention_status(self) -> Dict[str, Any]:
        """Summarise whether the retention trigger can be used."""

        try:
            service = self._get_conversation_service()
        except RuntimeError:
            return {"available": False, "reason": "Conversation store unavailable."}

        runner = getattr(self.server, "run_conversation_retention", None)
        available, reason, _ = service.assess_retention_availability(
            runner=runner,
            context=self._build_retention_context(),
        )
        return {"available": available, "reason": reason}

    def run_conversation_retention(self) -> Dict[str, Any]:
        """Trigger conversation retention using the server facade."""

        try:
            service = self._get_conversation_service()
        except RuntimeError:
            return {"success": False, "error": "Conversation store unavailable."}

        runner = getattr(self.server, "run_conversation_retention", None)
        return service.run_conversation_retention(
            runner=runner,
            context=self._build_retention_context(),
        )

    # ------------------------------------------------------------------
    # Conversation repository helpers
    # ------------------------------------------------------------------
    def _get_conversation_service(self) -> ConversationService:
        service = self.conversation_service
        if service is None:
            raise RuntimeError("Conversation service is not initialized.")
        return service

    def add_conversation_history_listener(
        self, listener: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register a callback notified when the conversation list changes."""

        self._get_conversation_service().add_listener(listener)

    def remove_conversation_history_listener(
        self, listener: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Remove a previously registered conversation listener."""

        self._get_conversation_service().remove_listener(listener)

    def notify_conversation_updated(
        self, conversation_id: Any, *, reason: str = "updated"
    ) -> None:
        """Notify UI components that a conversation has changed."""

        self._get_conversation_service().notify_updated(conversation_id, reason=reason)

    def get_recent_conversations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent conversations for the active tenant."""

        try:
            service = self._get_conversation_service()
        except RuntimeError:
            return []
        return service.get_recent_conversations(limit)

    def list_all_conversations(self, *, order: str = "desc") -> List[Dict[str, Any]]:
        """Return all conversations for the tenant in the requested order."""

        try:
            service = self._get_conversation_service()
        except RuntimeError:
            return []
        return service.list_all_conversations(order=order)

    def search_conversations(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Search stored conversations using the server route when available."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")

        request_payload = dict(payload)

        server_method = getattr(self.server, "search_conversations", None)
        if callable(server_method):
            context = {"tenant_id": self.tenant_id or "default"}
            try:
                response = server_method(request_payload, context=context)
            except Exception as exc:  # pragma: no cover - logged for diagnostics
                if hasattr(self.logger, "warning"):
                    self.logger.warning(
                        "Server conversation search failed, falling back to repository: %s",
                        exc,
                        exc_info=True,
                    )
            else:
                if isinstance(response, Mapping):
                    return dict(response)

        try:
            service = self._get_conversation_service()
        except RuntimeError:
            return {"count": 0, "items": []}

        allowed_keys = {
            "text",
            "metadata",
            "vector",
            "conversation_ids",
            "limit",
            "offset",
            "order",
            "top_k",
        }
        filtered = {key: request_payload[key] for key in allowed_keys if key in request_payload}
        return service.search_conversations(**filtered)

    def get_conversation_messages(
        self,
        conversation_id: Any,
        *,
        limit: Optional[int] = None,
        include_deleted: bool = True,
        batch_size: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return messages for ``conversation_id`` using the conversation store."""

        try:
            service = self._get_conversation_service()
        except RuntimeError:
            return []
        return service.get_conversation_messages(
            conversation_id,
            limit=limit,
            include_deleted=include_deleted,
            batch_size=batch_size,
        )

    def reset_conversation_messages(self, conversation_id: Any) -> Dict[str, Any]:
        """Remove stored messages for ``conversation_id`` while preserving the record."""

        try:
            service = self._get_conversation_service()
        except RuntimeError:
            return {"success": False, "error": "Conversation store unavailable."}
        return service.reset_conversation_messages(conversation_id)

    def delete_conversation(self, conversation_id: Any) -> Dict[str, Any]:
        """Permanently delete ``conversation_id`` and associated assets."""

        try:
            service = self._get_conversation_service()
        except RuntimeError:
            return {"success": False, "error": "Conversation store unavailable."}
        return service.delete_conversation(conversation_id)

    def get_negotiation_log(self) -> List[Dict[str, Any]]:
        """Expose recorded negotiation traces to the UI layer."""

        try:
            service = self._get_conversation_service()
        except RuntimeError:
            return []
        return service.get_negotiation_log()

    def _build_retention_context(self) -> Dict[str, Any]:
        tenant_id = self.tenant_id or "default"
        context: Dict[str, Any] = {"tenant_id": tenant_id}

        roles = list(self._resolve_active_user_roles())
        if roles:
            context["roles"] = tuple(roles)

        try:
            username, display_name = self._ensure_user_identity()
        except Exception:  # pragma: no cover - defensive fallback
            username = None
            display_name = None

        if username:
            context["user_id"] = username

        metadata: Dict[str, Any] = {}
        if display_name:
            metadata["user_display_name"] = display_name
        if metadata:
            context["metadata"] = metadata

        return context

    def _resolve_active_user_roles(self) -> Tuple[str, ...]:
        manager = self.config_manager
        roles: List[str] = []

        for source in (getattr(manager, "env_config", None), getattr(manager, "config", None)):
            if not isinstance(source, Mapping):
                continue
            for key in ("ATLAS_ACTIVE_USER_ROLES", "ACTIVE_USER_ROLES", "UI_ACTIVE_ROLES"):
                raw_value = source.get(key)
                for role in self._sanitize_role_tokens(raw_value):
                    if role not in roles:
                        roles.append(role)

        return tuple(roles)

    @staticmethod
    def _sanitize_role_tokens(value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            candidates: Iterable[Any] = value.replace(";", ",").split(",")
        elif isinstance(value, Mapping):
            candidates = value.values()
        elif isinstance(value, Iterable):
            candidates = value
        else:
            return []

        roles: List[str] = []
        for candidate in candidates:
            text = str(candidate).strip()
            if not text:
                continue
            if text not in roles:
                roles.append(text)
        return roles

    def compute_persona_locked_content(
        self,
        persona_name: Optional[str] = None,
        *,
        general: Optional[Dict[str, Any]] = None,
        flags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Compute start/end locked content previews for persona editors."""

        return self._require_persona_manager().compute_locked_content(
            persona_name,
            general=general,
            flags=flags,
        )

    def set_persona_flag(
        self,
        persona_name: str,
        flag: str,
        enabled: Any,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Toggle persona flags using the persona manager facade."""

        return self._require_persona_manager().set_flag(
            persona_name,
            flag,
            enabled,
            extras,
        )

    def update_persona_from_editor(
        self,
        persona_name: str,
        general: Optional[Dict[str, Any]] = None,
        persona_type: Optional[Dict[str, Any]] = None,
        provider: Optional[Dict[str, Any]] = None,
        speech: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Persist persona updates captured from the editor views."""

        return self._require_persona_manager().update_persona_from_form(
            persona_name,
            general,
            persona_type,
            provider,
            speech,
            tools,
            skills,
        )

    def show_persona_message(self, role: str, message: str) -> None:
        """Proxy persona messages to the configured dispatcher."""

        self._require_persona_manager().show_message(role, message)

    def export_persona_bundle(self, persona_name: str, *, signing_key: str) -> Dict[str, Any]:
        """Export ``persona_name`` via the shared server routes."""

        response = self.server.export_persona_bundle(
            persona_name,
            signing_key=signing_key,
        )

        if not response.get("success"):
            return response

        bundle = response.get("bundle")
        if isinstance(bundle, str):
            try:
                response["bundle_bytes"] = base64.b64decode(bundle.encode("utf-8"))
            except (binascii.Error, ValueError) as exc:
                return {"success": False, "error": f"Failed to decode bundle payload: {exc}"}
        else:
            return {"success": False, "error": "Server response did not include bundle data."}

        return response

    def import_persona_bundle(
        self,
        *,
        bundle_bytes: bytes,
        signing_key: str,
        rationale: str = "Imported via UI",
    ) -> Dict[str, Any]:
        """Import a persona bundle through the server routes."""

        encoded = base64.b64encode(bundle_bytes).decode("ascii")
        return self.server.import_persona_bundle(
            bundle_base64=encoded,
            signing_key=signing_key,
            rationale=rationale,
        )

    def get_persona_metrics(
        self,
        persona_name: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 20,
        metric_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return aggregated persona analytics via the shared server routes."""

        return self.server.get_persona_metrics(
            persona_name,
            start=start,
            end=end,
            limit=limit,
            metric_type=metric_type,
        )

    def get_persona_review_status(self, persona_name: str) -> Dict[str, Any]:
        """Return the review status payload for ``persona_name``."""

        return self.server.get_persona_review_status(persona_name)

    def attest_persona_review(
        self,
        persona_name: str,
        *,
        expires_in_days: Optional[int] = None,
        expires_at: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a review attestation for ``persona_name``."""

        reviewer, _display = self._ensure_user_identity()
        return self.server.attest_persona_review(
            persona_name,
            reviewer=reviewer,
            expires_in_days=expires_in_days,
            expires_at=expires_at,
            notes=notes,
        )

    def _require_chat_session(self) -> ChatSession:
        """Return the initialized chat session or raise an error if unavailable."""

        session = getattr(self, "chat_session", None)
        if session is None:
            raise RuntimeError("Chat session is not initialized.")

        return session

    def reset_chat_history(self) -> Dict[str, Any]:
        """Reset the active chat session and provide a normalized payload for the UI."""

        try:
            session = self._require_chat_session()
        except RuntimeError as exc:
            message = str(exc)
            self.logger.error(message)
            return {
                "success": False,
                "error": message,
            }

        try:
            session.reset_conversation()
        except Exception as exc:  # noqa: BLE001 - propagate details to the UI payload
            message = f"Failed to reset chat history: {exc}"
            self.logger.error(message, exc_info=True)
            return {
                "success": False,
                "error": message,
            }

        payload: Dict[str, Any] = {
            "success": True,
            "message": "Chat cleared and session reset.",
        }

        try:
            payload["status_summary"] = self.get_chat_status_summary()
        except Exception as exc:  # pragma: no cover - defensive logging only
            self.logger.error(
                "Failed to refresh chat status summary after reset: %s",
                exc,
                exc_info=True,
            )

        return payload

    def export_chat_history(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Export the chat history and normalize the outcome for the UI layer."""

        try:
            session = self._require_chat_session()
        except RuntimeError as exc:
            message = str(exc)
            self.logger.error(message)
            return {
                "success": False,
                "error": message,
            }

        try:
            result = session.export_history(path)
        except ChatHistoryExportError as exc:
            message = str(exc) or "Failed to export chat history."
            return {
                "success": False,
                "error": message,
            }
        except Exception as exc:  # noqa: BLE001 - unexpected failure surfaced to UI
            message = f"Unexpected error while exporting chat history: {exc}"
            self.logger.error(message, exc_info=True)
            return {
                "success": False,
                "error": message,
            }

        return {
            "success": True,
            "message": f"Exported {result.message_count} messages to: {result.path}",
            "path": str(result.path),
            "message_count": result.message_count,
        }

    def send_chat_message_async(
        self,
        message: str,
        *,
        on_success: Optional[Callable[[str, str], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        thread_name: Optional[str] = None,
    ) -> Future[str]:
        """Dispatch ``message`` via the chat session on a background worker.

        The provided callbacks are invoked with the active persona's name along
        with the generated response (for successes) or the raised exception (for
        failures).  This keeps persona lookups consolidated within the backend
        layer so UI components only handle widget updates.
        """

        session = self._require_chat_session()

        def _invoke_success(response: str) -> None:
            if on_success is None:
                return
            persona_name = self.get_active_persona_name()
            try:
                on_success(persona_name, response)
            except Exception as callback_exc:  # noqa: BLE001 - log unexpected UI failures.
                self.logger.error(
                    "Chat success callback failed: %s", callback_exc, exc_info=True
                )

        def _invoke_error(exc: Exception) -> None:
            persona_name = self.get_active_persona_name()
            if on_error is None:
                self.logger.error("Chat background task failed: %s", exc, exc_info=True)
                return
            try:
                on_error(persona_name, exc)
            except Exception as callback_exc:  # noqa: BLE001 - log unexpected UI failures.
                self.logger.error(
                    "Chat error callback failed: %s", callback_exc, exc_info=True
                )

        return session.run_in_background(
            lambda: session.send_message(message),
            on_success=_invoke_success if on_success is not None else None,
            on_error=_invoke_error,
            thread_name=thread_name or "ChatResponseWorker",
        )

    def get_available_providers(self) -> List[str]:
        """
        Retrieve all available providers from the ProviderManager.

        Returns:
            List[str]: A list of provider names.
        """
        return self._require_provider_facade().get_available_providers()

    async def test_huggingface_token(self, token: Optional[str] = None) -> Dict[str, Any]:
        """Validate a HuggingFace API token via the provider manager."""

        return await self._require_provider_facade().test_huggingface_token(token)

    def list_hf_models(self) -> Dict[str, Any]:
        """List installed HuggingFace models via the provider manager."""

        return self._require_provider_facade().list_hf_models()

    async def load_hf_model(self, model_name: str, force_download: bool = False) -> Dict[str, Any]:
        """Load a HuggingFace model using the provider manager helper."""

        return await self._require_provider_facade().load_hf_model(
            model_name, force_download=force_download
        )

    async def unload_hf_model(self) -> Dict[str, Any]:
        """Unload the active HuggingFace model via the provider manager."""

        return await self._require_provider_facade().unload_hf_model()

    async def remove_hf_model(self, model_name: str) -> Dict[str, Any]:
        """Remove a cached HuggingFace model through the provider manager."""

        return await self._require_provider_facade().remove_hf_model(model_name)

    async def download_hf_model(self, model_id: str, force: bool = False) -> Dict[str, Any]:
        """Download a HuggingFace model through the provider manager."""

        return await self._require_provider_facade().download_hf_model(
            model_id, force=force
        )

    async def search_hf_models(
        self,
        search_query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search HuggingFace models using the provider manager helper."""

        return await self._require_provider_facade().search_hf_models(
            search_query,
            filters,
            limit=limit,
        )

    def update_hf_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Persist HuggingFace settings via the provider manager."""

        return self._require_provider_facade().update_hf_settings(settings)

    def clear_hf_cache(self) -> Dict[str, Any]:
        """Clear cached HuggingFace artefacts through the provider manager."""

        return self._require_provider_facade().clear_hf_cache()

    def save_hf_token(self, token: Optional[str]) -> Dict[str, Any]:
        """Save a Hugging Face token via the provider manager."""

        return self._require_provider_facade().save_hf_token(token)

    def get_provider_api_key_status(self, provider_name: str) -> Dict[str, Any]:
        """Fetch credential metadata for a provider using the provider manager."""

        return self._require_provider_facade().get_provider_api_key_status(provider_name)

    async def update_provider_api_key(
        self, provider_name: str, new_api_key: Optional[str]
    ) -> Dict[str, Any]:
        """Persist a provider API key through the provider manager facade."""

        return await self._require_provider_facade().update_provider_api_key(
            provider_name, new_api_key
        )

    def ensure_huggingface_ready(self) -> Dict[str, Any]:
        """Ensure the HuggingFace helper is initialized via the provider manager."""

        return self._require_provider_facade().ensure_huggingface_ready()

    def run_in_background(
        self,
        coroutine_factory: Callable[[], Awaitable[Any]],
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        thread_name: Optional[str] = None,
    ) -> Future:
        """Execute an awaitable in a background thread using the shared task helper."""

        if self.provider_facade is not None:
            return self.provider_facade.run_in_background(
                coroutine_factory,
                on_success=on_success,
                on_error=on_error,
                thread_name=thread_name,
            )

        return run_async_in_thread(
            coroutine_factory,
            on_success=on_success,
            on_error=on_error,
            logger=self.logger,
            thread_name=thread_name,
        )

    def run_provider_manager_task(
        self,
        coroutine_factory: Callable[[], Awaitable[Any]],
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        thread_name: Optional[str] = None,
    ) -> Future:
        """Schedule a provider-manager coroutine using the shared background runner."""

        return self._require_provider_facade().run_provider_manager_task(
            coroutine_factory,
            on_success=on_success,
            on_error=on_error,
            thread_name=thread_name,
        )

    def set_current_provider_in_background(
        self,
        provider: str,
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> Future:
        """Schedule a provider switch without blocking the caller."""

        return self._require_provider_facade().set_current_provider_in_background(
            provider,
            on_success=on_success,
            on_error=on_error,
        )

    def update_provider_api_key_in_background(
        self,
        provider_name: str,
        new_api_key: Optional[str],
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> Future:
        """Persist provider credentials using a background worker thread."""

        return self._require_provider_facade().update_provider_api_key_in_background(
            provider_name,
            new_api_key,
            on_success=on_success,
            on_error=on_error,
        )

    async def refresh_current_provider(
        self, provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reload the active provider configuration when the names align."""

        return await self._require_provider_facade().refresh_current_provider(
            provider_name
        )

    async def set_current_provider(self, provider: str):
        """
        Asynchronously set the current provider in the ProviderManager.
        """
        await self._require_provider_facade().set_current_provider(provider)

    def add_provider_change_listener(self, listener: Callable[[Dict[str, str]], None]) -> None:
        """Register a callback to be notified when the provider or model changes."""

        if self.provider_facade is not None:
            self.provider_facade.add_provider_change_listener(listener)
        else:
            if not callable(listener):
                raise TypeError("listener must be callable")
            if listener in self._pending_provider_change_listeners:
                return
            self._pending_provider_change_listeners.append(listener)

    def remove_provider_change_listener(self, listener: Callable[[Dict[str, str]], None]) -> None:
        """Remove a previously registered provider change callback if present."""

        if self.provider_facade is not None:
            self.provider_facade.remove_provider_change_listener(listener)
        elif listener in self._pending_provider_change_listeners:
            self._pending_provider_change_listeners.remove(listener)

    def _notify_provider_change_listeners(self) -> None:
        """Invoke all registered provider change callbacks."""

        if self.provider_facade is not None:
            self.provider_facade.notify_provider_change_listeners()

    def add_persona_change_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Register callbacks that react to persona prompt/context updates."""

        if not callable(listener):
            raise TypeError("listener must be callable")

        if listener in self._persona_change_listeners:
            return

        self._persona_change_listeners.append(listener)

        try:
            listener(self.get_current_persona_context())
        except Exception:  # pragma: no cover - listener failures are logged elsewhere
            self.logger.debug("Persona change listener raised during registration", exc_info=True)

    def remove_persona_change_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a previously registered persona change listener."""

        try:
            self._persona_change_listeners.remove(listener)
        except ValueError:
            pass

    def _notify_persona_change_listeners(self) -> None:
        """Notify listeners that persona context has changed."""

        if not self._persona_change_listeners:
            return

        try:
            context = self.get_current_persona_context()
        except Exception as exc:  # pragma: no cover - defensive logging only
            self.logger.error("Failed to assemble persona context snapshot: %s", exc, exc_info=True)
            context = {"system_prompt": "", "substitutions": {}, "persona_name": None}

        snapshot = dict(context)
        for listener in list(self._persona_change_listeners):
            try:
                listener(snapshot)
            except Exception as exc:
                self.logger.error(
                    "Persona change listener %s failed: %s", listener, exc, exc_info=True
                )

    def register_message_dispatcher(self, dispatcher: Callable[[str, str], None]) -> None:
        """Register a callback that handles persona-related messages from the backend."""
        if not callable(dispatcher):
            raise TypeError("dispatcher must be callable")

        if dispatcher in self._message_dispatchers:
            return

        self._message_dispatchers.append(dispatcher)
        self._refresh_message_dispatcher()

    def unregister_message_dispatcher(self, dispatcher: Callable[[str, str], None]) -> None:
        """Remove a previously registered persona message callback."""
        if dispatcher in self._message_dispatchers:
            self._message_dispatchers.remove(dispatcher)
            self._refresh_message_dispatcher()

    def _refresh_message_dispatcher(self) -> None:
        """Update the aggregated dispatcher exposed to backend components."""
        if not self._message_dispatchers:
            self.message_dispatcher = None
            return

        def aggregated(role: str, message: str) -> None:
            for callback in list(self._message_dispatchers):
                try:
                    callback(role, message)
                except Exception as exc:
                    self.logger.error(
                        "Message dispatcher %s failed: %s", callback, exc, exc_info=True
                    )

        self.message_dispatcher = aggregated

    def log_history(self):
        """
        Handle history-related functionality.
        """
        self.logger.debug("History button clicked")
        print("History button clicked")

    def show_settings(self):
        """
        Handle settings-related functionality.
        """
        self.logger.debug("Settings page clicked")
        print("Settings page clicked")

    def get_default_provider(self) -> str:
        """
        Get the default provider from the ProviderManager.

        Returns:
            str: The name of the default provider.
        """
        if self.provider_facade is not None:
            provider = self.provider_facade.get_default_provider()
        elif self.provider_manager is not None:
            provider = self.provider_manager.get_current_provider()
        else:
            provider = None
        return provider or ""

    def get_default_model(self) -> str:
        """
        Get the default model from the ProviderManager.

        Returns:
            str: The name of the default model.
        """
        if self.provider_facade is not None:
            model = self.provider_facade.get_default_model()
        elif self.provider_manager is not None:
            model = self.provider_manager.get_current_model()
        else:
            model = None
        return model or ""

    def get_openai_llm_settings(self) -> Dict[str, Any]:
        """Expose the persisted OpenAI LLM defaults via the provider manager."""

        return self._require_provider_facade().get_openai_llm_settings()

    def get_google_llm_settings(self) -> Dict[str, Any]:
        """Expose the persisted Google Gemini defaults via the provider manager."""

        return self._require_provider_facade().get_google_llm_settings()

    def get_anthropic_settings(self) -> Dict[str, Any]:
        """Return Anthropic defaults via the provider manager facade."""

        return self._require_provider_facade().get_anthropic_settings()

    async def list_openai_models(
        self,
        *,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch available OpenAI models through the provider manager facade."""

        return await self._require_provider_facade().list_openai_models(
            base_url=base_url,
            organization=organization,
        )

    async def list_anthropic_models(
        self,
        *,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch available Anthropic models through the provider manager facade."""

        return await self._require_provider_facade().list_anthropic_models(
            base_url=base_url,
        )

    async def list_google_models(
        self,
        *,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch available Google Gemini models through the provider manager facade."""

        return await self._require_provider_facade().list_google_models(
            base_url=base_url,
        )

    def set_openai_llm_settings(
        self,
        *,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        function_calling: Optional[bool] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        json_mode: Optional[Any] = None,
        json_schema: Optional[Any] = None,
        audio_enabled: Optional[bool] = None,
        audio_voice: Optional[str] = None,
        audio_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist OpenAI defaults through the provider manager facade."""

        return self._require_provider_facade().set_openai_llm_settings(
            model=model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            stream=stream,
            function_calling=function_calling,
            base_url=base_url,
            organization=organization,
            reasoning_effort=reasoning_effort,
            json_mode=json_mode,
            json_schema=json_schema,
            audio_enabled=audio_enabled,
            audio_voice=audio_voice,
            audio_format=audio_format,
        )

    def set_google_llm_settings(
        self,
        *,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[Any] = None,
        candidate_count: Optional[int] = None,
        max_output_tokens: Optional[Any] = None,
        stop_sequences: Optional[Any] = None,
        safety_settings: Optional[Any] = None,
        response_mime_type: Optional[str] = None,
        system_instruction: Optional[str] = None,
        stream: Optional[bool] = None,
        function_calling: Optional[bool] = None,
        function_call_mode: Optional[str] = None,
        allowed_function_names: Optional[Any] = None,
        response_schema: Optional[Any] = None,
        cached_allowed_function_names: Optional[Any] = None,
        seed: Optional[Any] = None,
        response_logprobs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Persist Google defaults through the provider manager facade."""

        return self._require_provider_facade().set_google_llm_settings(
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            candidate_count=candidate_count,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences,
            safety_settings=safety_settings,
            response_mime_type=response_mime_type,
            system_instruction=system_instruction,
            stream=stream,
            function_calling=function_calling,
            function_call_mode=function_call_mode,
            allowed_function_names=allowed_function_names,
            response_schema=response_schema,
            cached_allowed_function_names=cached_allowed_function_names,
            seed=seed,
            response_logprobs=response_logprobs,
        )

    def set_anthropic_settings(
        self,
        *,
        model: str,
        stream: bool,
        function_calling: bool,
        temperature: float,
        top_p: float,
        top_k: Any = ConfigManager.UNSET,
        max_output_tokens: Optional[int],
        timeout: int,
        max_retries: int,
        retry_delay: int,
        stop_sequences: Any = ConfigManager.UNSET,
    ) -> Dict[str, Any]:
        """Persist Anthropic defaults through the provider manager facade."""

        return self._require_provider_facade().set_anthropic_settings(
            model=model,
            stream=stream,
            function_calling=function_calling,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            stop_sequences=stop_sequences,
        )

    def get_models_for_provider(self, provider: str) -> List[str]:
        """Return cached model names for the requested provider."""

        return self._require_provider_facade().get_models_for_provider(provider)

    def get_chat_status_summary(self) -> Dict[str, str]:
        """Return a consolidated snapshot of chat-related status information."""

        return self._require_provider_facade().get_chat_status_summary()

    def format_chat_status(self, status_summary: Optional[Dict[str, str]] = None) -> str:
        """Generate the human-readable chat status message for display."""

        return self._require_provider_facade().format_chat_status(status_summary)

    def get_speech_defaults(self) -> Dict[str, Any]:
        """Expose global speech defaults for UI consumers."""

        return self._require_speech_facade().get_speech_defaults()

    def get_speech_provider_status(self, provider_name: str) -> Dict[str, Any]:
        """Return credential metadata for a speech provider."""

        return self._require_speech_facade().get_speech_provider_status(provider_name)

    def get_speech_voice_options(
        self, provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return the available voice options for a speech provider."""

        return self._require_speech_facade().get_speech_voice_options(provider)

    def get_active_speech_voice(self) -> Dict[str, Optional[str]]:
        """Return the active speech provider and voice name."""

        return self._require_speech_facade().get_active_speech_voice()

    def update_speech_defaults(
        self,
        *,
        tts_enabled: bool,
        tts_provider: Optional[str],
        stt_enabled: bool,
        stt_provider: Optional[str],
    ) -> Dict[str, Any]:
        """Persist global speech defaults via the speech manager."""

        return self._require_speech_facade().update_speech_defaults(
            tts_enabled=tts_enabled,
            tts_provider=tts_provider,
            stt_enabled=stt_enabled,
            stt_provider=stt_provider,
        )

    def update_elevenlabs_settings(
        self,
        *,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update ElevenLabs credentials or active voice selection."""

        return self._require_speech_facade().update_elevenlabs_settings(
            api_key=api_key,
            voice_id=voice_id,
        )

    def update_google_speech_settings(
        self,
        credentials_path: str,
        *,
        tts_voice: Optional[str] = None,
        stt_language: Optional[str] = None,
        auto_punctuation: Optional[bool] = None,
    ) -> None:
        """Persist Google speech credentials and preferences via the speech manager."""

        self._require_speech_facade().update_google_speech_settings(
            credentials_path,
            tts_voice=tts_voice,
            stt_language=stt_language,
            auto_punctuation=auto_punctuation,
        )

    def get_google_speech_credentials_path(self) -> Optional[str]:
        """Return the persisted Google speech credentials path."""

        return self._require_speech_facade().get_google_speech_credentials_path()

    def get_google_speech_settings(self) -> Dict[str, Any]:
        """Expose persisted Google speech configuration for UI rendering."""

        return self._require_speech_facade().get_google_speech_settings()

    def get_openai_speech_options(self) -> Dict[str, List[Tuple[str, Optional[str]]]]:
        """Return the OpenAI speech option sets for UI rendering."""

        return self._require_speech_facade().get_openai_speech_options()

    def get_openai_speech_configuration(self) -> Dict[str, Optional[str]]:
        """Return persisted OpenAI speech configuration values."""

        return self._require_speech_facade().get_openai_speech_configuration()

    def update_openai_speech_settings(
        self, display_payload: Dict[str, Any]
    ) -> Dict[str, Optional[str]]:
        """Validate and persist OpenAI speech settings supplied by the UI."""

        return self._require_speech_facade().update_openai_speech_settings(
            display_payload
        )

    def get_transcription_history(
        self, *, formatted: bool = False
    ) -> List[Dict[str, Any]]:
        """Return transcription history records from the speech manager."""

        return self._require_speech_facade().get_transcription_history(
            formatted=formatted
        )

    def get_transcript_export_preferences(self) -> Dict[str, Any]:
        """Expose configured transcript export preferences for UI rendering."""

        return self._require_speech_facade().get_transcript_export_preferences()

    def update_transcript_export_preferences(
        self,
        *,
        formats: Iterable[str] | None = None,
        directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist transcript export preferences supplied by the UI."""

        return self._require_speech_facade().update_transcript_export_preferences(
            formats=formats,
            directory=directory,
        )

    class _LegacySubscriptionHandle:
        """Wrapper for legacy event system subscriptions."""

        __slots__ = ("event_name", "callback")

        def __init__(self, event_name: str, callback: Callable[..., Any]) -> None:
            self.event_name = event_name
            self.callback = callback

        def cancel(self) -> None:
            _legacy_event_system.unsubscribe(self.event_name, self.callback)

    def subscribe_event(
        self,
        event_name: str,
        callback: Callable[..., Any],
        *,
        include_message: bool = False,
        retry_attempts: int = 3,
        retry_delay: float = 0.1,
        concurrency: int = 1,
        legacy_only: bool = False,
    ) -> Any:
        """Subscribe *callback* to backend events via the facade.

        When ``legacy_only`` is ``True`` the subscription is limited to the
        in-process legacy event system. Otherwise the combined message bus and
        legacy subscription from :func:`modules.Tools.tool_event_system.subscribe_bus_event`
        is used.
        """

        if legacy_only:
            _legacy_event_system.subscribe(event_name, callback)
            return self._LegacySubscriptionHandle(event_name, callback)

        return _subscribe_bus_event(
            event_name,
            callback,
            include_message=include_message,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            concurrency=concurrency,
        )

    def unsubscribe_event(
        self,
        handle: Any,
        *,
        event_name: Optional[str] = None,
        callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Cancel a subscription obtained via :meth:`subscribe_event`."""

        if handle is None:
            return

        try:
            if isinstance(handle, DualSubscription):
                handle.cancel()
                return
            if isinstance(handle, self._LegacySubscriptionHandle):
                handle.cancel()
                return
            cancel = getattr(handle, "cancel", None)
            if callable(cancel):
                cancel()
                return
        except Exception as exc:  # pragma: no cover - defensive cancellation
            self.logger.debug("Event unsubscribe failed: %s", exc, exc_info=True)
            return

        if event_name and callback:
            try:
                _legacy_event_system.unsubscribe(event_name, callback)
            except Exception as exc:  # pragma: no cover - defensive cancellation
                self.logger.debug("Legacy unsubscribe failed: %s", exc, exc_info=True)

    def create_blackboard_entry(
        self,
        scope_id: str,
        *,
        scope_type: str = "conversation",
        category: str,
        title: str,
        content: str,
        author: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a new blackboard entry for the active tenant."""

        scope_token = str(scope_id or "").strip()
        if not scope_token:
            raise ValueError("scope_id is required")

        category_token = str(category or "").strip()
        if not category_token:
            raise ValueError("category is required")

        title_text = str(title or "").strip()
        content_text = str(content or "").strip()
        if not title_text or not content_text:
            raise ValueError("title and content are required")

        payload: Dict[str, Any] = {
            "category": category_token,
            "title": title_text,
            "content": content_text,
        }

        if author is not None:
            author_text = str(author).strip()
            if author_text:
                payload["author"] = author_text

        if tags is not None:
            if isinstance(tags, (str, bytes, bytearray)):
                raise TypeError("tags must be an iterable of strings")
            normalized_tags = [
                str(tag).strip()
                for tag in tags
                if str(tag or "").strip()
            ]
            payload["tags"] = normalized_tags

        metadata_payload: Dict[str, Any] | None
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata must be a mapping")
            metadata_payload = dict(metadata)
        else:
            metadata_payload = {}

        tenant_id = self.tenant_id or "default"
        if metadata_payload is not None:
            metadata_payload.setdefault("tenant_id", tenant_id)
            if metadata_payload:
                payload["metadata"] = metadata_payload

        server = self._require_server()
        context = {"tenant_id": tenant_id}
        return server.create_blackboard_entry(
            str(scope_type or "conversation"),
            scope_token,
            payload,
            context=context,
        )

    def update_blackboard_entry(
        self,
        scope_id: str,
        entry_id: str,
        *,
        scope_type: str = "conversation",
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update a previously created blackboard entry."""

        scope_token = str(scope_id or "").strip()
        if not scope_token:
            raise ValueError("scope_id is required")

        entry_token = str(entry_id or "").strip()
        if not entry_token:
            raise ValueError("entry_id is required")

        payload: Dict[str, Any] = {}
        if title is not None:
            title_text = str(title or "").strip()
            if not title_text:
                raise ValueError("title must not be empty")
            payload["title"] = title_text

        if content is not None:
            content_text = str(content or "").strip()
            if not content_text:
                raise ValueError("content must not be empty")
            payload["content"] = content_text

        if tags is not None:
            if isinstance(tags, (str, bytes, bytearray)):
                raise TypeError("tags must be an iterable of strings")
            payload["tags"] = [
                str(tag).strip()
                for tag in tags
                if str(tag or "").strip()
            ]

        metadata_payload: Optional[Dict[str, Any]] = None
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata must be a mapping")
            metadata_payload = dict(metadata)

        tenant_id = self.tenant_id or "default"
        if metadata_payload is not None:
            metadata_payload.setdefault("tenant_id", tenant_id)
            payload["metadata"] = metadata_payload

        if not payload:
            raise ValueError("At least one field must be provided for update")

        server = self._require_server()
        context = {"tenant_id": tenant_id}
        return server.update_blackboard_entry(
            str(scope_type or "conversation"),
            scope_token,
            entry_token,
            payload,
            context=context,
        )

    def delete_blackboard_entry(
        self,
        scope_id: str,
        entry_id: str,
        *,
        scope_type: str = "conversation",
    ) -> Dict[str, Any]:
        """Remove a blackboard entry for the active tenant."""

        scope_token = str(scope_id or "").strip()
        if not scope_token:
            raise ValueError("scope_id is required")

        entry_token = str(entry_id or "").strip()
        if not entry_token:
            raise ValueError("entry_id is required")

        server = self._require_server()
        context = {"tenant_id": self.tenant_id or "default"}
        return server.delete_blackboard_entry(
            str(scope_type or "conversation"),
            scope_token,
            entry_token,
            context=context,
        )

    def get_blackboard_summary(
        self,
        scope_id: str,
        *,
        scope_type: str = "conversation",
    ) -> Dict[str, Any]:
        """Return the aggregated blackboard summary for the requested scope."""

        store = _get_blackboard()
        client = store.client_for(scope_id, scope_type=scope_type)
        return client.summary()

    async def stream_blackboard_summary(
        self,
        scope_id: str,
        *,
        scope_type: str = "conversation",
    ) -> TypingAsyncIterator[Dict[str, Any]]:
        """Yield up-to-date blackboard summaries as events are received."""

        async for _event in _stream_blackboard(scope_id, scope_type=scope_type):
            try:
                yield self.get_blackboard_summary(scope_id, scope_type=scope_type)
            except Exception as exc:  # pragma: no cover - defensive logging only
                self.logger.debug(
                    "Failed to refresh blackboard summary during stream: %s",
                    exc,
                    exc_info=True,
                )


    async def close(self):
        """
        Perform cleanup operations.
        """
        await self.provider_manager.close()
        await self.speech_manager.close()
        if self._user_account_service is not None:
            try:
                self._user_account_service.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        self.tooling_service.shutdown_jobs()
        self.logger.info("ATLAS closed and all providers unloaded.")

    async def maybe_text_to_speech(self, response_text: Any) -> None:
        """Run text-to-speech for the provided response when enabled.

        Args:
            response_text: The response payload to vocalize. May be a plain
                string or a mapping containing ``text``/``audio`` fields.
        """

        try:
            await self._require_speech_facade().maybe_text_to_speech(response_text)
        except RuntimeError:
            self.logger.debug("Speech facade unavailable; skipping TTS synthesis.")
            return

    def start_stt_listening(self) -> Dict[str, Any]:
        """Begin speech-to-text recording via the active provider.

        Returns:
            Dict[str, Any]: Structured payload describing the resulting state.
        """
        return self._require_speech_facade().start_stt_listening()

    def stop_stt_and_transcribe(
        self,
        *,
        export_formats: Optional[Iterable[str]] = None,
        export_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stop recording (if active) and transcribe in the background.

        Returns:
            Dict[str, Any]: Structured payload with an attached transcription future.
        """

        return self._require_speech_facade().stop_stt_and_transcribe(
            export_formats=export_formats,
            export_directory=export_directory,
        )

    async def generate_response(
        self, messages: List[Dict[str, str]]
    ) -> Union[str, TypingAsyncIterator[str]]:
        """
        Generate a response using the current provider and model.
        Additionally, perform TTS generation if enabled.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.

        Returns:
            Union[str, TypingAsyncIterator[str]]: The generated response or a stream of tokens.
        """
        if not self.current_persona:
            self.logger.error("No persona is currently loaded. Cannot generate response.")
            return "Error: No persona is currently loaded. Please select a persona."

        try:
            conversation_id = self.chat_session.conversation_id
            try:
                self.provider_manager.set_current_conversation_id(conversation_id)
            except AttributeError:
                self.logger.debug("Provider manager missing set_current_conversation_id; continuing.")
            response = await self.provider_manager.generate_response(
                messages=messages,
                current_persona=self.current_persona,
                user=self._ensure_user_identity()[0],
                conversation_id=conversation_id,
                conversation_manager=self.chat_session,
            )

            # Perform TTS if enabled
            try:
                await self.maybe_text_to_speech(response)
            except Exception as tts_exc:
                self.logger.error(
                    "Optional text-to-speech failed: %s", tts_exc, exc_info=True
                )

            return response
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}", exc_info=True)
            return "Error: Failed to generate response. Please try again later."
