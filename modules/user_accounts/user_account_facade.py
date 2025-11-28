"""High-level orchestration for user account operations within ATLAS."""

from __future__ import annotations

import getpass
from typing import Callable, Dict, List, Optional, Tuple

from ATLAS.config import ConfigManager
from modules.background_tasks import run_async_in_thread
from modules.conversation_store import ConversationStoreRepository
from modules.user_accounts.user_account_service import PasswordRequirements


class UserAccountFacade:
    """Coordinate account lookups and lifecycle events for ATLAS."""

    def __init__(
        self,
        *,
        config_manager: ConfigManager,
        conversation_repository: ConversationStoreRepository | None,
        logger,
    ) -> None:
        self._config_manager = config_manager
        self._conversation_repository = conversation_repository
        self._logger = logger
        self._user_account_service = None
        self._user_identifier: Optional[str] = None
        self._user_display_name: Optional[str] = None
        self._active_user_change_listeners: List[Callable[[str, str], None]] = []

    # ------------------------------------------------------------------
    # Conversation repository lifecycle
    # ------------------------------------------------------------------
    def update_conversation_repository(
        self, repository: ConversationStoreRepository | None
    ) -> None:
        """Update the repository used for profile lookups."""

        self._conversation_repository = repository

    # ------------------------------------------------------------------
    # Identity resolution helpers
    # ------------------------------------------------------------------
    def resolve_user_identity(
        self, *, prefer_generic: bool = False
    ) -> Tuple[str, str]:
        """Return best-effort user identifier and display name."""

        username: Optional[str] = None
        display_name: Optional[str] = None

        try:
            account_service = self._get_user_account_service()
        except Exception as exc:  # pragma: no cover - optional dependency issues
            self._logger.debug("User account service unavailable: %s", exc)
        else:
            try:
                preferred = account_service.get_active_user()
                users = account_service.list_users()
                account_record = None

                if preferred:
                    account_record = next(
                        (record for record in users if record.get("username") == preferred),
                        None,
                    )

                if account_record is None and users and not prefer_generic:
                    account_record = users[0]

                if account_record:
                    username = account_record.get("username") or username
                    recorded_name = account_record.get("name")
                    if recorded_name:
                        display_name = str(recorded_name)
            except Exception as exc:  # pragma: no cover - defensive logging only
                self._logger.debug("User account lookup failed: %s", exc)

        repository = self._conversation_repository
        if repository is not None:
            try:
                if username:
                    record = repository.get_user_profile(username)
                    if record:
                        stored_username = record.get("username")
                        if stored_username:
                            username = stored_username
                        profile_data = record.get("profile") or {}
                        candidate_name = (
                            record.get("display_name")
                            or profile_data.get("Full Name")
                            or profile_data.get("full_name")
                            or profile_data.get("Display Name")
                            or profile_data.get("display_name")
                            or profile_data.get("name")
                            or profile_data.get("Username")
                        )
                        if candidate_name:
                            display_name = str(candidate_name)
                elif not prefer_generic:
                    profiles = repository.list_user_profiles()
                    for record in profiles:
                        profile_data = record.get("profile") or {}
                        stored_username = record.get("username")
                        candidate_name = (
                            record.get("display_name")
                            or profile_data.get("Full Name")
                            or profile_data.get("full_name")
                            or profile_data.get("Display Name")
                            or profile_data.get("display_name")
                            or profile_data.get("name")
                            or profile_data.get("Username")
                        )
                        if stored_username and not username:
                            username = stored_username
                        if candidate_name:
                            display_name = str(candidate_name)
                            break
            except Exception as exc:  # pragma: no cover - repository lookup issues
                self._logger.debug("User profile lookup failed: %s", exc)

        if not username:
            try:
                username = getpass.getuser()
            except Exception:  # pragma: no cover - system specific failures
                username = None

        if not username:
            username = "User"

        if not display_name:
            display_name = username

        return username, display_name

    def refresh_active_user_identity(
        self, *, prefer_generic: bool = False
    ) -> Tuple[str, str]:
        """Refresh cached user metadata when the active account changes."""

        previous_username = self._user_identifier
        previous_display = self._user_display_name

        username, display_name = self.resolve_user_identity(
            prefer_generic=prefer_generic
        )
        self._user_identifier = username
        self._user_display_name = display_name

        if username != previous_username or display_name != previous_display:
            self._dispatch_active_user_change(username, display_name)

        return username, display_name

    def ensure_user_identity(self) -> Tuple[str, str]:
        """Ensure user identifier and display name are loaded."""

        if not self._user_identifier or not self._user_display_name:
            self.refresh_active_user_identity()

        return self._user_identifier, self._user_display_name

    def get_user_display_name(self) -> str:
        """Return the friendly name for the active user."""

        _, display_name = self.ensure_user_identity()
        return display_name

    def override_user_identity(self, value: str) -> None:
        """Manually override the cached identity (legacy behaviour)."""

        sanitized = (value or "").strip()
        if not sanitized:
            sanitized = "User"
        self._user_identifier = sanitized
        self._user_display_name = sanitized

    # ------------------------------------------------------------------
    # Listener management
    # ------------------------------------------------------------------
    def add_active_user_change_listener(
        self, listener: Callable[[str, str], None]
    ) -> None:
        """Register a callback notified whenever the active user changes."""

        if not callable(listener):
            raise TypeError("Listener must be callable")

        if listener not in self._active_user_change_listeners:
            self._active_user_change_listeners.append(listener)

        username, display_name = self.ensure_user_identity()
        try:
            listener(username, display_name)
        except Exception:  # pragma: no cover - listener exceptions are logged elsewhere
            self._logger.debug(
                "Active user listener raised during registration", exc_info=True
            )

    def remove_active_user_change_listener(
        self, listener: Callable[[str, str], None]
    ) -> None:
        """Remove a previously registered active user listener."""

        try:
            self._active_user_change_listeners.remove(listener)
        except ValueError:
            pass

    def _dispatch_active_user_change(self, username: str, display_name: str) -> None:
        """Notify registered listeners about an active user update."""

        for listener in list(self._active_user_change_listeners):
            try:
                listener(username, display_name)
            except Exception:  # pragma: no cover - listener failures logged defensively
                self._logger.error(
                    "Active user listener %s raised an exception", listener, exc_info=True
                )

    # ------------------------------------------------------------------
    # User account service helpers
    # ------------------------------------------------------------------
    def _get_user_account_service(self):
        """Return the lazily-initialized user account service."""

        if self._user_account_service is None:
            from modules.user_accounts.user_account_service import UserAccountService

            self._user_account_service = UserAccountService(
                config_manager=self._config_manager,
                conversation_repository=self._conversation_repository,
            )

        return self._user_account_service

    # ------------------------------------------------------------------
    # Async wrappers exposed to ATLAS
    # ------------------------------------------------------------------
    async def list_user_accounts(self) -> List[Dict[str, object]]:
        service = self._get_user_account_service()
        return await run_async_in_thread(service.list_users)

    async def search_user_accounts(self, query_text: str) -> List[Dict[str, object]]:
        service = self._get_user_account_service()
        return await run_async_in_thread(service.search_users, query_text)

    async def get_user_account_details(
        self, username: str, *, tenant_id: Optional[str] = None
    ) -> Optional[Dict[str, object]]:
        service = self._get_user_account_service()
        return await run_async_in_thread(
            service.get_user_details, username, tenant_id=tenant_id
        )

    async def get_user_account_overview(self) -> Dict[str, object]:
        service = self._get_user_account_service()
        return await run_async_in_thread(service.get_user_overview)

    async def activate_user_account(self, username: str) -> None:
        service = self._get_user_account_service()
        await run_async_in_thread(service.set_active_user, username)
        self.refresh_active_user_identity()

    async def register_user_account(
        self,
        username: str,
        password: str,
        email: str,
        name: Optional[str] = None,
        dob: Optional[str] = None,
    ) -> Dict[str, object]:
        service = self._get_user_account_service()

        account = await run_async_in_thread(
            service.register_user,
            username,
            password,
            email,
            name,
            dob,
        )

        await run_async_in_thread(service.set_active_user, account.username)
        self.refresh_active_user_identity()
        return {
            "id": account.id,
            "username": account.username,
            "email": account.email,
            "name": account.name,
            "dob": account.dob,
            "display_name": account.name or account.username,
        }

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
        service = self._get_user_account_service()

        active_username = await run_async_in_thread(service.get_active_user)

        account = await run_async_in_thread(
            service.update_user,
            username,
            password=password,
            current_password=current_password,
            email=email,
            name=name,
            dob=dob,
        )

        if active_username and account.username == active_username:
            await run_async_in_thread(service.set_active_user, account.username)

        self.refresh_active_user_identity()
        return {
            "id": account.id,
            "username": account.username,
            "email": account.email,
            "name": account.name,
            "dob": account.dob,
            "display_name": account.name or account.username,
        }

    def get_user_password_requirements(self) -> PasswordRequirements:
        service = self._get_user_account_service()
        return service.get_password_requirements()

    def describe_user_password_requirements(self) -> str:
        service = self._get_user_account_service()
        return service.describe_password_requirements()

    async def login_user_account(self, username: str, password: str) -> bool:
        service = self._get_user_account_service()
        success = await run_async_in_thread(service.authenticate_user, username, password)
        if success:
            await run_async_in_thread(service.set_active_user, username)
            self.refresh_active_user_identity()
        return bool(success)

    async def request_password_reset(
        self, identifier: str
    ) -> Optional[Dict[str, object]]:
        service = self._get_user_account_service()
        challenge = await run_async_in_thread(
            service.initiate_password_reset,
            identifier,
        )
        if not challenge:
            return None

        return {
            "username": challenge.username,
            "token": challenge.token,
            "expires_at": challenge.expires_at_iso(),
        }

    async def verify_password_reset_token(self, username: str, token: str) -> bool:
        service = self._get_user_account_service()
        return await run_async_in_thread(
            service.verify_password_reset_token,
            username,
            token,
        )

    async def complete_password_reset(
        self, username: str, token: str, new_password: str
    ) -> bool:
        service = self._get_user_account_service()
        return await run_async_in_thread(
            service.complete_password_reset,
            username,
            token,
            new_password,
        )

    async def logout_active_user(self) -> None:
        service = self._get_user_account_service()
        await run_async_in_thread(service.set_active_user, None)
        self.refresh_active_user_identity(prefer_generic=True)

    async def delete_user_account(self, username: str) -> None:
        service = self._get_user_account_service()
        deleted = await run_async_in_thread(service.delete_user, username)

        if not deleted:
            raise ValueError(f"Unknown user: {username}")

        self.refresh_active_user_identity(prefer_generic=True)
