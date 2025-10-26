"""User profile helpers backed by the conversation store."""

from __future__ import annotations

import platform
import re
import subprocess
from threading import Lock
from typing import Any, Dict, Mapping, Optional

from ATLAS.config import ConfigManager
from modules.conversation_store import ConversationStoreRepository
from modules.logging.logger import setup_logger


class SystemInfo:
    logger = None
    _cache_lock = Lock()
    _cached_info = None

    @staticmethod
    def set_logger(logger_instance):
        """Sets the logger for the SystemInfo class."""
        SystemInfo.logger = logger_instance

    @staticmethod
    def invalidate_cache():
        """Invalidate any cached system information."""
        with SystemInfo._cache_lock:
            SystemInfo._cached_info = None

    @staticmethod
    def run_command(command):
        """Runs a system command and returns the output."""
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
            )
            if result.returncode != 0 and SystemInfo.logger:
                SystemInfo.logger.warning(
                    "Command '%s' exited with code %s: %s",
                    command,
                    result.returncode,
                    result.stderr.strip(),
                )
            return result.stdout
        except Exception as exc:
            if SystemInfo.logger:
                SystemInfo.logger.warning("Error running command '%s': %s", command, exc)
            return ""

    @staticmethod
    def get_basic_info():
        """Gets basic system information based on the OS."""
        if platform.system() == "Windows":
            output = SystemInfo.run_command("systeminfo")
        else:
            output = SystemInfo.run_command("hostnamectl")
        return output

    @staticmethod
    def get_cpu_info():
        """Gets CPU information based on the OS."""
        if platform.system() == "Windows":
            output = SystemInfo.run_command(
                "wmic cpu get name,NumberOfCores,NumberOfLogicalProcessors /format:list"
            )
        else:
            output = SystemInfo.run_command("lscpu")
        return output

    @staticmethod
    def get_memory_info():
        """Gets memory information based on the OS."""
        if platform.system() == "Windows":
            output = SystemInfo.run_command("wmic MemoryChip get Capacity /format:list")
            capacities = []
            for line in output.splitlines():
                match = re.search(r"Capacity=(\d+)", line)
                if match:
                    capacities.append(int(match.group(1)))

            if capacities:
                total_memory = sum(capacities)
                return f"Total Physical Memory: {total_memory / (1024**3):.2f} GB"

            return "Total Physical Memory: Unknown"
        else:
            output = SystemInfo.run_command("free -h")
        return output

    @staticmethod
    def get_disk_info():
        """Gets disk information based on the OS."""
        if platform.system() == "Windows":
            output = SystemInfo.run_command("wmic diskdrive get model,size /format:list")
        else:
            output = SystemInfo.run_command("lsblk")
        return output

    @staticmethod
    def get_network_info():
        """Gets network configuration based on the OS."""
        if platform.system() == "Windows":
            output = SystemInfo.run_command("ipconfig /all")
        else:
            output = SystemInfo.run_command("ip addr")
        return output

    @staticmethod
    def get_detailed_system_info():
        """Compiles detailed system information from various sources."""
        with SystemInfo._cache_lock:
            if SystemInfo._cached_info is not None:
                return dict(SystemInfo._cached_info)

            info = {
                "Basic Info": SystemInfo.get_basic_info(),
                "CPU Info": SystemInfo.get_cpu_info(),
                "Memory Info": SystemInfo.get_memory_info(),
                "Disk Info": SystemInfo.get_disk_info(),
                "Network Info": SystemInfo.get_network_info(),
            }
            SystemInfo._cached_info = info
            return dict(SystemInfo._cached_info)


class UserDataManager:
    _system_info_cache = None
    _system_info_cache_lock = Lock()

    def __init__(
        self,
        user: str,
        *,
        repository: Optional[ConversationStoreRepository] = None,
        config_manager: Optional[ConfigManager] = None,
    ) -> None:
        """Initialise the manager with a username and backing repository."""

        self.config_manager = config_manager or ConfigManager()
        self.logger = setup_logger(__name__)
        SystemInfo.set_logger(self.logger)

        self.user = user
        self.profile: Optional[str] = None
        self.emr: Optional[str] = None

        self._profile_data: Optional[Dict[str, Any]] = None
        self._documents: Optional[Dict[str, Any]] = None
        self._system_info: Optional[str] = None

        self._repository = repository
        self._repository_lock = Lock()

    @classmethod
    def invalidate_system_info_cache(cls):
        """Clear cached system information for all UserDataManager instances."""
        SystemInfo.invalidate_cache()
        with cls._system_info_cache_lock:
            cls._system_info_cache = None

    def _resolve_repository(self) -> ConversationStoreRepository:
        if self._repository is not None:
            return self._repository

        with self._repository_lock:
            if self._repository is not None:
                return self._repository

            try:
                factory = self.config_manager.get_conversation_store_session_factory()
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Failed to resolve conversation store session factory: %s", exc)
                raise

            if factory is None:
                try:
                    self.config_manager.ensure_postgres_conversation_store()
                    factory = self.config_manager.get_conversation_store_session_factory()
                except Exception as exc:  # pragma: no cover - bootstrap failures
                    self.logger.error("Conversation store unavailable: %s", exc)
                    raise

            if factory is None:
                raise RuntimeError("Conversation store session factory is not configured.")

            retention = self.config_manager.get_conversation_retention_policies()
            repository = ConversationStoreRepository(factory, retention=retention)
            try:
                repository.create_schema()
            except Exception as exc:  # pragma: no cover - schema already exists
                self.logger.debug("Profile schema initialisation skipped: %s", exc)
            self._repository = repository
            return repository

    def _default_profile(self) -> Dict[str, Any]:
        return {"Username": self.user}

    def _derive_display_name(self, profile: Mapping[str, Any]) -> Optional[str]:
        for key in ("Full Name", "full_name", "Display Name", "display_name", "name"):
            value = profile.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _ensure_profile_state(self) -> Dict[str, Any]:
        if self._profile_data is not None and self._documents is not None:
            return {
                "profile": dict(self._profile_data),
                "documents": dict(self._documents),
            }

        repository = self._resolve_repository()
        record = repository.get_user_profile(self.user)

        if not record or not isinstance(record.get("profile"), Mapping) or not record.get("profile"):
            profile_payload = self._default_profile()
            display_name = self._derive_display_name(profile_payload)
            record = repository.upsert_user_profile(
                self.user,
                profile_payload,
                display_name=display_name,
                documents={},
            )

        profile_section = dict(record.get("profile") or {})
        if not profile_section.get("Username"):
            profile_section["Username"] = self.user

        documents_section = dict(record.get("documents") or {})

        self._profile_data = profile_section
        self._documents = documents_section
        self.profile = None
        self.emr = None

        return {
            "profile": profile_section,
            "documents": documents_section,
            "display_name": record.get("display_name"),
        }

    def get_profile(self) -> Dict[str, Any]:
        """Retrieve the user's profile data from the repository."""
        if self._profile_data is not None:
            return dict(self._profile_data)

        self.logger.info("Entering get_profile() method")
        try:
            bundle = self._ensure_profile_state()
        except Exception as exc:
            self.logger.error("Error loading profile metadata: %s", exc)
            fallback = self._default_profile()
            self._profile_data = fallback
            self._documents = {}
            return dict(fallback)

        profile_section = dict(bundle.get("profile") or {})
        self._profile_data = profile_section
        return dict(profile_section)

    def format_profile_as_text(self, profile_json: Mapping[str, Any]) -> str:
        """Formats the user's profile as a string."""
        self.logger.info("Formatting profile.")
        profile_lines = []
        for key, value in profile_json.items():
            line = f"{key}: {value}"
            profile_lines.append(line)
        return "\n".join(profile_lines)

    def get_profile_text(self) -> str:
        """Retrieves the user's profile as a formatted string."""
        if self.profile is not None:
            return self.profile

        self.logger.info("Entering get_profile_text() method")
        profile_json = self.get_profile()
        self.profile = self.format_profile_as_text(profile_json)
        return self.profile

    def get_emr(self) -> str:
        """Retrieve the stored EMR text for the user."""
        if self.emr is not None:
            return self.emr

        self.logger.info("Getting EMR.")
        try:
            if self._documents is None:
                bundle = self._ensure_profile_state()
                documents = bundle.get("documents", {})
            else:
                documents = self._documents
        except Exception as exc:
            self.logger.error("Error loading EMR metadata: %s", exc)
            self.emr = ""
            return self.emr

        emr_payload = ""
        if isinstance(documents, Mapping):
            raw_emr = documents.get("emr")
            if raw_emr:
                emr_payload = str(raw_emr)

        if not emr_payload:
            self.emr = ""
            return self.emr

        normalised = emr_payload.replace("\n", " ")
        normalised = re.sub(r"\s+", " ", normalised).strip()
        self.emr = normalised
        return self.emr

    def get_system_info(self) -> str:
        """Retrieves and formats detailed system information for persona personalisation."""
        if self._system_info is not None:
            return self._system_info

        try:
            with self.__class__._system_info_cache_lock:
                if self.__class__._system_info_cache is not None:
                    self._system_info = self.__class__._system_info_cache
                    return self._system_info

            detailed_info = SystemInfo.get_detailed_system_info()
            formatted_info = ""
            for category, info in detailed_info.items():
                self.logger.debug("Retrieving %s information:", category)
                self.logger.debug(info)
                formatted_info += f"--- {category} ---\n{info}\n"
            self.logger.info("System information retrieved successfully.")

            with self.__class__._system_info_cache_lock:
                if self.__class__._system_info_cache is None:
                    self.__class__._system_info_cache = formatted_info

                self._system_info = self.__class__._system_info_cache

            return self._system_info
        except Exception as exc:
            self.logger.warning("Error retrieving system information: %s", exc)
            self._system_info = "System information not available"
            return self._system_info
