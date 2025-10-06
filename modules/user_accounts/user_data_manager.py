# modules/user_accounts/user_data_manager.py

import re
import json
import platform
import subprocess
from pathlib import Path
from threading import Lock
from typing import Optional

from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger

try:
    from platformdirs import user_data_dir as _user_data_dir
except ImportError:  # pragma: no cover - optional dependency
    try:
        from appdirs import user_data_dir as _user_data_dir  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        _user_data_dir = None


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
        except Exception as e:
            if SystemInfo.logger:
                SystemInfo.logger.warning(f"Error running command '{command}': {e}")
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
            output = SystemInfo.run_command("wmic cpu get name,NumberOfCores,NumberOfLogicalProcessors /format:list")
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

    def __init__(self, user, base_dir: Optional[str] = None):
        """
        Initializes the UserDataManager with the given user.

        Args:
            user (str): The username of the user.
            base_dir (Optional[str]): Optional base directory override for user data.
        """
        self.config_manager = ConfigManager()
        self.logger = setup_logger(__name__)
        SystemInfo.set_logger(self.logger)
        self.user = user
        self.profile = None
        self.emr = None
        self._profile_data = None
        self._system_info = None
        self._base_directory = self._determine_base_directory(base_dir)
        self._profiles_dir = self._base_directory / 'user_profiles'

    @classmethod
    def invalidate_system_info_cache(cls):
        """Clear cached system information for all UserDataManager instances."""
        SystemInfo.invalidate_cache()
        with cls._system_info_cache_lock:
            cls._system_info_cache = None


    def _determine_base_directory(self, override_dir: Optional[str]) -> Path:
        if override_dir is not None:
            self.logger.debug(
                "Using provided base directory for user data: %s",
                override_dir,
            )
            return Path(override_dir).expanduser().resolve()

        app_root = self._get_app_root()
        if app_root:
            resolved = Path(app_root).expanduser().resolve() / 'modules' / 'user_accounts'
            self.logger.debug("Using app root directory for user data: %s", resolved)
            return resolved

        if _user_data_dir:
            fallback_base = Path(_user_data_dir("ATLAS", "ATLAS"))
            self.logger.debug(
                "Using OS user data directory for user data: %s",
                fallback_base,
            )
        else:  # pragma: no cover - only used when helpers unavailable
            fallback_base = Path.home() / '.atlas'
            self.logger.debug(
                "Falling back to home directory for user data: %s",
                fallback_base,
            )

        return fallback_base

    def _get_app_root(self) -> Optional[str]:
        get_app_root = getattr(self.config_manager, 'get_app_root', None)
        if not callable(get_app_root):
            return None

        try:
            return get_app_root()
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(
                "Failed to retrieve app root from ConfigManager: %s",
                exc,
            )
            return None

    def _load_profile_template(self):
        """Load the default profile template from disk."""

        template_path = Path(__file__).resolve().with_name('user_template')
        try:
            with template_path.open('r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.error(f"Failed to load profile template: {exc}")
            return {"Username": self.user}

    def get_profile(self):
        """
        Retrieves the user's profile from a JSON file.

        Returns:
            dict: The user's profile as a dictionary.
        """
        self.logger.info("Entering get_profile() method")

        try:
            profile_path = self._profiles_dir / f"{self.user}.json"

            cached_profile = self._profile_data or {}

            if not profile_path.exists():
                self.logger.error(f"Profile file does not exist: {profile_path}")
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                profile = self._load_profile_template()
                profile['Username'] = self.user

                for key in ('Email', 'Full Name'):
                    value = cached_profile.get(key)
                    if value:
                        profile[key] = value

                with profile_path.open('w', encoding='utf-8') as file:
                    json.dump(profile, file, indent=4)

                self._profile_data = profile
                self.profile = None
                return dict(self._profile_data)

            if self._profile_data is not None:
                return dict(self._profile_data)

            with profile_path.open('r', encoding='utf-8') as file:
                profile = json.load(file)
                self.logger.info("Profile found")
                self._profile_data = profile
                self.profile = None
                return dict(self._profile_data)

        except Exception as e:
            self.logger.error(f"Error loading profile: {e}")
            return {}
        
    def format_profile_as_text(self, profile_json):
        """
        Formats the user's profile as a string.

        Args:
            profile_json (dict): The user's profile as a dictionary.

        Returns:
            str: The formatted profile as a string.
        """
        self.logger.info("Formatting profile.")
        profile_lines = []
        for key, value in profile_json.items():
            line = f"{key}: {value}"
            profile_lines.append(line)
        return '\n'.join(profile_lines)
    
    def get_profile_text(self):
        """
        Retrieves the user's profile as a formatted string.

        Returns:
            str: The user's profile as a formatted string.
        """
        if self.profile is not None:
            return self.profile

        self.logger.info("Entering get_profile_text() method")
        profile_json = self.get_profile()
        self.profile = self.format_profile_as_text(profile_json)
        return self.profile
    
    def get_emr(self):
        """
        Retrieves the user's EMR (Electronic Medical Record) from a text file.

        Returns:
            str: The user's EMR as a string.
        """
        if self.emr is not None:
            return self.emr

        self.logger.info("Getting EMR.")
        EMR_path = self._profiles_dir / f"{self.user}_emr.txt"

        self.logger.info(f"EMR path: {EMR_path}")

        if not EMR_path.exists():
            self.logger.error(f"EMR file does not exist: {EMR_path}")
            self.emr = ""
            return self.emr

        try:
            with EMR_path.open('r', encoding='utf-8') as file:
                EMR = file.read()
                EMR = EMR.replace("\n", " ")
                EMR = re.sub(r'\s+', ' ', EMR)
                self.emr = EMR.strip()
                return self.emr
        except Exception as e:
            self.logger.error(f"Error loading EMR: {e}")
            self.emr = ""
            return self.emr

    def get_system_info(self):
        """
        Retrieves and formats detailed system information for persona personalization.

        Returns:
            str: The formatted system information as a string.
        """
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
                self.logger.debug(f"Retrieving {category} information:")
                self.logger.debug(info)
                formatted_info += f"--- {category} ---\n{info}\n"
            self.logger.info("System information retrieved successfully.")

            with self.__class__._system_info_cache_lock:
                if self.__class__._system_info_cache is None:
                    self.__class__._system_info_cache = formatted_info

                self._system_info = self.__class__._system_info_cache

            return self._system_info
        except Exception as e:
            self.logger.warning(f"Error retrieving system information: {e}")
            self._system_info = "System information not available"
            return self._system_info
