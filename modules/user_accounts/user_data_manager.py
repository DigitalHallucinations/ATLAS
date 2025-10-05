# modules/user_accounts/user_data_manager.py

import os
import re
import json
import platform
import subprocess
from threading import Lock
from ATLAS.config import ConfigManager
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
            total_memory = sum([int(re.search(r'Capacity=(\d+)', line).group(1)) for line in output.splitlines() if "Capacity" in line])
            return f"Total Physical Memory: {total_memory / (1024**3):.2f} GB"
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

    def __init__(self, user):
        """
        Initializes the UserDataManager with the given user.

        Args:
            user (str): The username of the user.
        """
        self.config_manager = ConfigManager()
        self.logger = setup_logger(__name__)
        SystemInfo.set_logger(self.logger)
        self.user = user
        self.profile = None
        self.emr = None
        self._system_info = None

    @classmethod
    def invalidate_system_info_cache(cls):
        """Clear cached system information for all UserDataManager instances."""
        SystemInfo.invalidate_cache()
        with cls._system_info_cache_lock:
            cls._system_info_cache = None
        

    def get_profile(self):
        """
        Retrieves the user's profile from a JSON file.

        Returns:
            dict: The user's profile as a dictionary.
        """
        self.logger.info("Entering get_profile() method")
        try:
            profile_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'modules', 'user_accounts', 'user_profiles',
                f"{self.user}.json"
            )

            if not os.path.exists(profile_path):
                self.logger.error(f"Profile file does not exist: {profile_path}")
                return {}

            with open(profile_path, 'r', encoding='utf-8') as file:
                profile = json.load(file)
                self.logger.info("Profile found")
                return profile

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
        EMR_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'modules', 'user_accounts', 'user_profiles',
            f"{self.user}_emr.txt"
        )

        self.logger.info(f"EMR path: {EMR_path}")

        if not os.path.exists(EMR_path):
            self.logger.error(f"EMR file does not exist: {EMR_path}")
            self.emr = ""
            return self.emr
        
        try:
            with open(EMR_path, 'r', encoding='utf-8') as file:
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
