# ATLAS/persona_manager.py

import os
import json
from typing import Any, Dict, List, Optional
from modules.user_accounts.user_data_manager import UserDataManager
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger

class PersonaManager:
    """
    The PersonaManager class is responsible for managing personas within the application.
    It loads persona data from corresponding folders and updates the current persona based on user selection.

    Attributes:
        master (object): A reference to the master application object.
        user (str): The username of the current user.
        personas (dict): A cache of loaded personas.
        current_persona (dict): The currently selected persona.
        current_system_prompt (str): The system prompt generated based on the selected persona.
    """
    
    PERSONA_TYPE_KEYS = [
        'Agent', 'medical_persona', 'educational_persona', 'fitness_persona', 'language_instructor',
        'legal_persona', 'financial_advisor', 'tech_support', 'personal_assistant', 'therapist',
        'travel_guide', 'storyteller', 'game_master', 'chef'
    ]

    def __init__(self, master, user: str):
        self.master = master
        self.user = user
        self.config_manager = ConfigManager()
        self.logger = setup_logger(__name__)
        self.persona_base_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'Personas')
        self.persona_names: List[str] = self.load_persona_names(self.persona_base_path)
        self.personas: Dict[str, dict] = {}  # Cache for loaded personas
        self.default_persona_name = "ATLAS"
        self.current_persona = None
        self.current_system_prompt = None

        # Load the default persona and generate the system prompt
        self.load_default_persona()

    def load_default_persona(self):
        """Loads and personalizes the default persona."""
        persona = self.load_persona(self.default_persona_name)
        if persona:
            self.current_persona = persona
            self.current_system_prompt = self.build_system_prompt(persona)
            self.logger.info(f"Default persona '{self.default_persona_name}' loaded with personalized system prompt.")
        else:
            self.logger.error(f"Failed to load default persona: '{self.default_persona_name}'")

    def load_persona_names(self, persona_path: str) -> List[str]:
        """Get the list of persona directories."""
        try:
            persona_names = [name for name in os.listdir(persona_path) if os.path.isdir(os.path.join(persona_path, name))]
            self.logger.info(f"Persona names loaded: {persona_names}")
            return persona_names
        except OSError as e:
            self.logger.error(f"Error loading persona names: {e}")
            return []

    def load_persona(self, persona_name: str) -> Optional[dict]:
        """
        Load a single persona from its respective folder.

        Args:
            persona_name (str): The name of the persona to load.

        Returns:
            dict: The loaded persona data, or None if loading fails.
        """
        if persona_name in self.personas:  # Check cache first
            self.logger.info(f"Persona '{persona_name}' retrieved from cache.")
            return self.personas[persona_name]
        
        persona_folder = os.path.join(self.master.config_manager.get_app_root(), 'modules', 'Personas', persona_name, 'Persona')
        json_file = os.path.join(persona_folder, f'{persona_name}.json')

        self.logger.debug(f"Attempting to load persona from folder: {persona_folder}")
        self.logger.debug(f"Persona JSON file path: {json_file}")

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as file:
                    persona_data = json.load(file)
                    if "persona" in persona_data and isinstance(persona_data["persona"], list) and len(persona_data["persona"]) > 0:
                        self.personas[persona_name] = persona_data["persona"][0]  # Cache the loaded persona
                        self.logger.info(f"Persona '{persona_name}' loaded successfully from '{json_file}'.")
                        return self.personas[persona_name]
                    else:
                        self.logger.error(f"Invalid persona format in '{json_file}'. Expected a list under 'persona' key.")
                        return None
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.logger.error(f"Error loading persona '{persona_name}': {e}")
                return None
        else:
            self.logger.error(f"JSON file for persona '{persona_name}' not found at '{json_file}'.")
            return None


    def get_persona(self, persona_name: str) -> Optional[dict]:
        """Retrieve the persona by name, loading it from disk if necessary."""
        return self.load_persona(persona_name)

    def _as_bool(self, value: Any) -> bool:
        """Convert serialized persona truthy values into booleans."""
        if isinstance(value, str):
            return value.lower() == "true"
        return bool(value)

    def _build_editor_state(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """Return an editor-friendly snapshot for a loaded persona."""

        content = persona.get('content') or {}

        def _string(value: Any) -> str:
            return self._normalize_string(value)

        persona_type_raw = persona.get('type') or {}
        type_state: Dict[str, Dict[str, Any]] = {}
        keys = set(self.PERSONA_TYPE_KEYS) | set(persona_type_raw.keys())

        for key in keys:
            entry = persona_type_raw.get(key) or {}
            extras = {
                extra_key: _string(extra_value)
                for extra_key, extra_value in entry.items()
                if extra_key != 'enabled'
            }
            type_state[key] = {'enabled': self._as_bool(entry.get('enabled')), **extras}

        state = {
            'original_name': _string(persona.get('name')),
            'general': {
                'name': _string(persona.get('name')),
                'meaning': _string(persona.get('meaning')),
                'content': {
                    'start_locked': _string(content.get('start_locked')),
                    'editable_content': _string(content.get('editable_content')),
                    'end_locked': _string(content.get('end_locked')),
                },
            },
            'flags': {
                'sys_info_enabled': self._as_bool(persona.get('sys_info_enabled')),
                'user_profile_enabled': self._as_bool(persona.get('user_profile_enabled')),
                'type': type_state,
            },
            'provider': {
                'provider': _string(persona.get('provider')),
                'model': _string(persona.get('model')),
            },
            'speech': {
                'Speech_provider': _string(persona.get('Speech_provider')),
                'voice': _string(persona.get('voice')),
            },
            'ui_state': dict(persona.get('ui_state') or {}),
        }

        return state

    def get_editor_state(self, persona_name: str) -> Optional[Dict[str, Any]]:
        """Return typed persona information for the editor views."""

        persona = self.get_persona(persona_name)
        if persona is None:
            return None
        return self._build_editor_state(persona)

    def updater(self, selected_persona_name: str):
        """Update the current persona."""
        self.logger.info(f"Attempting to update persona to '{selected_persona_name}'.")
        
        persona = self.get_persona(selected_persona_name)
        if not persona:
            self.logger.error(f"Failed to update persona: '{selected_persona_name}' not found or invalid.")
            return

        self.current_persona = persona
        self.current_system_prompt = self.build_system_prompt(persona)

        self.master.system_name = selected_persona_name
        self.master.system_name_tag = selected_persona_name

        self.logger.info(f"Persona switched to '{selected_persona_name}' with new system prompt.")
        self.logger.info(f"Current persona is now: {self.current_persona}")


    def build_system_prompt(self, persona: dict) -> str:
        """
        Builds the system prompt using user-specific information,
        including the EMR only for medical personas and profile only if enabled.

        Args:
            persona (dict): The persona data to personalize.

        Returns:
            str: The personalized system prompt.
        """
        self.logger.info(f"Building system prompt for persona '{persona.get('name')}'.")
        user_data_manager = UserDataManager(self.user)

        # General user data available to all personas
        user_data = {
            "<<name>>": self.user,
        }

        # Only add Profile if user_profile_enabled is True
        if persona.get("user_profile_enabled") == "True":
            self.logger.info(f"Adding Profile to persona: '{persona['name']}'.")
            user_data["<<Profile>>"] = user_data_manager.get_profile_text()
        else:
            self.logger.info(f"Profile not added for persona: '{persona['name']}'.")
            user_data["<<Profile>>"] = "Profile not available for this persona."

        # Only add EMR if the persona is a medical persona
        if persona.get("medical_persona") == "True":
            self.logger.info(f"Adding EMR to medical persona: '{persona['name']}'.")
            user_data["<<emr>>"] = user_data_manager.get_emr()
        else:
            self.logger.info(f"EMR not added for non-medical persona: '{persona['name']}'.")
            user_data["<<emr>>"] = "EMR not available for this persona."

        # Only add sysinfo if the persona is a sys_admin_persona
        if persona.get("sys_info_enabled") == "True":
            self.logger.info(f"Adding system info to persona: '{persona['name']}'.")
            user_data["<<sysinfo>>"] = user_data_manager.get_system_info()
        else:
            self.logger.info(f"System info not added for persona: '{persona['name']}'.")
            user_data["<<sysinfo>>"] = "System info not available for this persona."

        # Assemble the system prompt from the content parts
        content = persona.get("content", {})
        parts = [
            content.get("start_locked", "").strip(),
            content.get("editable_content", "").strip(),
            content.get("end_locked", "").strip(),
        ]
        # Filter out empty strings and join with spaces
        personalized_content = ' '.join(filter(None, parts))

        # Replace placeholders in the assembled content with user data
        for placeholder, data in user_data.items():
            personalized_content = personalized_content.replace(placeholder, data)

        self.logger.info(f"System prompt built for persona '{persona['name']}': {personalized_content}")
        return personalized_content

    def update_persona(self, persona):
        """Update the persona settings and save them to the corresponding file."""
        persona_name = persona.get("name")
        persona_folder = os.path.join(self.persona_base_path, persona_name, 'Persona')
        json_file = os.path.join(persona_folder, f'{persona_name}.json')

        try:
            with open(json_file, 'w', encoding='utf-8') as file:
                json.dump({"persona": [persona]}, file, indent=4)
            self.logger.info(f"Persona '{persona_name}' updated successfully.")
        except OSError as e:
            self.logger.error(f"Error saving persona '{persona_name}': {e}")

    def _persist_persona(self, original_name: str, persona: dict) -> None:
        """Persist persona changes and refresh cache entries."""
        if not persona:
            return

        current_name = persona.get("name") or original_name
        if original_name:
            self.personas[original_name] = persona
        if current_name:
            self.personas[current_name] = persona

        self.update_persona(persona)

    def _normalize_bool(self, value: Any) -> str:
        """Convert a truthy value into the persona serialization format."""
        if isinstance(value, str):
            return "True" if value.lower() == "true" else "False"
        return "True" if bool(value) else "False"

    def _normalize_string(self, value: Any) -> str:
        """Ensure persona string fields are serialized consistently."""
        if value is None:
            return ""
        return str(value)

    def update_general_info(
        self,
        persona_name: str,
        name: str,
        meaning: Optional[str],
        content: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Update general persona metadata (name/meaning/content)."""

        persona = self.get_persona(persona_name)
        if persona is None:
            return {"success": False, "errors": [f"Persona '{persona_name}' could not be loaded."]}

        normalized_name = self._normalize_string(name)
        persona['name'] = normalized_name
        persona['meaning'] = self._normalize_string(meaning)

        content = content or {}
        persona['content'] = {
            'start_locked': self._normalize_string(content.get('start_locked')),
            'editable_content': self._normalize_string(content.get('editable_content')),
            'end_locked': self._normalize_string(content.get('end_locked')),
        }

        self._persist_persona(persona_name, persona)
        return {"success": True, "persona": persona}

    def set_flag(
        self,
        persona_name: str,
        flag: str,
        enabled: Any,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Toggle persona boolean flags while normalizing and persisting changes."""

        persona = self.get_persona(persona_name)
        if persona is None:
            return {"success": False, "errors": [f"Persona '{persona_name}' could not be loaded."]}

        normalized_enabled = self._normalize_bool(enabled)

        if flag in {"sys_info_enabled", "user_profile_enabled"}:
            persona[flag] = normalized_enabled
        else:
            persona_type = persona.setdefault('type', {})
            entry = {'enabled': normalized_enabled}

            if normalized_enabled == "True":
                for key, value in (extras or {}).items():
                    if key == 'enabled':
                        continue
                    if value not in (None, ""):
                        entry[key] = self._normalize_string(value)
            persona_type[flag] = entry

        self._persist_persona(persona_name, persona)
        return {"success": True, "persona": persona}

    def set_provider_defaults(
        self,
        persona_name: str,
        provider: str,
        model: str,
    ) -> Dict[str, Any]:
        """Update the provider/model pair for a persona."""

        persona = self.get_persona(persona_name)
        if persona is None:
            return {"success": False, "errors": [f"Persona '{persona_name}' could not be loaded."]}

        persona['provider'] = self._normalize_string(provider)
        persona['model'] = self._normalize_string(model)

        self._persist_persona(persona_name, persona)
        return {"success": True, "persona": persona}

    def set_speech_defaults(
        self,
        persona_name: str,
        speech_provider: str,
        voice: str,
    ) -> Dict[str, Any]:
        """Update speech provider defaults for a persona."""

        persona = self.get_persona(persona_name)
        if persona is None:
            return {"success": False, "errors": [f"Persona '{persona_name}' could not be loaded."]}

        persona['Speech_provider'] = self._normalize_string(speech_provider)
        persona['voice'] = self._normalize_string(voice)

        self._persist_persona(persona_name, persona)
        return {"success": True, "persona": persona}

    def update_persona_from_form(
        self,
        persona_name: str,
        general: Optional[Dict[str, Any]] = None,
        persona_type: Optional[Dict[str, Any]] = None,
        provider: Optional[Dict[str, Any]] = None,
        speech: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply structured persona editor payloads and persist changes.

        Args:
            persona_name: The identifier of the persona being edited.
            general: Values captured from the "General" tab.
            persona_type: Persona type toggle state and optional fields.
            provider: Provider/model pair selected for the persona.
            speech: Speech provider defaults.

        Returns:
            Dict[str, Any]: A response describing success state and optional errors.
        """

        errors: List[str] = []

        if not persona_name:
            errors.append("Persona name is required for update.")
        persona = self.get_persona(persona_name) if not errors else None
        if persona is None:
            errors.append(f"Persona '{persona_name}' could not be loaded.")

        general = general or {}
        provider = provider or {}
        speech = speech or {}
        persona_type = persona_type or {}

        new_name = self._normalize_string(general.get("name"))
        if not new_name:
            errors.append("Persona must have a name.")

        provider_name = self._normalize_string(provider.get("provider"))
        if not provider_name:
            errors.append("Provider selection is required.")

        model_name = self._normalize_string(provider.get("model"))
        if not model_name:
            errors.append("Model selection is required.")

        if errors:
            return {"success": False, "errors": errors}

        persona_data = persona
        current_name = persona_name

        def _apply_result(result: Dict[str, Any], default_error: str) -> bool:
            nonlocal persona_data, current_name, errors
            if not result.get('success'):
                result_errors = result.get('errors')
                if result_errors:
                    errors.extend(result_errors)
                else:
                    errors.append(default_error)
                return False

            persona_data = result.get('persona') or persona_data
            if persona_data:
                current_name = persona_data.get('name', current_name) or current_name
            return True

        if not _apply_result(
            self.update_general_info(persona_name, new_name, general.get("meaning"), general.get("content")),
            "Failed to update persona general settings.",
        ):
            return {"success": False, "errors": errors}

        _apply_result(
            self.set_flag(current_name, 'sys_info_enabled', persona_type.get('sys_info_enabled', False)),
            "Failed to update system info flag.",
        )
        _apply_result(
            self.set_flag(current_name, 'user_profile_enabled', persona_type.get('user_profile_enabled', False)),
            "Failed to update user profile flag.",
        )

        submitted_types = persona_type.get('type') or {}
        for type_key, submitted_entry in submitted_types.items():
            extras = {k: v for k, v in submitted_entry.items() if k != 'enabled'}
            _apply_result(
                self.set_flag(current_name, type_key, submitted_entry.get('enabled', False), extras),
                f"Failed to update persona flag '{type_key}'.",
            )

        _apply_result(
            self.set_provider_defaults(current_name, provider_name, model_name),
            "Failed to update provider defaults.",
        )
        _apply_result(
            self.set_speech_defaults(
                current_name,
                speech.get('Speech_provider'),
                speech.get('voice'),
            ),
            "Failed to update speech defaults.",
        )

        if errors:
            return {"success": False, "errors": errors}

        return {"success": True, "persona": persona_data}

    def show_message(self, role: str, message: str):
        """Dispatch a persona-related message via the master callback when available."""
        dispatcher = getattr(self.master, "message_dispatcher", None)

        if not callable(dispatcher):
            self.logger.warning(
                "No message dispatcher registered on master. Unable to deliver persona message: %s",
                message,
            )
            return

        try:
            dispatcher(role, message)
        except Exception as exc:
            self.logger.error("Message dispatcher failed: %s", exc, exc_info=True)

    def get_current_persona_prompt(self) -> str:
        """Returns the current persona's system prompt."""
        return self.current_system_prompt
