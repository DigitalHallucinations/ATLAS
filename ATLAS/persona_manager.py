# ATLAS/persona_manager.py

import os
import json
import copy
from collections.abc import Iterable as IterableABC, Mapping as MappingABC
from typing import Any, Dict, List, Optional, Tuple
from modules.user_accounts.user_data_manager import UserDataManager
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger
from modules.Personas import (
    PersonaValidationError,
    _validate_persona_payload,
    build_tool_state as personas_build_tool_state,
    build_skill_state as personas_build_skill_state,
    load_persona_definition,
    load_tool_metadata,
    load_skill_catalog,
    normalize_allowed_tools,
    normalize_allowed_skills,
    persist_persona_definition,
)

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

    def __init__(self, master, user: str, config_manager: Optional[ConfigManager] = None):
        self.master = master
        self.user = user
        if config_manager is None and hasattr(master, "config_manager"):
            config_manager = master.config_manager
        self.config_manager = config_manager or ConfigManager()
        self.logger = setup_logger(__name__)
        self.user_data_manager = UserDataManager(self.user)
        self.persona_base_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'Personas')
        self.persona_names: List[str] = self.load_persona_names(self.persona_base_path)
        self.personas: Dict[str, dict] = {}  # Cache for loaded personas
        self.default_persona_name = "ATLAS"
        self.current_persona = None
        self.current_system_prompt = None
        self._tool_metadata_cache: Optional[Tuple[List[str], Dict[str, Any]]] = None
        self._skill_metadata_cache: Optional[Tuple[List[str], Dict[str, Any]]] = None

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

    def _get_tool_metadata(self) -> Tuple[List[str], Dict[str, Any]]:
        """Return cached shared tool metadata (order preserved)."""

        if self._tool_metadata_cache is None:
            order, lookup = load_tool_metadata(config_manager=self.config_manager)
            self._tool_metadata_cache = (order, lookup)
        return self._tool_metadata_cache

    def _get_skill_metadata(self) -> Tuple[List[str], Dict[str, Any]]:
        """Return cached shared skill metadata (order preserved)."""

        if self._skill_metadata_cache is None:
            order, lookup = load_skill_catalog(config_manager=self.config_manager)
            self._skill_metadata_cache = (order, lookup)
        return self._skill_metadata_cache

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
        
        try:
            order, _lookup = self._get_tool_metadata()
        except Exception:  # pragma: no cover - metadata load errors already logged
            order, _lookup = [], {}

        persona_data = load_persona_definition(
            persona_name,
            config_manager=self.config_manager,
            metadata_order=order,
            metadata_lookup=_lookup,
        )

        if persona_data is None:
            return None

        self.personas[persona_name] = persona_data
        self.logger.info("Persona '%s' loaded successfully.", persona_name)
        return persona_data

    def set_user(self, user: str) -> None:
        """Update personalization state for a new active user."""

        sanitized = (user or "User").strip() or "User"
        self.user = sanitized

        try:
            invalidate_cache = getattr(UserDataManager, "invalidate_system_info_cache")
        except AttributeError:
            invalidate_cache = None

        if callable(invalidate_cache):
            try:
                invalidate_cache()
            except Exception:  # pragma: no cover - defensive logging only
                self.logger.warning("Failed to invalidate shared system info cache", exc_info=True)

        self.user_data_manager = UserDataManager(self.user)

        if self.current_persona is not None:
            try:
                self.current_system_prompt = self.build_system_prompt(self.current_persona)
            except Exception:  # pragma: no cover - defensive logging only
                self.logger.error(
                    "Failed to rebuild system prompt for persona '%s'", self.current_persona.get("name"),
                    exc_info=True,
                )


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

        locked_sections = self._build_locked_sections(persona)

        def _string(value: Any) -> str:
            return self._normalize_string(value)

        persona_type_raw = persona.get('type') or {}
        type_state: Dict[str, Dict[str, Any]] = {}
        keys = set(self.PERSONA_TYPE_KEYS) | set(persona_type_raw.keys())
        bool_extras = {
            'personal_assistant': {'access_to_calendar', 'calendar_write_enabled'},
            'tech_support': {'access_to_logs'},
        }

        for key in keys:
            entry = persona_type_raw.get(key) or {}
            extras: Dict[str, Any] = {}
            extra_bool_fields = bool_extras.get(key, set())
            for extra_key, extra_value in entry.items():
                if extra_key == 'enabled':
                    continue
                if extra_key in extra_bool_fields:
                    extras[extra_key] = self._as_bool(extra_value)
                else:
                    extras[extra_key] = _string(extra_value)
            if key == 'personal_assistant':
                access_enabled = extras.get('access_to_calendar', False)
                extras.setdefault('access_to_calendar', bool(access_enabled))
                extras['calendar_write_enabled'] = bool(extras.get('calendar_write_enabled', False)) and bool(extras['access_to_calendar'])
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
                    'computed_start_locked': locked_sections['start_locked'],
                    'computed_end_locked': locked_sections['end_locked'],
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

        try:
            order, lookup = self._get_tool_metadata()
        except Exception:  # pragma: no cover - fallback when metadata fails
            order, lookup = [], {}

        try:
            state['tools'] = personas_build_tool_state(
                persona,
                config_manager=self.config_manager,
                metadata_order=order,
                metadata_lookup=lookup,
            )
        except Exception:
            self.logger.error("Failed to build tool state for persona '%s'", persona.get('name'), exc_info=True)
            state['tools'] = {'allowed': normalize_allowed_tools(persona.get('allowed_tools')), 'available': []}

        try:
            skill_order, skill_lookup = self._get_skill_metadata()
        except Exception:  # pragma: no cover - fallback when metadata fails
            skill_order, skill_lookup = [], {}

        try:
            state['skills'] = personas_build_skill_state(
                persona,
                config_manager=self.config_manager,
                metadata_order=skill_order,
                metadata_lookup=skill_lookup,
            )
        except Exception:
            self.logger.error("Failed to build skill state for persona '%s'", persona.get('name'), exc_info=True)
            state['skills'] = {
                'allowed': normalize_allowed_skills(persona.get('allowed_skills')),
                'available': [],
            }

        return state

    def get_editor_state(self, persona_name: str) -> Optional[Dict[str, Any]]:
        """Return typed persona information for the editor views."""

        persona = self.get_persona(persona_name)
        if persona is None:
            return None
        return self._build_editor_state(persona)

    def _compute_start_locked_text(self, persona: Dict[str, Any]) -> str:
        """Generate the default start_locked text for a persona."""

        name = self._normalize_string(persona.get('name')).strip() or "Assistant"
        meaning = self._normalize_string(persona.get('meaning')).strip()

        base = f"The name of the user you are speaking to is <<name>>. Your name is {name}"
        if meaning:
            return f"{base}: ({meaning})."
        return f"{base}."

    def _type_entry(self, persona: Dict[str, Any], key: str) -> Dict[str, Any]:
        persona_type = persona.get('type') or {}
        entry = persona_type.get(key) or {}
        if isinstance(entry, dict):
            return entry
        return {}

    def _compute_end_locked_text(self, persona: Dict[str, Any]) -> str:
        """Generate the dynamic end_locked text based on persona flags."""

        dynamic_parts: List[str] = []

        if self._as_bool(persona.get('user_profile_enabled')):
            dynamic_parts.append("User Profile: <<Profile>>")

        medical_entry = self._type_entry(persona, 'medical_persona')
        if self._as_bool(medical_entry.get('enabled')):
            dynamic_parts.append("User EMR: <<emr>>")

        educational_entry = self._type_entry(persona, 'educational_persona')
        if self._as_bool(educational_entry.get('enabled')):
            subject = self._normalize_string(educational_entry.get('subject_specialization')).strip()
            level = self._normalize_string(educational_entry.get('education_level')).strip()
            subject = subject or "General"
            level = level or "High School"
            dynamic_parts.append(f"Subject: {subject}")
            dynamic_parts.append(f"Level: {level}")
            dynamic_parts.append("Provide explanations suitable for the student's level.")

        fitness_entry = self._type_entry(persona, 'fitness_persona')
        if self._as_bool(fitness_entry.get('enabled')):
            goal = self._normalize_string(fitness_entry.get('fitness_goal')).strip()
            preference = self._normalize_string(fitness_entry.get('exercise_preference')).strip()
            goal = goal or "Weight Loss"
            preference = preference or "Gym Workouts"
            dynamic_parts.append(f"Fitness Goal: {goal}")
            dynamic_parts.append(f"Exercise Preference: {preference}")
            dynamic_parts.append("Offer motivational support and track progress.")

        language_entry = self._type_entry(persona, 'language_instructor')
        if self._as_bool(language_entry.get('enabled')):
            target_language = self._normalize_string(language_entry.get('target_language')).strip()
            proficiency_level = self._normalize_string(language_entry.get('proficiency_level')).strip()
            target_language = target_language or "Spanish"
            proficiency_level = proficiency_level or "Beginner"
            dynamic_parts.append(f"Target Language: {target_language}")
            dynamic_parts.append(f"Proficiency Level: {proficiency_level}")
            dynamic_parts.append("Engage in conversation to practice the target language.")

        if not dynamic_parts:
            return ""

        dynamic_content = " ".join(dynamic_parts)
        dynamic_content += " Clear responses and relevant information are key for a great user experience. Ask for clarity or offer input as needed."
        return dynamic_content

    def _build_locked_sections(self, persona: Dict[str, Any]) -> Dict[str, str]:
        """Return computed locked sections for the provided persona payload."""

        return {
            'start_locked': self._compute_start_locked_text(persona),
            'end_locked': self._compute_end_locked_text(persona),
        }

    def compute_locked_content(
        self,
        persona_name: Optional[str] = None,
        general: Optional[Dict[str, Any]] = None,
        flags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Compute preview locked content using persona data and optional overrides."""

        persona: Dict[str, Any] = {}

        if persona_name:
            persona_data = self.get_persona(persona_name)
            if persona_data:
                persona = copy.deepcopy(persona_data)

        if not persona:
            persona = {}

        general = general or {}
        if 'name' in general:
            persona['name'] = general.get('name')
        if 'meaning' in general:
            persona['meaning'] = general.get('meaning')

        flags = flags or {}
        if 'user_profile_enabled' in flags:
            persona['user_profile_enabled'] = flags.get('user_profile_enabled')
        if 'sys_info_enabled' in flags:
            persona['sys_info_enabled'] = flags.get('sys_info_enabled')

        type_overrides = flags.get('type') if isinstance(flags, dict) else None
        if type_overrides:
            persona_type = persona.setdefault('type', {})
            for key, entry in type_overrides.items():
                existing = dict(persona_type.get(key) or {})
                if not isinstance(entry, dict):
                    persona_type[key] = entry
                    continue
                updated = existing.copy()
                for field, value in entry.items():
                    updated[field] = value
                persona_type[key] = updated

        return self._build_locked_sections(persona)

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


    def _build_substitution_data(self, persona: dict) -> Dict[str, str]:
        """Return the placeholder substitution dictionary for ``persona``."""

        substitutions: Dict[str, str] = {"<<name>>": self.user}
        user_data_manager = self.user_data_manager

        # Profile placeholder
        try:
            if persona.get("user_profile_enabled") == "True":
                self.logger.info(f"Adding Profile to persona: '{persona['name']}'.")
                substitutions["<<Profile>>"] = user_data_manager.get_profile_text()
            else:
                self.logger.info(f"Profile not added for persona: '{persona['name']}'.")
                substitutions["<<Profile>>"] = "Profile not available for this persona."
        except Exception as exc:  # pragma: no cover - defensive logging only
            self.logger.error("Failed to load profile text for persona '%s'", persona.get("name"), exc_info=True)
            substitutions["<<Profile>>"] = f"Profile unavailable: {exc}"

        # EMR placeholder
        try:
            if persona.get("medical_persona") == "True":
                self.logger.info(f"Adding EMR to medical persona: '{persona['name']}'.")
                substitutions["<<emr>>"] = user_data_manager.get_emr()
            else:
                self.logger.info(f"EMR not added for non-medical persona: '{persona['name']}'.")
                substitutions["<<emr>>"] = "EMR not available for this persona."
        except Exception as exc:  # pragma: no cover - defensive logging only
            self.logger.error("Failed to load EMR for persona '%s'", persona.get("name"), exc_info=True)
            substitutions["<<emr>>"] = f"EMR unavailable: {exc}"

        # System info placeholder
        try:
            if persona.get("sys_info_enabled") == "True":
                self.logger.info(f"Adding system info to persona: '{persona['name']}'.")
                substitutions["<<sysinfo>>"] = user_data_manager.get_system_info()
            else:
                self.logger.info(f"System info not added for persona: '{persona['name']}'.")
                substitutions["<<sysinfo>>"] = "System info not available for this persona."
        except Exception as exc:  # pragma: no cover - defensive logging only
            self.logger.error(
                "Failed to load system info for persona '%s'", persona.get("name"), exc_info=True
            )
            substitutions["<<sysinfo>>"] = f"System info unavailable: {exc}"

        return substitutions

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
        substitutions = self._build_substitution_data(persona)

        # Assemble the system prompt from the content parts
        content = persona.get("content", {})
        parts = [
            content.get("start_locked", "").strip(),
            content.get("editable_content", "").strip(),
            content.get("end_locked", "").strip(),
        ]
        # Filter out empty strings and join with spaces
        personalized_content = " ".join(filter(None, parts))

        # Replace placeholders in the assembled content with user data
        for placeholder, data in substitutions.items():
            personalized_content = personalized_content.replace(placeholder, str(data or ""))

        self.logger.info(
            "System prompt built for persona '%s': %s", persona.get("name"), personalized_content
        )
        return personalized_content

    def get_current_persona_context(self) -> Dict[str, Any]:
        """Return the active persona prompt and substitution metadata."""

        persona = self.current_persona or {}
        substitutions: Dict[str, str] = {}
        allowed_tools: List[str] = []
        capability_tags: List[str] = []

        def _normalize_strings(values: Any) -> List[str]:
            normalized: List[str] = []
            seen: set[str] = set()

            if isinstance(values, str):
                candidates = [values]
            elif isinstance(values, IterableABC):
                candidates = list(values)
            else:
                candidates = []

            for item in candidates:
                if isinstance(item, MappingABC):
                    candidate_value = item.get("name")
                else:
                    candidate_value = item

                if candidate_value is None:
                    continue

                text = str(candidate_value).strip()
                if text and text not in seen:
                    normalized.append(text)
                    seen.add(text)

            return normalized

        def _is_enabled_flag(value: Any) -> bool:
            if isinstance(value, str):
                return value.strip().lower() == "true"
            return bool(value)

        if persona:
            try:
                substitutions = self._build_substitution_data(persona)
            except Exception as exc:  # pragma: no cover - defensive logging only
                self.logger.error("Failed to compute substitution data: %s", exc, exc_info=True)
                substitutions = {}

            allowed_tools = _normalize_strings(persona.get("allowed_tools"))

            raw_capability_tags: List[str] = _normalize_strings(persona.get("capability_tags"))
            capability_tags = list(raw_capability_tags)

            persona_types = persona.get("type")
            if isinstance(persona_types, MappingABC):
                seen = set(capability_tags)
                for key, value in persona_types.items():
                    token = str(key).strip() if key is not None else ""
                    if not token or token in seen:
                        continue

                    enabled_flag = False
                    if isinstance(value, MappingABC):
                        enabled_flag = _is_enabled_flag(value.get("enabled"))
                    else:
                        enabled_flag = _is_enabled_flag(value)

                    if enabled_flag:
                        capability_tags.append(token)
                        seen.add(token)

        prompt = self.current_system_prompt
        if persona and not prompt:
            try:
                prompt = self.build_system_prompt(persona)
                self.current_system_prompt = prompt
            except Exception as exc:  # pragma: no cover - defensive logging only
                self.logger.error("Failed to rebuild system prompt: %s", exc, exc_info=True)
                prompt = None

        return {
            "system_prompt": prompt or "",
            "substitutions": substitutions,
            "persona_name": persona.get("name") if isinstance(persona, dict) else None,
            "allowed_tools": allowed_tools,
            "capability_tags": capability_tags,
        }

    def update_persona(self, persona, *, rationale: str = "Persona manager update"):
        """Update the persona settings and save them to the corresponding file."""
        persona_name = persona.get("name") or ""
        if not persona_name:
            self.logger.error("Cannot persist persona without a name: %s", persona)
            return

        try:
            persist_persona_definition(
                persona_name,
                persona,
                config_manager=self.config_manager,
                rationale=rationale,
            )
        except Exception:
            self.logger.error("Error saving persona '%s'", persona_name, exc_info=True)
        else:
            self.logger.info("Persona '%s' updated successfully.", persona_name)

    def _persist_persona(
        self,
        original_name: str,
        persona: dict,
        *,
        rationale: str = "Persona manager update",
    ) -> None:
        """Persist persona changes and refresh cache entries."""
        if not persona:
            return

        current_name = persona.get("name") or original_name
        if original_name:
            self.personas[original_name] = persona
        if current_name:
            self.personas[current_name] = persona

        self.update_persona(persona, rationale=rationale)

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

        self._persist_persona(
            persona_name,
            persona,
            rationale="Updated persona general information via persona manager",
        )
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

        self._persist_persona(
            persona_name,
            persona,
            rationale=f"Updated persona flag '{flag}' via persona manager",
        )
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

        self._persist_persona(
            persona_name,
            persona,
            rationale="Updated provider defaults via persona manager",
        )
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

        self._persist_persona(
            persona_name,
            persona,
            rationale="Updated speech defaults via persona manager",
        )
        return {"success": True, "persona": persona}

    def set_allowed_tools(self, persona_name: str, allowed_tools: Optional[List[str]]) -> Dict[str, Any]:
        """Persist persona-specific tool selections."""

        persona = self.get_persona(persona_name)
        if persona is None:
            return {"success": False, "errors": [f"Persona '{persona_name}' could not be loaded."]}

        metadata_order, metadata_lookup = self._get_tool_metadata()
        normalized = normalize_allowed_tools(
            allowed_tools or [], metadata_order=metadata_order
        )

        candidate_persona = dict(persona)
        candidate_persona['allowed_tools'] = normalized

        known_tools: set[str] = {str(name) for name in metadata_order}
        known_tools.update(str(name) for name in metadata_lookup.keys())
        existing_tools = persona.get('allowed_tools') or []
        known_tools.update(str(name) for name in existing_tools if str(name))

        try:
            skill_order, skill_lookup = self._get_skill_metadata()
        except Exception:  # pragma: no cover - defensive guard
            skill_order, skill_lookup = [], {}
        known_skills: set[str] = {str(name) for name in skill_order}
        known_skills.update(str(name) for name in skill_lookup.keys())
        existing_skills = persona.get('allowed_skills') or []
        known_skills.update(str(name) for name in existing_skills if str(name))

        try:
            _validate_persona_payload(
                {'persona': [candidate_persona]},
                persona_name=persona_name,
                tool_ids=known_tools,
                skill_ids=known_skills,
                config_manager=self.config_manager,
            )
        except PersonaValidationError as exc:
            return {"success": False, "errors": [str(exc)]}

        persona['allowed_tools'] = normalized

        self._persist_persona(
            persona_name,
            persona,
            rationale="Updated allowed tools via persona manager",
        )

        return {"success": True, "persona": persona}

    def set_allowed_skills(self, persona_name: str, allowed_skills: Optional[List[str]]) -> Dict[str, Any]:
        """Persist persona-specific skill selections."""

        persona = self.get_persona(persona_name)
        if persona is None:
            return {"success": False, "errors": [f"Persona '{persona_name}' could not be loaded."]}

        skill_order, skill_lookup = self._get_skill_metadata()
        normalized = normalize_allowed_skills(
            allowed_skills or [], metadata_order=skill_order
        )

        candidate_persona = dict(persona)
        candidate_persona['allowed_skills'] = normalized

        known_skills: set[str] = {str(name) for name in skill_order}
        known_skills.update(str(name) for name in skill_lookup.keys())
        existing_skills = persona.get('allowed_skills') or []
        known_skills.update(str(name) for name in existing_skills if str(name))

        try:
            tool_order, tool_lookup = self._get_tool_metadata()
        except Exception:  # pragma: no cover - defensive guard
            tool_order, tool_lookup = [], {}
        known_tools: set[str] = {str(name) for name in tool_order}
        known_tools.update(str(name) for name in tool_lookup.keys())
        existing_tools = persona.get('allowed_tools') or []
        known_tools.update(str(name) for name in existing_tools if str(name))

        try:
            _validate_persona_payload(
                {'persona': [candidate_persona]},
                persona_name=persona_name,
                tool_ids=known_tools,
                skill_ids=known_skills,
                config_manager=self.config_manager,
            )
        except PersonaValidationError as exc:
            return {"success": False, "errors": [str(exc)]}

        persona['allowed_skills'] = normalized

        self._persist_persona(
            persona_name,
            persona,
            rationale="Updated allowed skills via persona manager",
        )

        return {"success": True, "persona": persona}

    def update_persona_from_form(
        self,
        persona_name: str,
        general: Optional[Dict[str, Any]] = None,
        persona_type: Optional[Dict[str, Any]] = None,
        provider: Optional[Dict[str, Any]] = None,
        speech: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
        skills: Optional[List[str]] = None,
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

        if tools is not None:
            _apply_result(
                self.set_allowed_tools(current_name, tools),
                "Failed to update persona tools.",
            )

        if skills is not None:
            _apply_result(
                self.set_allowed_skills(current_name, skills),
                "Failed to update persona skills.",
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
