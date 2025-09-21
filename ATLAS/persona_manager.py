# ATLAS/persona_manager.py

import os
import json
import copy
from typing import Any, Dict, Iterable, List, Optional
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


    def build_start_locked(self, persona: Dict[str, Any]) -> str:
        """Compose the canonical start-locked section for a persona."""

        name = (persona.get("name") or "Assistant").strip()
        meaning = (persona.get("meaning") or "").strip()

        base = f"The name of the user you are speaking to is <<name>>. Your name is {name}"
        if meaning:
            meaning_text = meaning.rstrip()
            if not meaning_text.endswith((".", "!", "?")):
                meaning_text += "."
            base += f": {meaning_text}"
        else:
            base += "."
        return base

    def _entry_enabled(self, entry: Optional[Dict[str, Any]]) -> bool:
        if not entry:
            return False
        value = entry.get("enabled")
        if isinstance(value, str):
            return value.lower() == "true"
        return bool(value)

    def _format_clause(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            return ""
        if normalized.endswith((".", "!", "?")):
            return normalized
        return f"{normalized}."

    def _gather_type_clauses(self, persona: Dict[str, Any]) -> Iterable[str]:
        types = persona.get("type") or {}

        educational = types.get("educational_persona")
        if self._entry_enabled(educational):
            subject = (educational.get("subject_specialization") or "General").strip()
            level = (educational.get("education_level") or "High School").strip()
            style = (educational.get("teaching_style") or "Lecture Style").strip()
            yield self._format_clause(f"Subject focus: {subject}")
            yield self._format_clause(f"Learner level: {level}")
            yield self._format_clause(f"Preferred teaching style: {style}")

        fitness = types.get("fitness_persona")
        if self._entry_enabled(fitness):
            goal = (fitness.get("fitness_goal") or "General Fitness").strip()
            preference = (fitness.get("exercise_preference") or "Mixed Workouts").strip()
            yield self._format_clause(f"Fitness goal: {goal}")
            yield self._format_clause(f"Exercise preference: {preference}")

        language = types.get("language_instructor")
        if self._entry_enabled(language):
            target = (language.get("target_language") or "Spanish").strip()
            proficiency = (language.get("proficiency_level") or "Beginner").strip()
            yield self._format_clause(f"Target language: {target}")
            yield self._format_clause(f"Learner proficiency: {proficiency}")

        legal = types.get("legal_persona")
        if self._entry_enabled(legal):
            jurisdiction = (legal.get("jurisdiction") or "General").strip()
            area = (legal.get("area_of_law") or "Common").strip()
            disclaimer = (legal.get("disclaimer") or "Provide informational guidance only").strip()
            yield self._format_clause(f"Jurisdiction: {jurisdiction}")
            yield self._format_clause(f"Area of law: {area}")
            yield self._format_clause(disclaimer)

        financial = types.get("financial_advisor")
        if self._entry_enabled(financial):
            goals = (financial.get("investment_goals") or "Capital appreciation").strip()
            risk = (financial.get("risk_tolerance") or "Moderate").strip()
            horizon = (financial.get("time_horizon") or "Medium term").strip()
            yield self._format_clause(f"Investment goals: {goals}")
            yield self._format_clause(f"Risk tolerance: {risk}")
            yield self._format_clause(f"Time horizon: {horizon}")

        tech = types.get("tech_support")
        if self._entry_enabled(tech):
            specialization = (tech.get("product_specialization") or "General").strip()
            expertise = (tech.get("user_expertise_level") or "Beginner").strip()
            access_logs = tech.get("access_to_logs")
            access = "enabled" if isinstance(access_logs, str) and access_logs.lower() == "true" or bool(access_logs) else "disabled"
            yield self._format_clause(f"Tech support specialization: {specialization}")
            yield self._format_clause(f"Assume user expertise: {expertise}")
            yield self._format_clause(f"Access to diagnostic logs is {access}")

        assistant = types.get("personal_assistant")
        if self._entry_enabled(assistant):
            timezone = (assistant.get("time_zone") or "UTC").strip()
            style = (assistant.get("communication_style") or "Professional").strip()
            calendar = assistant.get("access_to_calendar")
            access = "enabled" if isinstance(calendar, str) and calendar.lower() == "true" or bool(calendar) else "disabled"
            yield self._format_clause(f"Preferred communication style: {style}")
            yield self._format_clause(f"Time zone reference: {timezone}")
            yield self._format_clause(f"Calendar access is {access}")

        therapist = types.get("therapist")
        if self._entry_enabled(therapist):
            therapy_style = (therapist.get("therapy_style") or "Supportive").strip()
            session_length = (therapist.get("session_length") or "Flexible").strip()
            confidentiality = (therapist.get("confidentiality") or "Maintain privacy and respect boundaries").strip()
            yield self._format_clause(f"Therapy style: {therapy_style}")
            yield self._format_clause(f"Typical session length: {session_length}")
            yield self._format_clause(confidentiality)

        travel = types.get("travel_guide")
        if self._entry_enabled(travel):
            preferences = (travel.get("destination_preferences") or "Open to ideas").strip()
            travel_style = (travel.get("travel_style") or "Balanced").strip()
            interests = (travel.get("interests") or "Culture and cuisine").strip()
            yield self._format_clause(f"Destination preferences: {preferences}")
            yield self._format_clause(f"Travel style: {travel_style}")
            yield self._format_clause(f"Interests: {interests}")

        storyteller = types.get("storyteller")
        if self._entry_enabled(storyteller):
            genre = (storyteller.get("genre") or "Adventure").strip()
            audience = (storyteller.get("audience_age_group") or "General").strip()
            length = (storyteller.get("story_length") or "Medium").strip()
            yield self._format_clause(f"Preferred genre: {genre}")
            yield self._format_clause(f"Target audience: {audience}")
            yield self._format_clause(f"Story length: {length}")

        game_master = types.get("game_master")
        if self._entry_enabled(game_master):
            game_type = (game_master.get("game_type") or "Tabletop RPG").strip()
            difficulty = (game_master.get("difficulty_level") or "Moderate").strip()
            theme = (game_master.get("theme") or "Fantasy").strip()
            yield self._format_clause(f"Game type: {game_type}")
            yield self._format_clause(f"Difficulty level: {difficulty}")
            yield self._format_clause(f"Theme: {theme}")

        chef = types.get("chef")
        if self._entry_enabled(chef):
            cuisine = (chef.get("cuisine_preferences") or "International").strip()
            dietary = (chef.get("dietary_restrictions") or "None").strip()
            skill = (chef.get("skill_level") or "Intermediate").strip()
            yield self._format_clause(f"Cuisine preferences: {cuisine}")
            yield self._format_clause(f"Dietary restrictions: {dietary}")
            yield self._format_clause(f"Assumed skill level: {skill}")

    def build_end_locked(self, persona: Dict[str, Any]) -> str:
        """Compose the canonical end-locked section for a persona."""

        clauses: List[str] = []

        if persona.get("user_profile_enabled") == "True":
            clauses.append("User Profile: <<Profile>>")

        medical_entry = persona.get("type", {}).get("medical_persona")
        if self._entry_enabled(medical_entry):
            clauses.append("User EMR: <<emr>>")

        clauses.extend(self._gather_type_clauses(persona))

        if clauses:
            clauses.append(
                "Clear responses and relevant information are key for a great user experience. "
                "Ask for clarity or offer input as needed."
            )

        return " ".join(filter(None, clauses))

    def get_default_editable_template(self, persona: Optional[Dict[str, Any]] = None) -> str:
        """Return the default editable content scaffold for personas."""

        persona_name = (persona or {}).get("name") or "the persona"
        return (
            "Goals:\n"
            "1) Provide accurate, concise answers.\n"
            "2) Ask clarifying questions when needed.\n"
            "Tone:\n"
            "- Friendly, professional, and helpful.\n"
            "Persona Focus:\n"
            f"- Keep advice aligned with {persona_name}'s expertise.\n"
            "Constraints:\n"
            "- Ground responses in provided context and known capabilities.\n"
        )

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
        start_locked = content.get("start_locked") or self.build_start_locked(persona)
        editable = content.get("editable_content", "").strip()
        end_locked = content.get("end_locked") or self.build_end_locked(persona)
        parts = [
            start_locked.strip(),
            editable,
            end_locked.strip(),
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

        persona['name'] = new_name
        persona['meaning'] = self._normalize_string(general.get("meaning"))

        content_payload = general.get("content") or {}
        persona.setdefault('content', {})
        persona['content']['editable_content'] = self._normalize_string(content_payload.get('editable_content'))

        persona['sys_info_enabled'] = self._normalize_bool(persona_type.get('sys_info_enabled', False))
        persona['user_profile_enabled'] = self._normalize_bool(persona_type.get('user_profile_enabled', False))

        submitted_types = persona_type.get('type') or {}
        existing_types = persona.get('type', {}).copy()
        normalized_types: Dict[str, Dict[str, str]] = {}
        for type_key in set(self.PERSONA_TYPE_KEYS).union(submitted_types.keys()).union(existing_types.keys()):
            submitted_entry = submitted_types.get(type_key)
            if submitted_entry is None:
                # Preserve existing entry when the form does not manage the type explicitly.
                if type_key in existing_types:
                    normalized_types[type_key] = existing_types[type_key].copy()
                else:
                    normalized_types[type_key] = {'enabled': 'False'}
                continue

            enabled = self._normalize_bool(submitted_entry.get('enabled', False))
            entry: Dict[str, str] = {'enabled': enabled}

            if enabled == "True":
                for opt_key, opt_value in submitted_entry.items():
                    if opt_key == 'enabled':
                        continue
                    if opt_value not in (None, ""):
                        entry[opt_key] = self._normalize_string(opt_value)
            normalized_types[type_key] = entry

        persona['type'] = normalized_types

        persona['provider'] = provider_name
        persona['model'] = model_name

        persona['Speech_provider'] = self._normalize_string(speech.get('Speech_provider'))
        persona['voice'] = self._normalize_string(speech.get('voice'))

        persona['content']['start_locked'] = self.build_start_locked(persona)
        persona['content']['end_locked'] = self.build_end_locked(persona)

        self.personas[persona_name] = persona
        if new_name != persona_name:
            self.personas[new_name] = persona

        self.update_persona(persona)
        return {"success": True, "persona": persona}

    def preview_prompt_sections(
        self,
        persona_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Return prompt sections for a persona without persisting changes."""

        overrides = overrides or {}
        persona = self.get_persona(persona_name)

        if persona is not None:
            persona_copy: Dict[str, Any] = copy.deepcopy(persona)
        else:
            persona_copy = {
                "name": overrides.get("name") or persona_name,
                "meaning": overrides.get("meaning") or "",
                "content": {"editable_content": ""},
                "type": {},
                "user_profile_enabled": "False",
                "sys_info_enabled": "False",
            }

        for key, value in overrides.items():
            if key == "type" and isinstance(value, dict):
                type_store = persona_copy.setdefault("type", {})
                for type_key, entry in value.items():
                    if isinstance(entry, dict):
                        type_entry = type_store.get(type_key, {}).copy()
                        for opt_key, opt_val in entry.items():
                            if opt_key == "enabled":
                                type_entry[opt_key] = self._normalize_bool(opt_val)
                            else:
                                type_entry[opt_key] = self._normalize_string(opt_val)
                        type_store[type_key] = type_entry
            elif key in {"user_profile_enabled", "sys_info_enabled"}:
                persona_copy[key] = self._normalize_bool(value)
            elif key == "content" and isinstance(value, dict):
                persona_copy.setdefault("content", {}).update({
                    sub_key: self._normalize_string(sub_val)
                    for sub_key, sub_val in value.items()
                })
            else:
                persona_copy[key] = self._normalize_string(value)

        start_locked = self.build_start_locked(persona_copy)
        end_locked = self.build_end_locked(persona_copy)

        return {
            "start_locked": start_locked,
            "end_locked": end_locked,
            "editable_template": self.get_default_editable_template(persona_copy),
        }

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
