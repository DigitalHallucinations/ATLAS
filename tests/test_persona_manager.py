import json
import sys
import types

import pytest

if 'yaml' not in sys.modules:
    yaml_stub = types.ModuleType('yaml')
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.dump = lambda *_args, **_kwargs: ''
    sys.modules['yaml'] = yaml_stub

if 'dotenv' not in sys.modules:
    dotenv_stub = types.ModuleType('dotenv')
    dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
    dotenv_stub.set_key = lambda *_args, **_kwargs: None
    dotenv_stub.find_dotenv = lambda *_args, **_kwargs: ''
    sys.modules['dotenv'] = dotenv_stub

import ATLAS.persona_manager as persona_manager_module
from ATLAS.persona_manager import PersonaManager


def _write_persona(personas_dir, persona):
    persona_dir = personas_dir / persona['name'] / 'Persona'
    persona_dir.mkdir(parents=True, exist_ok=True)
    file_path = persona_dir / f"{persona['name']}.json"
    file_path.write_text(json.dumps({'persona': [persona]}, indent=4), encoding='utf-8')
    return file_path


@pytest.fixture
def persona_manager(tmp_path, monkeypatch):
    personas_dir = tmp_path / 'modules' / 'Personas'
    default_dir = personas_dir / 'ATLAS' / 'Persona'
    default_dir.mkdir(parents=True, exist_ok=True)
    default_persona = {
        'name': 'ATLAS',
        'meaning': 'Default persona',
        'content': {
            'start_locked': 'Hello',
            'editable_content': 'Default',
            'end_locked': 'Goodbye',
        },
        'provider': 'openai',
        'model': 'gpt-4o',
        'sys_info_enabled': 'False',
        'user_profile_enabled': 'False',
        'type': {'Agent': {'enabled': 'False'}},
        'Speech_provider': '11labs',
        'voice': 'default',
    }
    (default_dir / 'ATLAS.json').write_text(json.dumps({'persona': [default_persona]}, indent=4), encoding='utf-8')

    class _StubConfigManager:
        def __init__(self):
            self._app_root = str(tmp_path)

        def get_app_root(self):
            return self._app_root

    class _StubUserDataManager:
        def __init__(self, _user):
            pass

        def get_profile_text(self):
            return 'Profile text'

        def get_emr(self):
            return 'EMR data'

        def get_system_info(self):
            return 'System info'

    monkeypatch.setattr(persona_manager_module, 'ConfigManager', _StubConfigManager)
    monkeypatch.setattr(persona_manager_module, 'UserDataManager', _StubUserDataManager)

    master = types.SimpleNamespace(config_manager=_StubConfigManager())
    manager = PersonaManager(master, user='tester')
    manager.persona_base_path = str(personas_dir)
    manager.persona_names = manager.load_persona_names(str(personas_dir))
    return manager, personas_dir


def test_update_persona_from_form_enables_optional_fields(persona_manager):
    manager, personas_dir = persona_manager
    persona = {
        'name': 'StudyBuddy',
        'meaning': 'Helps with studying',
        'content': {
            'start_locked': 'Intro',
            'editable_content': 'Content',
            'end_locked': 'Outro',
        },
        'provider': 'openai',
        'model': 'gpt-4o',
        'sys_info_enabled': 'False',
        'user_profile_enabled': 'False',
        'type': {
            'Agent': {'enabled': 'False'},
            'educational_persona': {'enabled': 'False'},
        },
        'Speech_provider': '11labs',
        'voice': 'jack',
    }
    file_path = _write_persona(personas_dir, persona)

    general_payload = {
        'name': 'StudyBuddy',
        'meaning': 'Updated meaning',
        'content': {
            'start_locked': 'Start',
            'editable_content': 'Editable',
            'end_locked': 'End',
        },
    }
    persona_type_payload = {
        'sys_info_enabled': True,
        'user_profile_enabled': False,
        'type': {
            'educational_persona': {
                'enabled': True,
                'subject_specialization': 'Physics',
                'education_level': 'College',
                'teaching_style': 'Interactive',
            }
        },
    }
    provider_payload = {'provider': 'openai', 'model': 'gpt-4o'}
    speech_payload = {'Speech_provider': '11labs', 'voice': 'armin'}

    result = manager.update_persona_from_form(
        persona['name'], general_payload, persona_type_payload, provider_payload, speech_payload
    )

    assert result['success'] is True
    saved = json.loads(file_path.read_text(encoding='utf-8'))['persona'][0]
    assert saved['sys_info_enabled'] == 'True'
    assert saved['user_profile_enabled'] == 'False'
    educational = saved['type']['educational_persona']
    assert educational['enabled'] == 'True'
    assert educational['subject_specialization'] == 'Physics'
    assert educational['education_level'] == 'College'
    assert educational['teaching_style'] == 'Interactive'
    assert saved['content']['editable_content'] == 'Editable'


def test_update_persona_from_form_disables_persona_and_clears_options(persona_manager):
    manager, personas_dir = persona_manager
    persona = {
        'name': 'Coach',
        'meaning': 'Fitness coach',
        'content': {
            'start_locked': 'Intro',
            'editable_content': 'Content',
            'end_locked': 'Outro',
        },
        'provider': 'openai',
        'model': 'gpt-4o',
        'sys_info_enabled': 'False',
        'user_profile_enabled': 'False',
        'type': {
            'Agent': {'enabled': 'False'},
            'fitness_persona': {
                'enabled': 'True',
                'fitness_goal': 'Strength',
                'exercise_preference': 'Weights',
            },
        },
        'Speech_provider': '11labs',
        'voice': 'jack',
    }
    file_path = _write_persona(personas_dir, persona)

    general_payload = {
        'name': 'Coach',
        'meaning': 'Fitness coach',
        'content': persona['content'],
    }
    persona_type_payload = {
        'type': {
            'fitness_persona': {'enabled': False},
        }
    }
    provider_payload = {'provider': 'openai', 'model': 'gpt-4o'}
    speech_payload = {'Speech_provider': '11labs', 'voice': 'jack'}

    result = manager.update_persona_from_form(
        persona['name'], general_payload, persona_type_payload, provider_payload, speech_payload
    )

    assert result['success'] is True
    saved = json.loads(file_path.read_text(encoding='utf-8'))['persona'][0]
    assert saved['type']['fitness_persona'] == {'enabled': 'False'}


def test_update_persona_from_form_updates_provider_and_speech(persona_manager):
    manager, personas_dir = persona_manager
    persona = {
        'name': 'Narrator',
        'meaning': 'Tells stories',
        'content': {
            'start_locked': 'Intro',
            'editable_content': 'Content',
            'end_locked': 'Outro',
        },
        'provider': 'openai',
        'model': 'gpt-4o',
        'sys_info_enabled': 'False',
        'user_profile_enabled': 'False',
        'type': {'Agent': {'enabled': 'False'}},
        'Speech_provider': '11labs',
        'voice': 'jack',
    }
    file_path = _write_persona(personas_dir, persona)

    general_payload = {
        'name': 'Narrator',
        'meaning': 'Tells stories',
        'content': persona['content'],
    }
    persona_type_payload = {}
    provider_payload = {'provider': 'anthropic', 'model': 'claude-3'}
    speech_payload = {'Speech_provider': 'openai_tts', 'voice': 'alloy'}

    result = manager.update_persona_from_form(
        persona['name'], general_payload, persona_type_payload, provider_payload, speech_payload
    )

    assert result['success'] is True
    saved = json.loads(file_path.read_text(encoding='utf-8'))['persona'][0]
    assert saved['provider'] == 'anthropic'
    assert saved['model'] == 'claude-3'
    assert saved['Speech_provider'] == 'openai_tts'
    assert saved['voice'] == 'alloy'


def test_update_persona_from_form_validates_required_fields(persona_manager):
    manager, personas_dir = persona_manager
    persona = {
        'name': 'Validator',
        'meaning': 'Tests validation',
        'content': {
            'start_locked': 'Intro',
            'editable_content': 'Content',
            'end_locked': 'Outro',
        },
        'provider': 'openai',
        'model': 'gpt-4o',
        'sys_info_enabled': 'False',
        'user_profile_enabled': 'False',
        'type': {'Agent': {'enabled': 'False'}},
        'Speech_provider': '11labs',
        'voice': 'jack',
    }
    file_path = _write_persona(personas_dir, persona)

    general_payload = {
        'name': '',
        'meaning': persona['meaning'],
        'content': persona['content'],
    }
    persona_type_payload = {}
    provider_payload = {'provider': '', 'model': 'gpt-4o'}
    speech_payload = {'Speech_provider': '11labs', 'voice': 'jack'}

    result = manager.update_persona_from_form(
        persona['name'], general_payload, persona_type_payload, provider_payload, speech_payload
    )

    assert result['success'] is False
    assert any('Provider' in error or 'name' in error.lower() for error in result['errors'])

    saved = json.loads(file_path.read_text(encoding='utf-8'))['persona'][0]
    assert saved['provider'] == 'openai'
    assert saved['name'] == 'Validator'
