import json
import sys
import types
from pathlib import Path

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

# Stub out heavy tool modules that introduce circular imports during tests.
if 'modules.Tools' not in sys.modules:
    tools_stub = types.ModuleType('modules.Tools')
    tools_stub.__path__ = []  # mark as package
    sys.modules['modules.Tools'] = tools_stub

if 'modules.Tools.Base_Tools' not in sys.modules:
    base_tools_stub = types.ModuleType('modules.Tools.Base_Tools')
    base_tools_stub.__path__ = []
    sys.modules['modules.Tools.Base_Tools'] = base_tools_stub

# Stub vector_store to prevent deep import chain
if 'modules.Tools.Base_Tools.vector_store' not in sys.modules:
    vector_store_stub = types.ModuleType('modules.Tools.Base_Tools.vector_store')
    
    class _StubQueryMatch:
        pass
    
    class _StubVectorRecord:
        pass
    
    class _StubVectorStoreService:
        def __init__(self, *args, **kwargs):
            pass
    
    vector_store_stub.QueryMatch = _StubQueryMatch
    vector_store_stub.VectorRecord = _StubVectorRecord
    vector_store_stub.VectorStoreService = _StubVectorStoreService
    sys.modules['modules.Tools.Base_Tools.vector_store'] = vector_store_stub

# Stub tool_event_system
if 'modules.Tools.tool_event_system' not in sys.modules:
    tool_event_stub = types.ModuleType('modules.Tools.tool_event_system')
    tool_event_stub.publish_bus_event = lambda *args, **kwargs: None
    sys.modules['modules.Tools.tool_event_system'] = tool_event_stub

# Stub task_queue
if 'modules.Tools.Base_Tools.task_queue' not in sys.modules:
    task_queue_stub = types.ModuleType('modules.Tools.Base_Tools.task_queue')
    
    class _StubBackgroundQueue:
        def __init__(self, *args, **kwargs):
            pass
        def submit(self, *args, **kwargs):
            pass
    
    class _StubTaskQueueService:
        def __init__(self, *args, **kwargs):
            pass
    
    task_queue_stub.BackgroundQueue = _StubBackgroundQueue
    task_queue_stub.TaskQueueService = _StubTaskQueueService
    task_queue_stub.get_background_queue = lambda *args, **kwargs: _StubBackgroundQueue()
    task_queue_stub.get_default_task_queue_service = lambda *args, **kwargs: _StubTaskQueueService()
    sys.modules['modules.Tools.Base_Tools.task_queue'] = task_queue_stub

if 'modules.Tools.Base_Tools.Google_search' not in sys.modules:
    google_search_stub = types.ModuleType('modules.Tools.Base_Tools.Google_search')

    class _StubGoogleSearch:  # pragma: no cover - behavior not exercised
        pass

    google_search_stub.GoogleSearch = _StubGoogleSearch
    sys.modules['modules.Tools.Base_Tools.Google_search'] = google_search_stub

setattr(sys.modules['modules.Tools.Base_Tools'], 'GoogleSearch', getattr(sys.modules['modules.Tools.Base_Tools.Google_search'], 'GoogleSearch'))

if 'ATLAS.config' not in sys.modules:
    atlas_config_stub = types.ModuleType('ATLAS.config')

    class _InitialStubConfigManager:
        def __init__(self):
            self._app_root = ''

        def get_app_root(self):
            return self._app_root

    atlas_config_stub.ConfigManager = _InitialStubConfigManager
    sys.modules['ATLAS.config'] = atlas_config_stub

import core.persona_manager as persona_manager_module
from core.persona_manager import PersonaManager


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

    _copy_schema(tmp_path)
    _write_tool_metadata(tmp_path, ['alpha_tool', 'beta_tool'])
    _write_skill_metadata(
        tmp_path,
        [
            {
                'name': 'analysis_skill',
                'instruction_prompt': 'Analyze with tool',
                'required_tools': ['beta_tool'],
            },
            {
                'name': 'shared_skill',
                'instruction_prompt': 'Optional skill',
                'required_tools': [],
            },
        ],
    )

    class _StubConfigManager:
        def __init__(self):
            self._app_root = str(tmp_path)

        def get_app_root(self):
            return self._app_root

    class _StubUserDataManager:
        profile_text = 'Profile text'
        emr_text = 'EMR data'
        system_info_text = 'System info'
        invalidate_calls = 0

        def __init__(self, _user):
            self._profile_cache = None
            self._emr_cache = None
            self._system_info_cache = None

        def get_profile_text(self):
            if self._profile_cache is None:
                self._profile_cache = self.__class__.profile_text
            return self._profile_cache

        def get_emr(self):
            if self._emr_cache is None:
                self._emr_cache = self.__class__.emr_text
            return self._emr_cache

        def get_system_info(self):
            if self._system_info_cache is None:
                self._system_info_cache = self.__class__.system_info_text
            return self._system_info_cache

        @classmethod
        def invalidate_system_info_cache(cls):
            cls.invalidate_calls += 1

    monkeypatch.setattr(persona_manager_module, 'ConfigManager', _StubConfigManager)
    monkeypatch.setattr(persona_manager_module, 'UserDataManager', _StubUserDataManager)

    master = types.SimpleNamespace(config_manager=_StubConfigManager())
    manager = PersonaManager(master, user='tester', config_manager=master.config_manager)
    manager.persona_base_path = str(personas_dir)
    manager.persona_names = manager.load_persona_names(str(personas_dir))
    manager._test_user_data_manager_cls = _StubUserDataManager
    return manager, personas_dir


def _write_tool_metadata(root: Path, tool_names: list[str]) -> None:
    manifest = root / 'modules' / 'Tools' / 'tool_maps' / 'functions.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            'name': name,
            'description': f'{name} description',
        }
        for name in tool_names
    ]
    manifest.write_text(json.dumps(payload, indent=4), encoding='utf-8')


def _write_skill_metadata(root: Path, entries: list[dict]) -> None:
    manifest = root / 'modules' / 'Skills' / 'skills.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for entry in entries:
        normalized_entry = {
            'name': entry['name'],
            'version': entry.get('version', '1.0.0'),
            'instruction_prompt': entry['instruction_prompt'],
            'required_tools': entry.get('required_tools', []),
            'required_capabilities': entry.get('required_capabilities', []),
            'safety_notes': entry.get('safety_notes', 'Use responsibly.'),
            'summary': entry.get('summary', entry['instruction_prompt']),
            'category': entry.get('category', 'general'),
            'capability_tags': entry.get('capability_tags', []),
        }

        collaboration = entry.get('collaboration')
        if collaboration is not None:
            normalized_entry['collaboration'] = collaboration

        normalized.append(normalized_entry)

    manifest.write_text(json.dumps(normalized, indent=4), encoding='utf-8')


def _copy_schema(root: Path) -> None:
    # Schema is at project_root/modules/Personas/schema.json
    project_root = Path(__file__).resolve().parents[2]
    schema_src = project_root / 'modules' / 'Personas' / 'schema.json'
    schema_dst = root / 'modules' / 'Personas' / 'schema.json'
    schema_dst.parent.mkdir(parents=True, exist_ok=True)
    schema_dst.write_text(schema_src.read_text(encoding='utf-8'), encoding='utf-8')


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


def test_set_allowed_tools_validates_against_metadata(persona_manager):
    manager, personas_dir = persona_manager
    root = personas_dir.parent.parent
    _copy_schema(root)
    _write_tool_metadata(root, ['alpha_tool', 'beta_tool'])

    persona_path = personas_dir / 'ATLAS' / 'Persona' / 'ATLAS.json'

    result = manager.set_allowed_tools('ATLAS', ['alpha_tool'])
    assert result['success'] is True
    assert result['persona']['allowed_tools'] == ['alpha_tool']

    failure = manager.set_allowed_tools('ATLAS', ['alpha_tool', 'invalid_tool'])
    assert failure['success'] is False
    error_text = ' '.join(failure.get('errors', []))
    assert 'invalid_tool' in error_text
    assert 'failed schema validation' in error_text

    saved = json.loads(persona_path.read_text(encoding='utf-8'))['persona'][0]
    assert saved['allowed_tools'] == ['alpha_tool']


def test_set_allowed_skills_rejects_missing_required_tools(persona_manager):
    manager, personas_dir = persona_manager
    persona_path = personas_dir / 'ATLAS' / 'Persona' / 'ATLAS.json'

    assert manager.set_allowed_tools('ATLAS', ['alpha_tool'])['success'] is True

    failure = manager.set_allowed_skills('ATLAS', ['analysis_skill'])
    assert failure['success'] is False
    error_text = ' '.join(failure.get('errors', []))
    assert 'analysis_skill' in error_text
    assert 'beta_tool' in error_text
    assert 'requires missing tools' in error_text

    saved = json.loads(persona_path.read_text(encoding='utf-8'))['persona'][0]
    assert saved.get('allowed_skills') in ([], None)


def test_set_allowed_skills_accepts_when_required_tools_present(persona_manager):
    manager, personas_dir = persona_manager
    persona_path = personas_dir / 'ATLAS' / 'Persona' / 'ATLAS.json'

    assert manager.set_allowed_tools('ATLAS', ['beta_tool'])['success'] is True

    success = manager.set_allowed_skills('ATLAS', ['analysis_skill'])
    assert success['success'] is True

    saved = json.loads(persona_path.read_text(encoding='utf-8'))['persona'][0]
    assert saved['allowed_skills'] == ['analysis_skill']


def test_set_user_refreshes_profile_for_same_user(persona_manager):
    manager, _personas_dir = persona_manager
    stub_cls = manager._test_user_data_manager_cls
    stub_cls.invalidate_calls = 0

    initial_profile = manager.user_data_manager.get_profile_text()
    assert initial_profile == 'Profile text'

    stub_cls.profile_text = 'Updated profile text'

    manager.set_user('tester')

    refreshed_profile = manager.user_data_manager.get_profile_text()
    assert refreshed_profile == 'Updated profile text'
    assert stub_cls.invalidate_calls >= 1


def test_load_persona_names_ignores_non_persona_directories(persona_manager):
    manager, personas_dir = persona_manager

    # Create valid and invalid persona directories
    _write_persona(
        personas_dir,
        {
            'name': 'Helper',
            'meaning': 'Assistive persona',
            'content': {
                'start_locked': 'Intro',
                'editable_content': 'Middle',
                'end_locked': 'Outro',
            },
            'provider': 'openai',
            'model': 'gpt-4o',
            'sys_info_enabled': 'False',
            'user_profile_enabled': 'False',
            'type': {'Agent': {'enabled': 'False'}},
            'Speech_provider': '11labs',
            'voice': 'jack',
        },
    )

    (personas_dir / '__pycache__').mkdir(exist_ok=True)
    (personas_dir / 'Incomplete').mkdir(exist_ok=True)

    persona_names = manager.load_persona_names(str(personas_dir))

    assert 'Helper' in persona_names
    assert 'ATLAS' in persona_names
    assert '__pycache__' not in persona_names
    assert 'Incomplete' not in persona_names


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
