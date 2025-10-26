import os
import sys
import types
import dataclasses

import pytest


if 'gi' not in sys.modules:
    gi = types.ModuleType('gi')

    def require_version(_namespace: str, _version: str) -> None:
        return None

    gi.require_version = require_version

    repository = types.ModuleType('gi.repository')
    Gtk = types.ModuleType('Gtk')

    class _BaseWidget:
        def __init__(self, *args, **kwargs):
            self.children = []
            self.visible = True

        def set_application(self, *args, **kwargs):
            return None

        def set_default_size(self, *args, **kwargs):
            return None

        def set_child(self, child):
            self.child = child

        def set_margin_top(self, *args, **kwargs):
            return None

        def set_margin_bottom(self, *args, **kwargs):
            return None

        def set_margin_start(self, *args, **kwargs):
            return None

        def set_margin_end(self, *args, **kwargs):
            return None

        def set_wrap(self, *args, **kwargs):
            return None

        def set_justify(self, *args, **kwargs):
            return None

        def set_use_markup(self, *args, **kwargs):
            return None

        def set_xalign(self, *args, **kwargs):
            return None

        def set_text(self, *args, **kwargs):
            self.text = args[0] if args else ""

        def set_visible(self, value):
            self.visible = bool(value)

        def set_css_classes(self, *args, **kwargs):
            return None

        def append(self, child):
            self.children.append(child)

        def connect(self, *args, **kwargs):
            return None

        def set_sensitive(self, *args, **kwargs):
            return None

        def set_vexpand(self, *args, **kwargs):
            return None

        def set_visible_child_name(self, name):
            self._visible_name = name

        def get_visible_child_name(self):
            return getattr(self, '_visible_name', '')

    class Application:
        pass

    class Window(_BaseWidget):
        def set_title(self, *args, **kwargs):
            return None

    class Box(_BaseWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class Label(_BaseWidget):
        pass

    class Button(_BaseWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class Orientation:
        VERTICAL = 'vertical'
        HORIZONTAL = 'horizontal'

    class Stack(_BaseWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self._pages = {}
            self._visible_name = ''

        def add_titled(self, widget, name, _title):
            self._pages[name] = widget
            if not self._visible_name:
                self._visible_name = name

        def set_visible_child_name(self, name):
            if name in self._pages:
                self._visible_name = name

        def get_visible_child_name(self):
            return self._visible_name

    Gtk.Application = Application
    Gtk.Window = Window
    Gtk.Box = Box
    Gtk.Label = Label
    Gtk.Button = Button
    Gtk.Stack = Stack
    Gtk.Orientation = Orientation

    repository.Gtk = Gtk
    gi.repository = repository
    sys.modules['gi'] = gi
    sys.modules['gi.repository'] = repository
    sys.modules['gi.repository.Gtk'] = Gtk


import ATLAS.config as config_module
from ATLAS.config import ConfigManager
from GTKUI.Setup.setup_wizard import (
    DatabaseState,
    JobSchedulingState,
    MessageBusState,
    ProviderState,
    RetryPolicyState,
    SetupWizardController,
    SpeechState,
    OptionalState,
    UserState,
)


@pytest.fixture
def config_manager(tmp_path, monkeypatch):
    env_file = tmp_path / '.env'
    env_file.write_text('')

    monkeypatch.setenv('OPENAI_API_KEY', 'initial-key')
    monkeypatch.setenv('DEFAULT_PROVIDER', 'OpenAI')
    monkeypatch.setenv('DEFAULT_MODEL', 'gpt-4o')

    recorded = {}

    def fake_set_key(path, key, value):
        recorded[(path, key)] = value

    monkeypatch.setattr(config_module, 'set_key', fake_set_key)
    monkeypatch.setattr(config_module, 'find_dotenv', lambda: str(env_file))
    monkeypatch.setattr(config_module, 'load_dotenv', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ConfigManager,
        '_load_env_config',
        lambda self: {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'DEFAULT_PROVIDER': os.getenv('DEFAULT_PROVIDER', 'OpenAI'),
            'DEFAULT_MODEL': os.getenv('DEFAULT_MODEL', 'gpt-4o'),
            'MISTRAL_API_KEY': None,
            'HUGGINGFACE_API_KEY': None,
            'GOOGLE_API_KEY': None,
            'ANTHROPIC_API_KEY': None,
            'GROK_API_KEY': None,
            'APP_ROOT': tmp_path.as_posix(),
            'OPENAI_BASE_URL': None,
            'OPENAI_ORGANIZATION': None,
        },
    )
    monkeypatch.setattr(ConfigManager, '_load_yaml_config', lambda self: {})

    manager = ConfigManager()
    manager._recorded_set_key = recorded
    manager._env_path = str(env_file)
    return manager


def test_controller_database_step_persists_dsn(config_manager):
    seen = []

    def fake_bootstrap(url):
        seen.append(url)
        return url + '?ensured'

    controller = SetupWizardController(config_manager=config_manager, bootstrap=fake_bootstrap)
    state = dataclasses.replace(
        controller.state.database,
        host='db.example.com',
        port=5433,
        database='atlas_demo',
        user='atlas',
        password='secret',
    )

    ensured = controller.apply_database_settings(state)

    assert seen[-1] == 'postgresql+psycopg://atlas:secret@db.example.com:5433/atlas_demo'
    assert ensured.endswith('?ensured')
    assert config_manager.get_conversation_database_config()['url'] == ensured


def test_controller_job_scheduling_persists_settings(config_manager):
    controller = SetupWizardController(config_manager=config_manager, bootstrap=lambda url: url)
    state = JobSchedulingState(
        enabled=True,
        job_store_url='sqlite:///jobs.sqlite',
        max_workers=8,
        retry_policy=RetryPolicyState(max_attempts=5, backoff_seconds=20.0, jitter_seconds=2.0, backoff_multiplier=1.5),
        timezone='UTC',
        queue_size=128,
    )

    controller.apply_job_scheduling(state)

    settings = config_manager.get_job_scheduling_settings()
    assert settings['enabled'] is True
    assert settings['job_store_url'] == 'sqlite:///jobs.sqlite'
    assert settings['max_workers'] == 8
    assert settings['retry_policy']['max_attempts'] == 5
    assert config_manager.config['task_queue']['jobstore_url'] == 'sqlite:///jobs.sqlite'


def test_controller_message_bus_settings(config_manager):
    controller = SetupWizardController(config_manager=config_manager, bootstrap=lambda url: url)
    state = MessageBusState(backend='redis', redis_url='redis://localhost:6379/0', stream_prefix='atlas')

    controller.apply_message_bus(state)

    messaging = config_manager.get_messaging_settings()
    assert messaging['backend'] == 'redis'
    assert messaging['redis_url'] == 'redis://localhost:6379/0'


def test_controller_provider_settings_persist_keys_and_defaults(config_manager):
    controller = SetupWizardController(config_manager=config_manager, bootstrap=lambda url: url)
    state = ProviderState(
        default_provider='OpenAI',
        default_model='gpt-4o-mini',
        api_keys={'OpenAI': 'sk-new'},
    )

    controller.apply_provider_settings(state)

    assert config_manager.env_config['DEFAULT_MODEL'] == 'gpt-4o-mini'
    assert config_manager.yaml_config['DEFAULT_PROVIDER'] == 'OpenAI'
    env_writes = config_manager._recorded_set_key
    assert any(key == 'OPENAI_API_KEY' for _, key in env_writes)


def test_controller_speech_settings(config_manager, monkeypatch):
    controller = SetupWizardController(config_manager=config_manager, bootstrap=lambda url: url)

    state = SpeechState(
        tts_enabled=True,
        stt_enabled=True,
        default_tts_provider='eleven_labs',
        default_stt_provider='whisper',
        elevenlabs_key='xi-secret',
        openai_key='sk-speech',
        google_credentials='/tmp/google.json',
    )

    controller.apply_speech_settings(state)

    assert config_manager.config['TTS_ENABLED'] is True
    assert config_manager.config['DEFAULT_TTS_PROVIDER'] == 'eleven_labs'
    assert config_manager.yaml_config['DEFAULT_STT_PROVIDER'] == 'whisper'


def test_controller_user_registration(config_manager, monkeypatch):
    controller = SetupWizardController(config_manager=config_manager, bootstrap=lambda url: url)

    class DummyRepository:
        def create_schema(self):
            return None

    controller._get_conversation_repository = lambda: DummyRepository()

    class DummyAccount:
        def __init__(self, username, email, name):
            self.username = username
            self.email = email
            self.name = name

    class DummyService:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def register_user(self, username, password, email, name=None):
            assert password == 'p@ssw0rd'
            return DummyAccount(username, email, name)

    monkeypatch.setattr('GTKUI.Setup.setup_wizard.UserAccountService', DummyService)

    state = UserState(username='admin', email='admin@example.com', password='p@ssw0rd', display_name='Administrator')
    result = controller.register_user(state)

    assert result['username'] == 'admin'
    assert config_manager.get_active_user() == 'admin'


def test_controller_optional_settings(config_manager):
    controller = SetupWizardController(config_manager=config_manager, bootstrap=lambda url: url)
    state = OptionalState(
        tenant_id='tenant-123',
        retention_days=45,
        retention_history_limit=1000,
        scheduler_timezone='US/Eastern',
        scheduler_queue_size=256,
        http_auto_start=True,
    )

    controller.apply_optional_settings(state)

    assert config_manager.config['tenant_id'] == 'tenant-123'
    retention = config_manager.get_conversation_retention_policies()
    assert retention['days'] == 45
    assert retention['history_message_limit'] == 1000
    job_settings = config_manager.get_job_scheduling_settings()
    assert job_settings['timezone'] == 'US/Eastern'
    assert job_settings['queue_size'] == 256
    assert config_manager.config['http_server']['auto_start'] is True
