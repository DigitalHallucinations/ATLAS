import sys
import types
from typing import Dict

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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

from modules.conversation_store import Base, ConversationStoreRepository
from modules.user_accounts import user_data_manager as user_data_manager_module


class _StubLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class _StubConfigManager:
    def __init__(self, factory):
        self._factory = factory

    def get_conversation_store_session_factory(self):
        return self._factory

    def ensure_postgres_conversation_store(self, **_kwargs):
        return None

    def get_conversation_retention_policies(self) -> Dict[str, object]:
        return {}


@pytest.fixture
def session_factory(postgresql):
    engine = create_engine(postgresql.dsn(), future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    yield factory
    engine.dispose()


@pytest.fixture
def repository(session_factory):
    repo = ConversationStoreRepository(session_factory)
    repo.create_schema()
    return repo


def test_system_info_lazy_loading_and_caching(repository, session_factory, monkeypatch):
    SystemInfo = user_data_manager_module.SystemInfo
    UserDataManager = user_data_manager_module.UserDataManager

    monkeypatch.setattr(
        user_data_manager_module,
        'setup_logger',
        lambda *_args, **_kwargs: _StubLogger(),
    )

    UserDataManager.invalidate_system_info_cache()

    call_count = {'count': 0}

    def fake_run_command(command):
        call_count['count'] += 1
        return f"output for {command}"

    monkeypatch.setattr(SystemInfo, 'run_command', staticmethod(fake_run_command))

    manager_one = UserDataManager(
        'tester',
        repository=repository,
        config_manager=_StubConfigManager(session_factory),
    )
    assert call_count['count'] == 0

    first_fetch = manager_one.get_system_info()
    assert call_count['count'] == 5

    second_fetch = manager_one.get_system_info()
    assert call_count['count'] == 5
    assert first_fetch == second_fetch

    manager_two = UserDataManager(
        'tester',
        repository=repository,
        config_manager=_StubConfigManager(session_factory),
    )
    assert call_count['count'] == 5
    assert manager_two.get_system_info() == first_fetch
    assert call_count['count'] == 5

    UserDataManager.invalidate_system_info_cache()

    manager_three = UserDataManager(
        'tester',
        repository=repository,
        config_manager=_StubConfigManager(session_factory),
    )
    assert call_count['count'] == 5
    manager_three.get_system_info()
    assert call_count['count'] == 10

    UserDataManager.invalidate_system_info_cache()


def test_get_memory_info_handles_missing_capacity(monkeypatch):
    SystemInfo = user_data_manager_module.SystemInfo

    monkeypatch.setattr(user_data_manager_module.platform, 'system', lambda: 'Windows')

    sample_output = """BankLabel=NODE 0 DIMM 0\nCapacity=17179869184\n\nBankLabel=NODE 0 DIMM 1\nManufacturer=Example"""

    monkeypatch.setattr(SystemInfo, 'run_command', staticmethod(lambda _command: sample_output))

    result = SystemInfo.get_memory_info()

    assert result == "Total Physical Memory: 16.00 GB"


def test_get_memory_info_handles_no_capacity_data(monkeypatch):
    SystemInfo = user_data_manager_module.SystemInfo

    monkeypatch.setattr(user_data_manager_module.platform, 'system', lambda: 'Windows')

    sample_output = """BankLabel=NODE 0 DIMM 0\nManufacturer=Example"""

    monkeypatch.setattr(SystemInfo, 'run_command', staticmethod(lambda _command: sample_output))

    result = SystemInfo.get_memory_info()

    assert result == "Total Physical Memory: Unknown"


def test_missing_profile_creates_default_record(repository, session_factory, monkeypatch):
    monkeypatch.setattr(
        user_data_manager_module,
        'setup_logger',
        lambda *_args, **_kwargs: _StubLogger(),
    )

    manager = user_data_manager_module.UserDataManager(
        'jordan',
        repository=repository,
        config_manager=_StubConfigManager(session_factory),
    )

    profile = manager.get_profile()
    assert profile['Username'] == 'jordan'

    stored = repository.get_user_profile('jordan')
    assert stored is not None
    assert stored['profile']['Username'] == 'jordan'


def test_profile_round_trip_through_repository(repository, session_factory, monkeypatch):
    monkeypatch.setattr(
        user_data_manager_module,
        'setup_logger',
        lambda *_args, **_kwargs: _StubLogger(),
    )

    repository.upsert_user_profile(
        'jordan',
        {
            'Username': 'jordan',
            'Full Name': 'Jordan Example',
        },
        display_name='Jordan Example',
    )

    manager = user_data_manager_module.UserDataManager(
        'jordan',
        repository=repository,
        config_manager=_StubConfigManager(session_factory),
    )

    profile = manager.get_profile()
    assert profile['Full Name'] == 'Jordan Example'

    repository.upsert_user_profile(
        'jordan',
        {
            'Username': 'jordan',
            'Full Name': 'Jordan Updated',
        },
        display_name='Jordan Updated',
    )

    manager = user_data_manager_module.UserDataManager(
        'jordan',
        repository=repository,
        config_manager=_StubConfigManager(session_factory),
    )

    refreshed = manager.get_profile()
    assert refreshed['Full Name'] == 'Jordan Updated'


def test_emr_round_trip_through_repository(repository, session_factory, monkeypatch):
    monkeypatch.setattr(
        user_data_manager_module,
        'setup_logger',
        lambda *_args, **_kwargs: _StubLogger(),
    )

    repository.upsert_user_profile(
        'jordan',
        {'Username': 'jordan'},
        documents={'emr': 'Line one\nLine two'},
    )

    manager = user_data_manager_module.UserDataManager(
        'jordan',
        repository=repository,
        config_manager=_StubConfigManager(session_factory),
    )

    first = manager.get_emr()
    assert first == 'Line one Line two'
    second = manager.get_emr()
    assert second == first
