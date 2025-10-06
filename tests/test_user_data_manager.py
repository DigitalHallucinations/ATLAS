import json
import sys
import types

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

from modules.user_accounts import user_data_manager as user_data_manager_module


class _StubLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _StubConfigManager:
    _app_root_value = '.'

    def __init__(self):
        self._app_root = self._app_root_value

    def get_app_root(self):
        return self._app_root


def test_system_info_lazy_loading_and_caching(monkeypatch):
    SystemInfo = user_data_manager_module.SystemInfo
    UserDataManager = user_data_manager_module.UserDataManager

    monkeypatch.setattr(user_data_manager_module, 'ConfigManager', _StubConfigManager)
    monkeypatch.setattr(user_data_manager_module, 'setup_logger', lambda *_args, **_kwargs: _StubLogger())

    UserDataManager.invalidate_system_info_cache()

    call_count = {'count': 0}

    def fake_run_command(command):
        call_count['count'] += 1
        return f"output for {command}"

    monkeypatch.setattr(SystemInfo, 'run_command', staticmethod(fake_run_command))
    monkeypatch.setattr(UserDataManager, 'get_profile_text', lambda self: 'Profile text', raising=False)
    monkeypatch.setattr(UserDataManager, 'get_emr', lambda self: 'EMR data', raising=False)

    manager_one = UserDataManager('tester')
    assert call_count['count'] == 0

    first_fetch = manager_one.get_system_info()
    assert call_count['count'] == 5

    second_fetch = manager_one.get_system_info()
    assert call_count['count'] == 5
    assert first_fetch == second_fetch

    manager_two = UserDataManager('tester')
    assert call_count['count'] == 5
    assert manager_two.get_system_info() == first_fetch
    assert call_count['count'] == 5

    UserDataManager.invalidate_system_info_cache()

    manager_three = UserDataManager('tester')
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


def test_profile_and_emr_respect_base_directory_override(tmp_path, monkeypatch):
    monkeypatch.setattr(user_data_manager_module, 'ConfigManager', _StubConfigManager)
    monkeypatch.setattr(
        user_data_manager_module,
        'setup_logger',
        lambda *_args, **_kwargs: _StubLogger(),
    )
    monkeypatch.setattr(
        _StubConfigManager,
        '_app_root_value',
        str(tmp_path / 'unused-root'),
    )

    base_dir = tmp_path / 'redirected-base'
    profiles_dir = base_dir / 'user_profiles'
    profiles_dir.mkdir(parents=True)

    profile_data = {
        'Username': 'jordan',
        'Full Name': 'Jordan Example',
    }
    profile_path = profiles_dir / 'jordan.json'
    profile_path.write_text(json.dumps(profile_data), encoding='utf-8')

    emr_path = profiles_dir / 'jordan_emr.txt'
    emr_path.write_text('Line one\nLine two', encoding='utf-8')

    manager = user_data_manager_module.UserDataManager('jordan', base_dir=str(base_dir))

    assert manager.get_profile() == profile_data
    assert manager.get_emr() == 'Line one Line two'
