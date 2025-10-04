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
    def __init__(self):
        self._app_root = '.'

    def get_app_root(self):
        return self._app_root


def test_system_info_commands_cached(monkeypatch):
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
    assert call_count['count'] == 5

    manager_one.get_system_info()
    assert call_count['count'] == 5

    manager_two = UserDataManager('tester')
    assert call_count['count'] == 5

    manager_two.get_system_info()
    assert call_count['count'] == 5

    UserDataManager.invalidate_system_info_cache()

    manager_three = UserDataManager('tester')
    assert call_count['count'] == 10

    UserDataManager.invalidate_system_info_cache()
