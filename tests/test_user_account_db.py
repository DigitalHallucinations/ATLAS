import json
import sqlite3
import sys
import types
from pathlib import Path

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

from modules.user_accounts import user_account_db


class _StubLogger:
    def __init__(self):
        self.infos = []

    def info(self, *args, **kwargs):
        self.infos.append((args, kwargs))

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _install_config_manager_stub(monkeypatch, app_root=None):
    class _StubConfigManager:
        def __init__(self):
            self._app_root = app_root

        def get_app_root(self):
            return self._app_root

    monkeypatch.setattr(user_account_db, 'ConfigManager', _StubConfigManager)


def _create_db(tmp_path, monkeypatch):
    _install_config_manager_stub(monkeypatch)
    monkeypatch.setattr(user_account_db, 'setup_logger', lambda *_args, **_kwargs: _StubLogger())

    return user_account_db.UserAccountDatabase(db_name='test_users.db', base_dir=str(tmp_path))


def test_add_user_stores_hashed_password(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    hashed_value = 'hashed-password'

    monkeypatch.setattr(
        user_account_db.UserAccountDatabase,
        '_hash_password',
        staticmethod(lambda _password: hashed_value),
    )

    db.add_user('alice', 'plain-text', 'alice@example.com', 'Alice', '2000-01-01')
    user_record = db.get_user('alice')

    try:
        assert user_record[2] == hashed_value
    finally:
        db.close_connection()


def test_add_user_creates_profile_files(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    try:
        db.add_user('dave', 'Password123', 'dave@example.com', 'Dave', '1995-09-01')

        profile_path = Path(db.user_profiles_dir) / 'dave.json'
        emr_path = Path(db.user_profiles_dir) / 'dave_emr.txt'

        assert profile_path.exists()
        profile_contents = json.loads(profile_path.read_text(encoding='utf-8'))
        assert profile_contents['Username'] == 'dave'
        assert profile_contents['Full Name'] == 'Dave'
        assert profile_contents['DOB'] == '1995-09-01'
        assert profile_contents['Email'] == 'dave@example.com'

        assert emr_path.exists()
        assert emr_path.read_text(encoding='utf-8') == ''
    finally:
        db.close_connection()


def test_verify_user_password_uses_hash_check(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    hashed_value = 'hashed-password'

    monkeypatch.setattr(
        user_account_db.UserAccountDatabase,
        '_hash_password',
        staticmethod(lambda _password: hashed_value),
    )

    db.add_user('bob', 'initial', 'bob@example.com', 'Bob', '1990-05-05')

    verify_calls = []

    def fake_verify(stored_hash, candidate):
        verify_calls.append((stored_hash, candidate))
        return stored_hash == hashed_value and candidate == 'initial'

    monkeypatch.setattr(
        user_account_db.UserAccountDatabase,
        '_verify_password',
        staticmethod(fake_verify),
    )

    try:
        assert db.verify_user_password('bob', 'initial')
        assert verify_calls == [(hashed_value, 'initial')]
        assert not db.verify_user_password('bob', 'wrong')
        assert not db.verify_user_password('unknown', 'initial')
    finally:
        db.close_connection()


def test_update_user_refreshes_profile(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    try:
        db.add_user('erin', 'Password1', 'erin@example.com', 'Erin', '1991-02-03')

        profile_path = Path(db.user_profiles_dir) / 'erin.json'
        original = json.loads(profile_path.read_text(encoding='utf-8'))
        assert original['Full Name'] == 'Erin'
        assert original['DOB'] == '1991-02-03'

        db.update_user('erin', name='Erin Updated', dob='1992-02-03')

        updated = json.loads(profile_path.read_text(encoding='utf-8'))
        assert updated['Full Name'] == 'Erin Updated'
        assert updated['DOB'] == '1992-02-03'
    finally:
        db.close_connection()


def test_update_user_refreshes_profile_on_email_change(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    try:
        db.add_user('frank', 'Password1', 'frank@example.com', 'Frank', '1985-06-07')

        profile_path = Path(db.user_profiles_dir) / 'frank.json'

        db.update_user('frank', email='frank.new@example.com')

        updated = json.loads(profile_path.read_text(encoding='utf-8'))
        assert updated['Email'] == 'frank.new@example.com'
        assert updated['Full Name'] == 'Frank'
        assert updated['DOB'] == '1985-06-07'
    finally:
        db.close_connection()


def test_database_uses_app_root_directory(tmp_path, monkeypatch):
    app_root = tmp_path / 'app-root'
    _install_config_manager_stub(monkeypatch, str(app_root))

    logger = _StubLogger()
    monkeypatch.setattr(user_account_db, 'setup_logger', lambda *_args, **_kwargs: logger)

    db = user_account_db.UserAccountDatabase(db_name='app_root.db')

    try:
        expected_dir = app_root / 'modules' / 'user_accounts' / 'user_profiles'
        assert db.user_profiles_dir == expected_dir
        assert db.db_path == expected_dir / 'app_root.db'
        assert db.db_path.exists()
    finally:
        db.close_connection()


def test_database_migrates_from_legacy_location(tmp_path, monkeypatch):
    app_root = tmp_path / 'new-root'
    _install_config_manager_stub(monkeypatch, str(app_root))

    logger = _StubLogger()
    monkeypatch.setattr(user_account_db, 'setup_logger', lambda *_args, **_kwargs: logger)

    legacy_dir = tmp_path / 'legacy'
    legacy_dir.mkdir()
    monkeypatch.setattr(user_account_db, 'LEGACY_USER_PROFILES_DIR', legacy_dir)

    legacy_db_path = legacy_dir / 'legacy.db'
    sqlite3.connect(str(legacy_db_path)).close()

    db = user_account_db.UserAccountDatabase(db_name='legacy.db')

    try:
        expected_new_path = app_root / 'modules' / 'user_accounts' / 'user_profiles' / 'legacy.db'
        assert db.db_path == expected_new_path
        assert db.db_path.exists()
        assert not legacy_db_path.exists()

        assert any('Migrated user account database' in args[0] for args, _ in logger.infos)
    finally:
        db.close_connection()


def test_delete_user_removes_profile_and_emr(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    try:
        db.add_user('carol', 'Password1', 'carol@example.com', 'Carol', '1980-07-07')

        profile_path = Path(db.user_profiles_dir) / 'carol.json'
        emr_path = Path(db.user_profiles_dir) / 'carol_emr.txt'

        deleted = db.delete_user('carol')

        assert deleted is True
        assert db.get_user('carol') is None
        assert not profile_path.exists()
        assert not emr_path.exists()
    finally:
        db.close_connection()
