import json
import sqlite3
import sys
import types
from datetime import date
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


def test_base_directory_prefers_os_user_data_dir_when_app_root_unwritable(tmp_path, monkeypatch):
    app_root = tmp_path / 'app-root'
    _install_config_manager_stub(monkeypatch, app_root=str(app_root))

    os_data_dir = tmp_path / 'os-data'
    monkeypatch.setattr(user_account_db, '_user_data_dir', lambda *_args, **_kwargs: str(os_data_dir))
    monkeypatch.setattr(user_account_db, 'setup_logger', lambda *_args, **_kwargs: _StubLogger())

    original_ensure = user_account_db.UserAccountDatabase._ensure_writable_directory
    app_base = app_root.resolve() / 'modules' / 'user_accounts'

    def fake_ensure(self, path):
        if path == app_base:
            return None
        return original_ensure(self, path)

    monkeypatch.setattr(user_account_db.UserAccountDatabase, '_ensure_writable_directory', fake_ensure)

    db = user_account_db.UserAccountDatabase(db_name='test_users.db')
    try:
        assert db.user_profiles_dir.parent == os_data_dir
    finally:
        db.close_connection()


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
        db.add_user('dave', 'Password123!', 'dave@example.com', 'Dave', '1995-09-01')

        profile_path = Path(db.user_profiles_dir) / 'dave.json'
        emr_path = Path(db.user_profiles_dir) / 'dave_emr.txt'

        assert profile_path.exists()
        profile_contents = json.loads(profile_path.read_text(encoding='utf-8'))
        assert profile_contents['Username'] == 'dave'
        assert profile_contents['Full Name'] == 'Dave'
        assert profile_contents['DOB'] == '1995-09-01'
        assert profile_contents['Email'] == 'dave@example.com'
        assert profile_contents['Interests']['observations'] == ''
        assert profile_contents['Location']['observations'] == ''
        assert profile_contents['Occupation']['observations'] == ''

        assert emr_path.exists()
        assert emr_path.read_text(encoding='utf-8') == ''
    finally:
        db.close_connection()


def test_existing_database_missing_last_login_column_is_migrated(tmp_path, monkeypatch):
    _install_config_manager_stub(monkeypatch)
    monkeypatch.setattr(user_account_db, 'setup_logger', lambda *_args, **_kwargs: _StubLogger())

    user_profiles_dir = Path(tmp_path) / 'user_profiles'
    user_profiles_dir.mkdir(parents=True)
    db_path = user_profiles_dir / 'test_users.db'

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE user_accounts (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                email TEXT NOT NULL,
                name TEXT,
                DOB TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO user_accounts (username, password, email, name, DOB) VALUES (?, ?, ?, ?, ?)",
            ('legacy', 'hashed', 'legacy@example.com', 'Legacy', '1990-01-01'),
        )
        conn.commit()
    finally:
        conn.close()

    db = user_account_db.UserAccountDatabase(db_name='test_users.db', base_dir=str(tmp_path))
    try:
        columns = [row[1] for row in db.conn.execute('PRAGMA table_info(user_accounts)')]
        assert 'last_login' in columns

        stored = db.get_user('legacy')
        assert stored[6] is None
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


def test_login_attempts_are_recorded(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    try:
        db.add_login_attempt('alice', '2024-05-20T10:00:00Z', True, None)
        db.add_login_attempt('alice', '2024-05-20T11:00:00Z', False, 'invalid-credentials')

        attempts = db.get_login_attempts('alice', limit=5)
        assert [entry['timestamp'] for entry in attempts] == [
            '2024-05-20T11:00:00Z',
            '2024-05-20T10:00:00Z',
        ]
        assert attempts[0]['successful'] is False
        assert attempts[0]['reason'] == 'invalid-credentials'
        assert attempts[1]['successful'] is True
        assert attempts[1]['reason'] is None
    finally:
        db.close_connection()


def test_login_attempt_pruning_keeps_recent_entries(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    try:
        for index in range(5):
            db.add_login_attempt('alice', f'2024-05-20T0{index}:00:00Z', False, 'invalid-credentials')

        db.prune_login_attempts('alice', 2)

        attempts = db.get_login_attempts('alice', limit=10)
        assert len(attempts) == 2
        assert [entry['timestamp'] for entry in attempts] == [
            '2024-05-20T04:00:00Z',
            '2024-05-20T03:00:00Z',
        ]
    finally:
        db.close_connection()


def test_lockout_entry_crud_helpers(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    try:
        first_attempts = ['2024-01-01T00:00:00Z']
        db.set_lockout_entry('locked', first_attempts, '2024-01-01T00:05:00Z')

        stored = db.get_lockout_entry('locked')
        assert stored == (first_attempts, '2024-01-01T00:05:00Z')

        all_entries = db.get_all_lockout_entries()
        assert ('locked', first_attempts, '2024-01-01T00:05:00Z') in all_entries

        updated_attempts = first_attempts + ['2024-01-01T00:02:00Z']
        db.set_lockout_entry('locked', updated_attempts, None)

        stored = db.get_lockout_entry('locked')
        assert stored == (updated_attempts, None)

        assert db.delete_lockout_entry('locked') is True
        assert db.get_lockout_entry('locked') is None
        assert db.delete_lockout_entry('locked') is False
    finally:
        db.close_connection()


def test_update_user_refreshes_profile(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    try:
        db.add_user('erin', 'Password1!', 'erin@example.com', 'Erin', '1991-02-03')

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
        db.add_user('frank', 'Password1!', 'frank@example.com', 'Frank', '1985-06-07')

        profile_path = Path(db.user_profiles_dir) / 'frank.json'

        db.update_user('frank', email='frank.new@example.com')

        updated = json.loads(profile_path.read_text(encoding='utf-8'))
        assert updated['Email'] == 'frank.new@example.com'
        assert updated['Full Name'] == 'Frank'
        assert updated['DOB'] == '1985-06-07'
    finally:
        db.close_connection()


def test_get_username_for_email_is_case_insensitive(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    try:
        db.add_user('gina', 'Password1!', 'gina@example.com', 'Gina', '1993-04-05')

        assert db.get_username_for_email('gina@example.com') == 'gina'
        assert db.get_username_for_email('GINA@EXAMPLE.COM') == 'gina'
        assert db.get_username_for_email('unknown@example.com') is None
    finally:
        db.close_connection()


def test_update_user_preserves_extra_profile_fields(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    try:
        db.add_user('gwen', 'Password1!', 'gwen@example.com', 'Gwen', '1993-04-05')

        profile_path = Path(db.user_profiles_dir) / 'gwen.json'
        profile_contents = json.loads(profile_path.read_text(encoding='utf-8'))
        profile_contents['CustomField'] = 'custom-value'
        profile_contents['Nested'] = {'key': 123}
        profile_path.write_text(json.dumps(profile_contents), encoding='utf-8')

        db.update_user('gwen', name='Gwen Updated')

        updated = json.loads(profile_path.read_text(encoding='utf-8'))
        assert updated['Full Name'] == 'Gwen Updated'
        assert updated['CustomField'] == 'custom-value'
        assert updated['Nested'] == {'key': 123}
    finally:
        db.close_connection()


def test_profile_json_includes_computed_age(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    def _expected_age(dob_str: str) -> int:
        dob_date = date.fromisoformat(dob_str)
        today = date.today()
        computed = today.year - dob_date.year - (
            (today.month, today.day) < (dob_date.month, dob_date.day)
        )
        return max(computed, 0)

    try:
        initial_dob = '2000-01-15'
        db.add_user('henry', 'Password1!', 'henry@example.com', 'Henry', initial_dob)

        profile_path = Path(db.user_profiles_dir) / 'henry.json'
        profile_contents = json.loads(profile_path.read_text(encoding='utf-8'))
        assert isinstance(profile_contents['Age'], int)
        assert profile_contents['Age'] == _expected_age(initial_dob)

        new_dob = '2010-06-30'
        db.update_user('henry', dob=new_dob)

        updated_contents = json.loads(profile_path.read_text(encoding='utf-8'))
        assert isinstance(updated_contents['Age'], int)
        assert updated_contents['Age'] == _expected_age(new_dob)
    finally:
        db.close_connection()


def test_search_users_filters_by_multiple_fields(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    try:
        db.add_user('alice', 'Password1!', 'alice@example.com', 'Alice', '1990-01-01')
        db.add_user('bob', 'Password1!', 'bob@example.com', 'Robert', '1991-02-02')
        db.add_user('carol', 'Password1!', 'carol@example.net', 'Carol', '1992-03-03')

        results = db.search_users('ali')
        assert [row[1] for row in results] == ['alice']

        results = db.search_users('EXAMPLE')
        assert {row[1] for row in results} == {'alice', 'bob', 'carol'}

        results = db.search_users('robert')
        assert [row[1] for row in results] == ['bob']

        results = db.search_users('unknown')
        assert results == []

        results = db.search_users(None)
        assert {row[1] for row in results} == {'alice', 'bob', 'carol'}
    finally:
        db.close_connection()


def test_get_user_details_returns_mapping(tmp_path, monkeypatch):
    db = _create_db(tmp_path, monkeypatch)

    try:
        db.add_user('dave', 'Password1!', 'dave@example.com', 'Dave', '1980-05-05')

        details = db.get_user_details('dave')
        assert details == {
            'id': details['id'],
            'username': 'dave',
            'email': 'dave@example.com',
            'name': 'Dave',
            'dob': '1980-05-05',
            'last_login': None,
        }

        assert db.get_user_details('missing') is None
    finally:
        db.close_connection()


def test_database_uses_app_root_directory(tmp_path, monkeypatch):
    app_root = tmp_path / 'app-root'
    _install_config_manager_stub(monkeypatch, str(app_root))

    logger = _StubLogger()
    monkeypatch.setattr(user_account_db, 'setup_logger', lambda *_args, **_kwargs: logger)
    monkeypatch.setattr(user_account_db, '_user_data_dir', None)

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
    monkeypatch.setattr(user_account_db, '_user_data_dir', None)

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
        db.add_user('carol', 'Password1!', 'carol@example.com', 'Carol', '1980-07-07')

        profile_path = Path(db.user_profiles_dir) / 'carol.json'
        emr_path = Path(db.user_profiles_dir) / 'carol_emr.txt'

        deleted = db.delete_user('carol')

        assert deleted is True
        assert db.get_user('carol') is None
        assert not profile_path.exists()
        assert not emr_path.exists()
    finally:
        db.close_connection()
