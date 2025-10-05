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

from modules.user_accounts import user_account_db


class _StubLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _StubConfigManager(types.SimpleNamespace):
    pass


def _create_db(tmp_path, monkeypatch):
    monkeypatch.setattr(user_account_db, 'ConfigManager', _StubConfigManager)
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
