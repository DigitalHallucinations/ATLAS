"""Unit tests for :mod:`modules.user_accounts.user_account_service`."""

from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Optional

import sqlite3

import pytest

yaml_stub = types.ModuleType('yaml')
yaml_stub.safe_load = lambda *_args, **_kwargs: {}
sys.modules.setdefault('yaml', yaml_stub)

dotenv_stub = types.ModuleType('dotenv')
dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
dotenv_stub.set_key = lambda *_args, **_kwargs: None
dotenv_stub.find_dotenv = lambda *_args, **_kwargs: ''
sys.modules.setdefault('dotenv', dotenv_stub)

from modules.background_tasks import run_async_in_thread
from modules.user_accounts import user_account_db
from modules.user_accounts import user_account_service


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


class _StubConfigManager:
    def __init__(self):
        self._active_user: Optional[str] = None

    def get_active_user(self) -> Optional[str]:
        return self._active_user

    def set_active_user(self, username: Optional[str]) -> Optional[str]:
        self._active_user = username
        return username


def _install_db_stubs(monkeypatch, app_root=None):
    class _ConfigManagerStub:
        def __init__(self):
            self._app_root = app_root

        def get_app_root(self):
            return self._app_root

    monkeypatch.setattr(user_account_db, 'ConfigManager', _ConfigManagerStub)
    monkeypatch.setattr(user_account_db, 'setup_logger', lambda *_args, **_kwargs: _StubLogger())


def _create_service(tmp_path, monkeypatch):
    _install_db_stubs(monkeypatch)
    monkeypatch.setattr(user_account_service, 'setup_logger', lambda *_args, **_kwargs: _StubLogger())

    database = user_account_db.UserAccountDatabase(db_name='service_users.db', base_dir=str(tmp_path))
    config = _StubConfigManager()
    service = user_account_service.UserAccountService(config_manager=config, database=database)
    return service, config


def test_register_user_persists_account(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        account = service.register_user('alice', 'password123', 'alice@example.com', 'Alice', '1999-01-01')
        assert account.username == 'alice'

        users = service.list_users()
        assert users[0]['username'] == 'alice'

        with pytest.raises(user_account_db.DuplicateUserError):
            service.register_user('alice', 'newpass', 'duplicate@example.com')
    finally:
        service.close()


def test_register_user_duplicate_email(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('eve', 'secret', 'shared@example.com')

        with pytest.raises(user_account_db.DuplicateUserError):
            service.register_user('frank', 'secret', 'shared@example.com')
    finally:
        service.close()


def test_authenticate_user_success_and_failure(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('bob', 'secure', 'bob@example.com')
        assert service.authenticate_user('bob', 'secure') is True
        assert service.authenticate_user('bob', 'wrong') is False
        assert service.authenticate_user('unknown', 'secure') is False
    finally:
        service.close()


def test_set_active_user_tracks_configuration(tmp_path, monkeypatch):
    service, config = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('carol', 'pw', 'carol@example.com')
        service.register_user('dave', 'pw', 'dave@example.com')

        service.set_active_user('carol')
        assert config.get_active_user() == 'carol'

        service.set_active_user('dave')
        assert config.get_active_user() == 'dave'

        service.set_active_user(None)
        assert config.get_active_user() is None

        with pytest.raises(ValueError):
            service.set_active_user('eve')
    finally:
        service.close()


def test_concurrent_database_access_is_serialised(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    def _coroutine_factory(func, /, *args, **kwargs):
        async def _runner():
            return func(*args, **kwargs)

        return _runner()

    try:
        registration_futures = [
            run_async_in_thread(
                lambda index=index: _coroutine_factory(
                    service.register_user,
                    f'user{index}',
                    'password',
                    f'user{index}@example.com',
                )
            )
            for index in range(5)
        ]

        lookup_futures = [
            run_async_in_thread(lambda: _coroutine_factory(service.list_users))
            for _ in range(5)
        ]

        for future in registration_futures + lookup_futures:
            try:
                future.result(timeout=5)
            except (sqlite3.ProgrammingError, sqlite3.OperationalError) as exc:
                pytest.fail(f'Database operation raised threading error: {exc}')

        # Ensure that registrations completed successfully.
        users = service.list_users()
        assert len(users) == 5
    finally:
        service.close()


def test_delete_user_removes_record_and_profile(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('henry', 'pw', 'henry@example.com')
        profile_path = Path(service._database.user_profiles_dir) / 'henry.json'
        profile_path.write_text('{"name": "Henry"}', encoding='utf-8')

        deleted = service.delete_user('henry')

        assert deleted is True
        assert not profile_path.exists()
        assert service.list_users() == []
        assert any(
            args[0] == "Deleted user '%s'" and args[1] == 'henry'
            for args, _kwargs in service.logger.infos
        )
    finally:
        service.close()


def test_delete_unknown_user_returns_false(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        deleted = service.delete_user('nobody')
        assert deleted is False
        assert any(
            args[0] == "No user found with username '%s' to delete" and args[1] == 'nobody'
            for args, _kwargs in service.logger.infos
        )
    finally:
        service.close()


def test_delete_active_user_clears_configuration(tmp_path, monkeypatch):
    service, config = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('ivy', 'pw', 'ivy@example.com')
        service.set_active_user('ivy')

        deleted = service.delete_user('ivy')

        assert deleted is True
        assert config.get_active_user() is None
        assert any(
            args[0] == "Deleted active user '%s' and cleared active user selection"
            and args[1] == 'ivy'
            for args, _kwargs in service.logger.infos
        )
    finally:
        service.close()

