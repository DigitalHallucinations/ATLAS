"""Unit tests for :mod:`modules.user_accounts.user_account_service`."""

from __future__ import annotations

import datetime as _dt
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
        self.errors = []

    def info(self, *args, **kwargs):
        self.infos.append((args, kwargs))

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        self.errors.append((args, kwargs))


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
        account = service.register_user('alice', 'Password123', 'alice@example.com', 'Alice', '1999-01-01')
        assert account.username == 'alice'

        users = service.list_users()
        assert users[0]['username'] == 'alice'
        assert users[0]['display_name'] == 'Alice'

        with pytest.raises(user_account_db.DuplicateUserError):
            service.register_user('alice', 'Newpass1', 'duplicate@example.com')
    finally:
        service.close()


def test_register_user_duplicate_email(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('eve', 'Secret12', 'shared@example.com')

        with pytest.raises(user_account_db.DuplicateUserError):
            service.register_user('frank', 'Secret12', 'shared@example.com')
    finally:
        service.close()


def test_register_user_rejects_invalid_email(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        with pytest.raises(ValueError, match='valid email address'):
            service.register_user('invalid', 'Password1', 'not-an-email')

        assert any(
            'Invalid email address provided' in args[0]
            for args, _kwargs in service.logger.errors
        )
        assert service.list_users() == []
    finally:
        service.close()


def test_register_user_rejects_invalid_username(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        with pytest.raises(ValueError, match='Username must be 3-32 characters'):
            service.register_user('ab', 'Password1', 'short@example.com')

        with pytest.raises(ValueError, match='Username must be 3-32 characters'):
            service.register_user('this-username-is-way-too-long-for-the-system', 'Password1', 'long@example.com')
    finally:
        service.close()


def test_register_user_rejects_invalid_dob(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        with pytest.raises(ValueError, match='YYYY-MM-DD'):
            service.register_user('validuser', 'Password1', 'valid@example.com', dob='01-01-2000')

        future = (_dt.date.today().replace(year=_dt.date.today().year + 1)).isoformat()
        with pytest.raises(ValueError, match='future'):
            service.register_user('futureuser', 'Password1', 'future@example.com', dob=future)
    finally:
        service.close()


def test_update_user_validates_and_persists(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('alice', 'Password123', 'alice@example.com', 'Alice', '1999-01-01')

        updated_account = service.update_user(
            'alice',
            password='Newpass123',
            email='alice.new@example.com',
            name='Alice Updated',
            dob='2000-02-02',
        )

        assert updated_account.email == 'alice.new@example.com'
        assert updated_account.name == 'Alice Updated'
        assert updated_account.dob == '2000-02-02'

        # Password is hashed in storage, so authenticate to confirm it changed.
        assert service.authenticate_user('alice', 'Newpass123') is True

        stored = service.list_users()[0]
        assert stored['email'] == 'alice.new@example.com'
        assert stored['name'] == 'Alice Updated'
        assert stored['dob'] == '2000-02-02'
        assert stored['display_name'] == 'Alice Updated'

        with pytest.raises(ValueError, match='valid email address'):
            service.update_user('alice', email='not-an-email')

        with pytest.raises(ValueError, match='Password must be at least 8 characters'):
            service.update_user('alice', password='short')
    finally:
        service.close()


def test_update_user_allows_clearing_optional_fields(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('clearable', 'Password123', 'clear@example.com', 'Clear Me', '1999-01-01')

        updated = service.update_user('clearable', name='', dob='')
        assert updated.name is None
        assert updated.dob is None

        stored = service.list_users()[0]
        assert stored['name'] is None
        assert stored['dob'] is None
        assert stored['display_name'] == 'clearable'
    finally:
        service.close()


def test_register_user_rejects_weak_password(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        with pytest.raises(ValueError, match='Password must be at least 8 characters'):
            service.register_user('weak', 'short', 'weak@example.com')

        with pytest.raises(ValueError, match='Password must be at least 8 characters'):
            service.register_user('weak2', 'longpassword', 'weak2@example.com')

        assert any(
            'Password failed minimum length requirement' in args[0]
            for args, _kwargs in service.logger.errors
        )
        assert any(
            'Password missing required character diversity' in args[0]
            for args, _kwargs in service.logger.errors
        )
        assert service.list_users() == []
    finally:
        service.close()


def test_update_user_duplicate_email_raises(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('eve', 'Secret12', 'shared@example.com')
        service.register_user('frank', 'Secret12', 'frank@example.com')

        with pytest.raises(user_account_db.DuplicateUserError):
            service.update_user('frank', email='shared@example.com')
    finally:
        service.close()


def test_authenticate_user_success_and_failure(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('bob', 'Secure123', 'bob@example.com')
        assert service.authenticate_user('bob', 'Secure123') is True
        assert service.authenticate_user('bob', 'wrong') is False
        assert service.authenticate_user('unknown', 'Secure123') is False
    finally:
        service.close()


def test_search_users_returns_sorted_results(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('alice', 'Password123', 'alice@example.com', 'Alice', '1990-01-01')
        service.register_user('bob', 'Password123', 'bob@example.com', 'Bob', '1991-02-02')

        results = service.search_users('bo')
        assert [item['username'] for item in results] == ['bob']
        assert results[0]['display_name'] == 'Bob'

        results = service.search_users('example')
        assert [item['username'] for item in results] == ['alice', 'bob']

        results = service.search_users('unknown')
        assert results == []
    finally:
        service.close()


def test_get_user_details_returns_mapping(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('carol', 'Password123', 'carol@example.com', 'Carol', '1988-03-03')

        details = service.get_user_details('carol')
        assert details['username'] == 'carol'
        assert details['email'] == 'carol@example.com'
        assert details['name'] == 'Carol'
        assert service.get_user_details('missing') is None
    finally:
        service.close()


def test_set_active_user_tracks_configuration(tmp_path, monkeypatch):
    service, config = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('carol', 'Password1', 'carol@example.com')
        service.register_user('dave', 'Password1', 'dave@example.com')

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
                    'Password1',
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


def test_run_async_in_thread_accepts_sync_callables(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        account_future = run_async_in_thread(
            service.register_user,
            'alice',
            'Password123',
            'alice@example.com',
        )
        account = account_future.result(timeout=5)
        assert account.username == 'alice'

        users_future = run_async_in_thread(service.list_users)
        users = users_future.result(timeout=5)
        assert users and users[0]['username'] == 'alice'
    finally:
        service.close()


def test_delete_user_removes_record_and_profile(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('henry', 'Password1', 'henry@example.com')
        profile_path = Path(service._database.user_profiles_dir) / 'henry.json'
        profile_path.write_text('{"name": "Henry"}', encoding='utf-8')
        emr_path = Path(service._database.user_profiles_dir) / 'henry_emr.txt'
        emr_path.write_text('EMR data', encoding='utf-8')

        deleted = service.delete_user('henry')

        assert deleted is True
        assert not profile_path.exists()
        assert not emr_path.exists()
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
        service.register_user('ivy', 'Password1', 'ivy@example.com')
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

