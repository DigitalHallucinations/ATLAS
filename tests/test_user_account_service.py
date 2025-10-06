"""Unit tests for :mod:`modules.user_accounts.user_account_service`."""

from __future__ import annotations

import datetime as _dt
import threading
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
        self.warnings = []

    def info(self, *args, **kwargs):
        self.infos.append((args, kwargs))

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        self.warnings.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.errors.append((args, kwargs))


class _StubConfigManager:
    def __init__(self, overrides: Optional[dict[str, object]] = None):
        self._active_user: Optional[str] = None
        self._overrides = dict(overrides or {})

    def get_active_user(self) -> Optional[str]:
        return self._active_user

    def set_active_user(self, username: Optional[str]) -> Optional[str]:
        self._active_user = username
        return username

    def get_config(self, key: str, default=None):
        return self._overrides.get(key, default)


def _install_db_stubs(monkeypatch, app_root=None):
    class _ConfigManagerStub:
        def __init__(self):
            self._app_root = app_root

        def get_app_root(self):
            return self._app_root

    monkeypatch.setattr(user_account_db, 'ConfigManager', _ConfigManagerStub)
    monkeypatch.setattr(user_account_db, 'setup_logger', lambda *_args, **_kwargs: _StubLogger())


def _create_service(tmp_path, monkeypatch, *, config_overrides=None, clock=None):
    _install_db_stubs(monkeypatch)
    monkeypatch.setattr(user_account_service, 'setup_logger', lambda *_args, **_kwargs: _StubLogger())

    database = user_account_db.UserAccountDatabase(db_name='service_users.db', base_dir=str(tmp_path))
    config = _StubConfigManager(overrides=config_overrides)
    service = user_account_service.UserAccountService(
        config_manager=config,
        database=database,
        clock=clock,
    )
    return service, config


def test_lockout_settings_use_defaults_for_invalid_config(tmp_path, monkeypatch):
    overrides = {
        'ACCOUNT_LOCKOUT_MAX_FAILURES': 'invalid',
        'ACCOUNT_LOCKOUT_WINDOW_SECONDS': {},
        'ACCOUNT_LOCKOUT_DURATION_SECONDS': -10,
    }

    service, _ = _create_service(tmp_path, monkeypatch, config_overrides=overrides)

    try:
        assert service._lockout_threshold == 5
        assert service._lockout_window_seconds == 300
        assert service._lockout_duration_seconds == 300

        warning_keys = [args[1] for args, _kwargs in service.logger.warnings]
        assert warning_keys.count('ACCOUNT_LOCKOUT_MAX_FAILURES') == 1
        assert warning_keys.count('ACCOUNT_LOCKOUT_WINDOW_SECONDS') == 1
        assert warning_keys.count('ACCOUNT_LOCKOUT_DURATION_SECONDS') == 1
    finally:
        service.close()


def test_password_requirements_follow_config_overrides(tmp_path, monkeypatch):
    overrides = {
        'ACCOUNT_PASSWORD_MIN_LENGTH': '6',
        'ACCOUNT_PASSWORD_REQUIRE_UPPERCASE': 'false',
        'ACCOUNT_PASSWORD_REQUIRE_LOWERCASE': 'true',
        'ACCOUNT_PASSWORD_REQUIRE_DIGIT': True,
        'ACCOUNT_PASSWORD_REQUIRE_SYMBOL': 'YES',
        'ACCOUNT_PASSWORD_FORBID_WHITESPACE': 0,
    }

    service, _ = _create_service(tmp_path, monkeypatch, config_overrides=overrides)

    try:
        requirements = service.get_password_requirements()
        assert requirements.min_length == 6
        assert requirements.require_uppercase is False
        assert requirements.require_lowercase is True
        assert requirements.require_digit is True
        assert requirements.require_symbol is True
        assert requirements.forbid_whitespace is False

        description = service.describe_password_requirements()
        assert '6' in description
        assert 'symbol' in description
    finally:
        service.close()


def test_register_user_persists_account(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        account = service.register_user('alice', 'Password123!', 'alice@example.com', 'Alice', '1999-01-01')
        assert account.username == 'alice'
        assert account.last_login is None

        users = service.list_users()
        assert users[0]['username'] == 'alice'
        assert users[0]['display_name'] == 'Alice'

        with pytest.raises(user_account_db.DuplicateUserError):
            service.register_user('alice', 'Newpass1!@', 'duplicate@example.com')
    finally:
        service.close()


def test_register_user_duplicate_email(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('eve', 'Secret12!@', 'shared@example.com')

        with pytest.raises(user_account_db.DuplicateUserError):
            service.register_user('frank', 'Secret12!@', 'shared@example.com')
    finally:
        service.close()


def test_register_user_canonicalises_email_and_rejects_case_duplicates(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        account = service.register_user('caseuser', 'Password123!', 'MiXeD@Example.COM ')
        assert account.email == 'mixed@example.com'

        users = service.list_users()
        assert users[0]['email'] == 'mixed@example.com'

        with pytest.raises(user_account_db.DuplicateUserError):
            service.register_user('caseuser2', 'Password123!', 'MIXED@example.com')
    finally:
        service.close()


def test_register_user_rejects_invalid_email(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        with pytest.raises(ValueError, match='valid email address'):
            service.register_user('invalid', 'Password1!', 'not-an-email')

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
            service.register_user('ab', 'Password1!', 'short@example.com')

        with pytest.raises(ValueError, match='Username must be 3-32 characters'):
            service.register_user('this-username-is-way-too-long-for-the-system', 'Password1!', 'long@example.com')
    finally:
        service.close()


def test_register_user_rejects_invalid_dob(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        with pytest.raises(ValueError, match='YYYY-MM-DD'):
            service.register_user('validuser', 'Password1!', 'valid@example.com', dob='01-01-2000')

        future = (_dt.date.today().replace(year=_dt.date.today().year + 1)).isoformat()
        with pytest.raises(ValueError, match='future'):
            service.register_user('futureuser', 'Password1!', 'future@example.com', dob=future)
    finally:
        service.close()


def test_update_user_validates_and_persists(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('alice', 'Password123!', 'alice@example.com', 'Alice', '1999-01-01')

        updated_account = service.update_user(
            'alice',
            password='Newpass123!',
            current_password='Password123!',
            email='alice.new@example.com',
            name='Alice Updated',
            dob='2000-02-02',
        )

        assert updated_account.email == 'alice.new@example.com'
        assert updated_account.name == 'Alice Updated'
        assert updated_account.dob == '2000-02-02'
        assert updated_account.last_login is None

        # Password is hashed in storage, so authenticate to confirm it changed.
        assert service.authenticate_user('alice', 'Newpass123!') is True

        stored = service.list_users()[0]
        assert stored['email'] == 'alice.new@example.com'
        assert stored['name'] == 'Alice Updated'
        assert stored['dob'] == '2000-02-02'
        assert stored['display_name'] == 'Alice Updated'

        with pytest.raises(ValueError, match='valid email address'):
            service.update_user('alice', email='not-an-email')

        with pytest.raises(ValueError, match='Passwords must'):
            service.update_user('alice', password='short', current_password='Newpass123!')
    finally:
        service.close()


def test_update_user_requires_current_password_for_password_change(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('alice', 'Password123!', 'alice@example.com')

        with pytest.raises(ValueError, match='Current password is required'):
            service.update_user('alice', password='Newpass123!')
    finally:
        service.close()


def test_update_user_rejects_incorrect_current_password(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('alice', 'Password123!', 'alice@example.com')

        with pytest.raises(user_account_service.InvalidCurrentPasswordError):
            service.update_user('alice', password='Newpass123!', current_password='Wrong123')
    finally:
        service.close()


def test_update_user_rejects_duplicate_email_with_case_insensitive_match(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('alpha', 'Password123!', 'user@example.com')
        service.register_user('beta', 'Password123!', 'beta@example.com')

        with pytest.raises(user_account_db.DuplicateUserError):
            service.update_user('beta', email='USER@EXAMPLE.COM')

        updated = service.update_user('beta', email='BETA@EXAMPLE.COM')
        assert updated.email == 'beta@example.com'

        users = {user['username']: user for user in service.list_users()}
        assert users['beta']['email'] == 'beta@example.com'
    finally:
        service.close()


def test_update_user_allows_clearing_optional_fields(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('clearable', 'Password123!', 'clear@example.com', 'Clear Me', '1999-01-01')

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
        with pytest.raises(ValueError, match='Passwords must'):
            service.register_user('weak', 'short', 'weak@example.com')

        with pytest.raises(ValueError, match='Passwords must'):
            service.register_user('weak2', 'longpassword', 'weak2@example.com')

        with pytest.raises(ValueError, match='Passwords must'):
            service.register_user('weak3', 'NoSymbol12', 'weak3@example.com')

        error_messages = [args[0] for args, _kwargs in service.logger.errors]
        assert any('Password failed minimum length requirement' in message for message in error_messages)
        assert any('Password missing uppercase character' in message for message in error_messages)
        assert any('Password missing symbol character' in message for message in error_messages)
        assert service.list_users() == []
    finally:
        service.close()


def test_register_user_rejects_whitespace_symbol(tmp_path, monkeypatch):
    overrides = {'ACCOUNT_PASSWORD_FORBID_WHITESPACE': False}
    service, _ = _create_service(tmp_path, monkeypatch, config_overrides=overrides)

    try:
        with pytest.raises(ValueError, match='Passwords must'):
            service.register_user('spacey', 'Password12 ', 'spacey@example.com')

        error_messages = [args[0] for args, _kwargs in service.logger.errors]
        assert any('Password missing symbol character' in message for message in error_messages)
        assert service.list_users() == []
    finally:
        service.close()


def test_password_requirements_reporting(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        requirements = service.get_password_requirements()
        assert requirements.min_length == 10
        assert requirements.require_uppercase is True
        assert requirements.require_lowercase is True
        assert requirements.require_digit is True
        assert requirements.require_symbol is True
        assert requirements.forbid_whitespace is True

        description = service.describe_password_requirements()
        assert 'at least 10 characters' in description
        assert 'symbol' in description
        assert description.endswith('spaces.')
    finally:
        service.close()


def test_update_user_duplicate_email_raises(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('eve', 'Secret12!@', 'shared@example.com')
        service.register_user('frank', 'Secret12!@', 'frank@example.com')

        with pytest.raises(user_account_db.DuplicateUserError):
            service.update_user('frank', email='shared@example.com')
    finally:
        service.close()


def test_authenticate_user_success_and_failure(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('bob', 'Secure123!', 'bob@example.com')
        details = service.get_user_details('bob')
        assert details['last_login'] is None

        timestamp = '2024-05-20T10:00:00Z'
        monkeypatch.setattr(
            user_account_service.UserAccountService,
            '_current_timestamp',
            staticmethod(lambda: timestamp),
        )

        assert service.authenticate_user('bob', 'Secure123!') is True
        details = service.get_user_details('bob')
        assert details['last_login'] == timestamp
        assert service.list_users()[0]['last_login'] == timestamp

        next_timestamp = '2024-05-21T11:00:00Z'
        monkeypatch.setattr(
            user_account_service.UserAccountService,
            '_current_timestamp',
            staticmethod(lambda: next_timestamp),
        )

        assert service.authenticate_user('bob', 'wrong') is False
        assert service.get_user_details('bob')['last_login'] == timestamp
        assert service.authenticate_user('unknown', 'Secure123!') is False
    finally:
        service.close()


class _TestClock:
    def __init__(self, start: Optional[_dt.datetime] = None):
        self._now = start or _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)

    def __call__(self) -> _dt.datetime:
        return self._now

    def advance(self, seconds: int) -> None:
        self._now = self._now + _dt.timedelta(seconds=seconds)


def test_authenticate_user_lockout_and_success_reset(tmp_path, monkeypatch):
    clock = _TestClock()
    service, _ = _create_service(
        tmp_path,
        monkeypatch,
        config_overrides={
            'ACCOUNT_LOCKOUT_MAX_FAILURES': 2,
            'ACCOUNT_LOCKOUT_WINDOW_SECONDS': 60,
            'ACCOUNT_LOCKOUT_DURATION_SECONDS': 120,
        },
        clock=clock,
    )

    try:
        service.register_user('alice', 'Password123!', 'alice@example.com')

        assert service.authenticate_user('alice', 'wrong') is False
        assert service.authenticate_user('alice', 'wrong') is False

        with pytest.raises(user_account_service.AccountLockedError):
            service.authenticate_user('alice', 'wrong')

        clock.advance(200)
        assert service.authenticate_user('alice', 'Password123!') is True

        assert service.authenticate_user('alice', 'wrong') is False
        assert service.authenticate_user('alice', 'wrong') is False
        with pytest.raises(user_account_service.AccountLockedError):
            service.authenticate_user('alice', 'wrong')
    finally:
        service.close()


def test_authenticate_user_lockout_expires_after_timeout(tmp_path, monkeypatch):
    clock = _TestClock()
    service, _ = _create_service(
        tmp_path,
        monkeypatch,
        config_overrides={
            'ACCOUNT_LOCKOUT_MAX_FAILURES': 2,
            'ACCOUNT_LOCKOUT_WINDOW_SECONDS': 60,
            'ACCOUNT_LOCKOUT_DURATION_SECONDS': 90,
        },
        clock=clock,
    )

    try:
        service.register_user('carol', 'Password123!', 'carol@example.com')

        assert service.authenticate_user('carol', 'nope') is False
        assert service.authenticate_user('carol', 'nope') is False

        with pytest.raises(user_account_service.AccountLockedError):
            service.authenticate_user('carol', 'nope')

        clock.advance(91)
        assert service.authenticate_user('carol', 'nope') is False

        # Lockout should re-trigger after repeated failures post-timeout.
        assert service.authenticate_user('carol', 'nope') is False
        with pytest.raises(user_account_service.AccountLockedError):
            service.authenticate_user('carol', 'nope')
    finally:
        service.close()


def test_authenticate_user_lockout_consistency_with_threads(tmp_path, monkeypatch):
    clock = _TestClock()
    service, _ = _create_service(
        tmp_path,
        monkeypatch,
        config_overrides={
            'ACCOUNT_LOCKOUT_MAX_FAILURES': 2,
            'ACCOUNT_LOCKOUT_WINDOW_SECONDS': 60,
            'ACCOUNT_LOCKOUT_DURATION_SECONDS': 90,
        },
        clock=clock,
    )

    try:
        service.register_user('dave', 'Password123!', 'dave@example.com')

        barrier = threading.Barrier(3)
        results: list[bool] = []
        lockout_errors: list[str] = []

        def _attempt():
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                return

            try:
                results.append(service.authenticate_user('dave', 'wrong'))
            except user_account_service.AccountLockedError:
                lockout_errors.append('locked')

        threads = [threading.Thread(target=_attempt) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results.count(False) == 2
        assert len(lockout_errors) == 1
        assert 'dave' in service._active_lockouts
        assert len(service._failed_attempts.get('dave', [])) == 2

        with pytest.raises(user_account_service.AccountLockedError):
            service.authenticate_user('dave', 'Password123!')
    finally:
        service.close()


def test_search_users_returns_sorted_results(tmp_path, monkeypatch):
    service, _ = _create_service(tmp_path, monkeypatch)

    try:
        service.register_user('alice', 'Password123!', 'alice@example.com', 'Alice', '1990-01-01')
        service.register_user('bob', 'Password123!', 'bob@example.com', 'Bob', '1991-02-02')

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
        service.register_user('carol', 'Password123!', 'carol@example.com', 'Carol', '1988-03-03')

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
        service.register_user('carol', 'Password1!', 'carol@example.com')
        service.register_user('dave', 'Password1!', 'dave@example.com')

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
                    'Password1!',
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
            'Password123!',
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
        service.register_user('henry', 'Password1!', 'henry@example.com')
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
        service.register_user('ivy', 'Password1!', 'ivy@example.com')
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

