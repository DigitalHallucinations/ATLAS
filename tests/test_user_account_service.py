"""Unit tests for :mod:`modules.user_accounts.user_account_service`."""

from __future__ import annotations

from typing import Optional

import pytest

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

        with pytest.raises(ValueError):
            service.register_user('alice', 'newpass', 'duplicate@example.com')
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

