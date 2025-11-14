import dataclasses
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import importlib
import pytest

sqlalchemy_module = types.ModuleType("sqlalchemy")
sqlalchemy_engine_module = types.ModuleType("sqlalchemy.engine")
sqlalchemy_url_module = types.ModuleType("sqlalchemy.engine.url")
sqlalchemy_orm_module = types.ModuleType("sqlalchemy.orm")
sqlalchemy_module.__path__ = []  # mark as package
sqlalchemy_engine_module.__path__ = []
sqlalchemy_url_module.URL = object
sqlalchemy_url_module.make_url = lambda value: value
sqlalchemy_engine_module.url = sqlalchemy_url_module
sqlalchemy_engine_module.Engine = object
sqlalchemy_module.engine = sqlalchemy_engine_module
sqlalchemy_module.create_engine = lambda *args, **kwargs: types.SimpleNamespace(dispose=lambda: None)
sqlalchemy_module.inspect = lambda *args, **kwargs: types.SimpleNamespace(get_table_names=lambda: [])
sqlalchemy_module.Column = lambda *args, **kwargs: None
sqlalchemy_module.DateTime = lambda *args, **kwargs: object
sqlalchemy_module.Enum = lambda *args, **kwargs: None
sqlalchemy_module.ForeignKey = lambda *args, **kwargs: None
sqlalchemy_module.Index = lambda *args, **kwargs: None
sqlalchemy_module.Integer = int
sqlalchemy_module.String = str
sqlalchemy_module.Text = str
sqlalchemy_module.UniqueConstraint = lambda *args, **kwargs: None
sqlalchemy_module.Boolean = bool
sqlalchemy_module.Float = float
sqlalchemy_module.text = lambda value: value
sqlalchemy_module.JSON = dict
sqlalchemy_module.MetaData = lambda *args, **kwargs: object


class _StubTable:
    def __init__(self, *_args, **_kwargs):
        self.c = types.SimpleNamespace(expires_at=object())


sqlalchemy_module.Table = lambda *args, **kwargs: _StubTable()
sqlalchemy_module.delete = lambda *args, **kwargs: None
sqlalchemy_module.select = lambda *args, **kwargs: None


class _SQLAlchemyFunc:
    def __getattr__(self, _name):  # pragma: no cover - simple stub
        return lambda *args, **kwargs: None


sqlalchemy_module.func = _SQLAlchemyFunc()
sqlalchemy_orm_module.sessionmaker = lambda *args, **kwargs: None
sqlalchemy_orm_module.relationship = lambda *args, **kwargs: None
sqlalchemy_orm_module.declarative_base = lambda *args, **kwargs: object
sqlalchemy_module.orm = sqlalchemy_orm_module
sys.modules["sqlalchemy"] = sqlalchemy_module
sys.modules["sqlalchemy.engine"] = sqlalchemy_engine_module
sys.modules["sqlalchemy.engine.url"] = sqlalchemy_url_module
sys.modules["sqlalchemy.orm"] = sqlalchemy_orm_module

config_module = types.ModuleType("ATLAS.config")
config_module.__path__ = []
_real_config_module = importlib.import_module("ATLAS.config")
config_module.ConfigManager = type("ConfigManager", (), {"UNSET": object()})
config_module._DEFAULT_CONVERSATION_STORE_DSN_BY_BACKEND = {
    "postgresql": "postgresql+psycopg://atlas@localhost:5432/atlas",
}
config_module._DEFAULT_CONVERSATION_STORE_BACKENDS = (
    types.SimpleNamespace(
        name="postgresql",
        dsn="postgresql+psycopg://atlas@localhost:5432/atlas",
        dialect="postgresql",
    ),
)
config_module.get_default_conversation_store_backends = (
    lambda: config_module._DEFAULT_CONVERSATION_STORE_BACKENDS
)
config_module.infer_conversation_store_backend = lambda value: "postgresql"
sys.modules["ATLAS.config"] = config_module
sys.modules["ATLAS.config.config_manager"] = _real_config_module.config_manager

bootstrap_module = types.ModuleType("modules.conversation_store.bootstrap")
bootstrap_module.BootstrapError = Exception
bootstrap_module.bootstrap_conversation_store = lambda dsn: dsn
sys.modules["modules.conversation_store.bootstrap"] = bootstrap_module

repository_module = types.ModuleType("modules.conversation_store.repository")

class _ConversationStoreRepository:
    def __init__(self, *args, **kwargs):
        pass

    def create_schema(self):
        return None


repository_module.ConversationStoreRepository = _ConversationStoreRepository
sys.modules["modules.conversation_store.repository"] = repository_module

user_account_module = types.ModuleType("modules.user_accounts.user_account_service")


class _UserAccountService:
    def __init__(self, *args, **kwargs):
        pass

    def register_user(
        self,
        *,
        username,
        password,
        email,
        name=None,
        dob=None,
        full_name=None,
        domain=None,
        tenant_id=None,
    ):
        return types.SimpleNamespace(
            username=username,
            password=password,
            email=email,
            name=name,
            tenant_id=tenant_id,
        )


user_account_module.UserAccountService = _UserAccountService
sys.modules["modules.user_accounts.user_account_service"] = user_account_module

from ATLAS.setup import BootstrapError, OptionalState, UserState
from ATLAS.setup.cli import SetupUtility
from ATLAS.setup.controller import (
    AdminProfile,
    DatabaseState,
    JobSchedulingState,
    KvStoreState,
    MessageBusState,
    VectorStoreState,
    PrivilegedCredentialState,
    ProviderState,
    SetupTypeState,
    SpeechState,
)


class DummyController:
    def __init__(self) -> None:
        self.state = types.SimpleNamespace(
            database=DatabaseState(),
            kv_store=KvStoreState(),
            job_scheduling=JobSchedulingState(),
            message_bus=MessageBusState(),
            vector_store=VectorStoreState(),
            providers=ProviderState(),
            speech=SpeechState(),
            user=UserState(),
            optional=OptionalState(),
            setup_type=SetupTypeState(),
        )
        self._vector_settings = {"default_adapter": "in_memory", "adapters": {"in_memory": {}}}
        self.config_manager = types.SimpleNamespace(
            get_vector_store_settings=self._get_vector_store_settings,
            set_vector_store_settings=self._set_vector_store_settings,
        )
        self.applied_database_states: list[DatabaseState] = []
        self.privileged_credentials: list[tuple[str | None, str | None] | None] = []
        self.bootstrap_error_sequence: list[Exception] = []
        self.registered_users: list[UserState] = []
        self.staged_profiles: list[AdminProfile] = []
        self._privileged_credentials: tuple[str | None, str | None] | None = None
        self.set_privileged_credentials_calls: list[tuple[str | None, str | None] | None] = []
        self.applied_setup_modes: list[str] = []

    def apply_database_settings(
        self,
        state: DatabaseState,
        *,
        privileged_credentials: tuple[str | None, str | None] | None = None,
    ) -> str:
        self.applied_database_states.append(dataclasses.replace(state))
        self.privileged_credentials.append(privileged_credentials)
        if self.bootstrap_error_sequence:
            exc = self.bootstrap_error_sequence.pop(0)
            raise exc
        ensured = (
            f"postgresql+psycopg://{state.user}:{state.password}@{state.host}:{state.port}/{state.database}"
        )
        self.state.database = dataclasses.replace(state, dsn=ensured)
        return ensured

    def apply_vector_store_settings(self, state: VectorStoreState):
        self.state.vector_store = dataclasses.replace(state)
        self._set_vector_store_settings(default_adapter=state.adapter)
        return self._vector_settings

    def apply_kv_store_settings(self, state: KvStoreState):  # pragma: no cover - not needed for these tests
        self.state.kv_store = dataclasses.replace(state)
        return {}

    def apply_job_scheduling(self, state: JobSchedulingState):  # pragma: no cover - not needed
        self.state.job_scheduling = dataclasses.replace(state)
        return {}

    def apply_message_bus(self, state: MessageBusState):  # pragma: no cover - not needed
        self.state.message_bus = dataclasses.replace(state)
        return {}

    def apply_provider_settings(self, state: ProviderState):  # pragma: no cover - not needed
        self.state.providers = dataclasses.replace(state)
        return self.state.providers

    def apply_speech_settings(self, state: SpeechState):  # pragma: no cover - not needed
        self.state.speech = dataclasses.replace(state)
        return self.state.speech

    def apply_optional_settings(self, state: OptionalState):  # pragma: no cover - not needed
        self.state.optional = dataclasses.replace(state)
        return self.state.optional

    def get_conversation_backend_options(self):  # pragma: no cover - not needed for most tests
        return (
            types.SimpleNamespace(name="postgresql"),
            types.SimpleNamespace(name="sqlite"),
        )

    def apply_setup_type(self, mode: str) -> SetupTypeState:
        normalized = (mode or "").strip().lower()
        self.applied_setup_modes.append(normalized)
        if normalized == "personal":
            self.state.message_bus = dataclasses.replace(
                self.state.message_bus,
                backend="in_memory",
                redis_url=None,
                stream_prefix=None,
            )
            current_job = self.state.job_scheduling
            self.state.job_scheduling = dataclasses.replace(
                current_job,
                enabled=False,
                job_store_url=None,
                max_workers=None,
                retry_policy=dataclasses.replace(current_job.retry_policy),
                timezone=None,
                queue_size=None,
            )
            self.state.kv_store = dataclasses.replace(
                self.state.kv_store,
                reuse_conversation_store=True,
                url=None,
            )
            self.state.optional = dataclasses.replace(
                self.state.optional,
                retention_days=None,
                retention_history_limit=None,
                http_auto_start=True,
            )
            setup_state = SetupTypeState(mode="personal", applied=True)
        elif normalized == "enterprise":
            current_message_bus = self.state.message_bus
            redis_url = current_message_bus.redis_url or "redis://localhost:6379/0"
            stream_prefix = current_message_bus.stream_prefix or "atlas"
            self.state.message_bus = dataclasses.replace(
                current_message_bus,
                backend="redis",
                redis_url=redis_url,
                stream_prefix=stream_prefix,
            )
            current_job = self.state.job_scheduling
            job_store_url = current_job.job_store_url or (
                "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_jobs"
            )
            self.state.job_scheduling = dataclasses.replace(
                current_job,
                enabled=True,
                job_store_url=job_store_url,
                max_workers=current_job.max_workers or 4,
                retry_policy=dataclasses.replace(current_job.retry_policy),
                timezone=current_job.timezone or "UTC",
                queue_size=current_job.queue_size or 100,
            )
            self.state.kv_store = dataclasses.replace(
                self.state.kv_store,
                reuse_conversation_store=False,
                url=self.state.kv_store.url
                or "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_cache",
            )
            self.state.optional = dataclasses.replace(
                self.state.optional,
                retention_days=30,
                retention_history_limit=500,
                http_auto_start=False,
            )
            setup_state = SetupTypeState(mode="enterprise", applied=True)
        else:
            fallback_mode = normalized or "custom"
            setup_state = SetupTypeState(mode=fallback_mode, applied=False)

        self.state.setup_type = dataclasses.replace(setup_state)
        return self.state.setup_type

    def register_user(self, state: UserState):
        self.registered_users.append(dataclasses.replace(state))
        self.state.user = dataclasses.replace(state)
        return {
            "username": state.username,
            "email": state.email,
            "display_name": state.display_name,
            "full_name": state.full_name or None,
            "domain": state.domain or None,
            "date_of_birth": state.date_of_birth or None,
        }

    def set_user_profile(self, profile: AdminProfile) -> UserState:
        self.staged_profiles.append(dataclasses.replace(profile))
        privileged_state = PrivilegedCredentialState(
            sudo_username=profile.sudo_username or "",
            sudo_password=profile.sudo_password or "",
        )
        self.state.user = UserState(
            username=profile.username or "",
            email=profile.email or "",
            password=profile.password or "",
            display_name=profile.display_name or "",
            full_name=profile.full_name or "",
            domain=profile.domain or "",
            date_of_birth=profile.date_of_birth or "",
            privileged_credentials=privileged_state,
        )
        db_username = profile.privileged_db_username or None
        db_password = profile.privileged_db_password or None
        if db_username or db_password:
            self.set_privileged_credentials((db_username, db_password))
        return self.state.user

    def set_privileged_credentials(self, credentials):
        self.set_privileged_credentials_calls.append(credentials)
        if credentials is None:
            self._privileged_credentials = None
        else:
            username, password = credentials
            cleaned_username = username or None
            cleaned_password = password or None
            if cleaned_username is None and cleaned_password is None:
                self._privileged_credentials = None
            else:
                self._privileged_credentials = (cleaned_username, cleaned_password)

    def get_privileged_credentials(self):
        return self._privileged_credentials

    def build_summary(self):  # pragma: no cover - not needed
        return {}

    def _get_vector_store_settings(self):
        return dict(self._vector_settings)

    def _set_vector_store_settings(self, *, default_adapter: str, adapter_settings=None):
        normalized = (default_adapter or "in_memory").strip().lower()
        adapters = dict(self._vector_settings.get("adapters", {}))
        adapters.setdefault(normalized, {} if adapter_settings is None else dict(adapter_settings))
        adapters.setdefault("in_memory", adapters.get("in_memory", {}))
        self._vector_settings = {"default_adapter": normalized, "adapters": adapters}
        return dict(self._vector_settings)


def test_choose_setup_type_defaults_to_personal_settings():
    controller = DummyController()
    controller.state.message_bus = dataclasses.replace(
        controller.state.message_bus,
        backend="redis",
        redis_url="redis://example",
        stream_prefix="custom",
    )
    controller.state.job_scheduling = dataclasses.replace(
        controller.state.job_scheduling,
        enabled=True,
        job_store_url="postgresql+psycopg://other",
        max_workers=8,
        timezone="UTC",
        queue_size=50,
    )
    controller.state.kv_store = dataclasses.replace(
        controller.state.kv_store,
        reuse_conversation_store=False,
        url="postgresql+psycopg://cache",
    )
    controller.state.optional = dataclasses.replace(
        controller.state.optional,
        retention_days=10,
        retention_history_limit=100,
        http_auto_start=False,
    )

    utility = SetupUtility(
        controller=controller,
        input_func=lambda prompt: "",  # accept default personal
        getpass_func=lambda prompt: "",
        print_func=lambda message: None,
    )

    result = utility.choose_setup_type()

    assert controller.applied_setup_modes == ["personal"]
    assert result.mode == "personal"
    assert controller.state.message_bus.backend == "in_memory"
    assert controller.state.message_bus.redis_url is None
    assert controller.state.job_scheduling.enabled is False
    assert controller.state.job_scheduling.job_store_url is None
    assert controller.state.kv_store.reuse_conversation_store is True
    assert controller.state.kv_store.url is None
    assert controller.state.optional.http_auto_start is True


def test_choose_setup_type_switches_to_enterprise_defaults():
    controller = DummyController()

    utility = SetupUtility(
        controller=controller,
        input_func=lambda prompt: "enterprise",
        getpass_func=lambda prompt: "",
        print_func=lambda message: None,
    )

    result = utility.choose_setup_type()

    assert controller.applied_setup_modes == ["enterprise"]
    assert result.mode == "enterprise"
    assert controller.state.message_bus.backend == "redis"
    assert controller.state.message_bus.redis_url == "redis://localhost:6379/0"
    assert controller.state.job_scheduling.enabled is True
    assert (
        controller.state.job_scheduling.job_store_url
        == "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_jobs"
    )
    assert controller.state.job_scheduling.max_workers == 4
    assert controller.state.job_scheduling.timezone == "UTC"
    assert controller.state.job_scheduling.queue_size == 100
    assert controller.state.kv_store.reuse_conversation_store is False
    assert (
        controller.state.kv_store.url
        == "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_cache"
    )
    assert controller.state.optional.http_auto_start is False


def test_configure_database_persists_dsn():
    controller = DummyController()
    responses = iter(["", "db.internal", "6543", "atlas_prod", "atlas_user"])
    passwords = iter(["s3cret"])

    utility = SetupUtility(
        controller=controller,
        input_func=lambda prompt: next(responses),
        getpass_func=lambda prompt: next(passwords),
        print_func=lambda message: None,
    )

    dsn = utility.configure_database()

    applied = controller.applied_database_states[-1]
    assert applied.host == "db.internal"
    assert applied.port == 6543
    assert controller.state.database.dsn == dsn
    assert dsn.endswith("@db.internal:6543/atlas_prod")


def test_configure_database_handles_missing_role_with_privileged_credentials():
    controller = DummyController()
    controller.state.database = dataclasses.replace(controller.state.database, password="stored")
    controller.bootstrap_error_sequence = [BootstrapError('role "atlas" does not exist')]

    responses = iter(["", "", "", "", "", "y", "postgres"])
    passwords = iter(["!clear!", "supersecret"])

    utility = SetupUtility(
        controller=controller,
        input_func=lambda prompt: next(responses),
        getpass_func=lambda prompt: next(passwords),
        print_func=lambda message: None,
    )

    dsn = utility.configure_database()

    assert controller.privileged_credentials == [None, ("postgres", "supersecret")]
    assert controller.set_privileged_credentials_calls[-1] == ("postgres", "supersecret")
    assert controller.applied_database_states[-1].password == ""
    assert controller.state.database.password == ""
    assert dsn.endswith("@localhost:5432/atlas")


def test_configure_database_reuses_staged_privileged_credentials():
    controller = DummyController()
    controller.set_privileged_credentials(("postgres", "supersecret"))
    controller.bootstrap_error_sequence = [BootstrapError('role "atlas" does not exist')]

    responses = iter(["", "", "", "", "", "n"])
    passwords = iter(["password"])

    utility = SetupUtility(
        controller=controller,
        input_func=lambda prompt: next(responses),
        getpass_func=lambda prompt: next(passwords),
        print_func=lambda message: None,
    )

    dsn = utility.configure_database()

    assert controller.privileged_credentials == [
        ("postgres", "supersecret"),
        ("postgres", "supersecret"),
    ]
    assert controller.set_privileged_credentials_calls[-1] == ("postgres", "supersecret")
    assert dsn.endswith("@localhost:5432/atlas")


def test_configure_vector_store_updates_adapter():
    controller = DummyController()
    controller._vector_settings["adapters"]["pinecone"] = {}
    responses = iter(["pinecone"])

    utility = SetupUtility(
        controller=controller,
        input_func=lambda prompt: next(responses),
        getpass_func=lambda prompt: "",
        print_func=lambda message: None,
    )

    settings = utility.configure_vector_store()

    assert controller.state.vector_store.adapter == "pinecone"
    assert settings["default_adapter"] == "pinecone"


def test_configure_user_stages_profile_and_normalizes_date():
    controller = DummyController()
    controller.set_privileged_credentials(("dbadmin", "dbpass"))
    responses_iter = iter(
        [
            "Ada Lovelace",  # full name
            "ada",  # username
            "ada@example.com",  # email
            "example.com",  # domain
            "12/10/1815",  # dob
            "Administrator",  # display name
            "root",  # sudo username
        ]
    )
    passwords_iter = iter([
        "P@ssw0rd!",  # account password
        "P@ssw0rd!",  # account confirm
        "S3cret!",  # sudo password
        "S3cret!",  # sudo confirm
    ])
    input_prompts: list[str] = []
    password_prompts: list[str] = []

    def record_input(prompt: str) -> str:
        input_prompts.append(prompt)
        return next(responses_iter)

    def record_getpass(prompt: str) -> str:
        password_prompts.append(prompt)
        return next(passwords_iter)

    utility = SetupUtility(
        controller=controller,
        input_func=record_input,
        getpass_func=record_getpass,
        print_func=lambda message: None,
    )

    result = utility.configure_user()

    assert input_prompts == [
        "Administrator full name: ",
        "Administrator username: ",
        "Administrator email: ",
        "Administrator domain: ",
        "Administrator date of birth (YYYY-MM-DD): ",
        "Display name: ",
        "Privileged sudo username: ",
    ]
    assert password_prompts == [
        "Administrator password: ",
        "Confirm password: ",
        "Privileged sudo password: ",
        "Confirm privileged sudo password: ",
    ]
    assert not controller.registered_users
    staged_profile = controller.staged_profiles[-1]
    assert staged_profile.full_name == "Ada Lovelace"
    assert staged_profile.date_of_birth == "1815-12-10"
    assert staged_profile.domain == "example.com"
    assert staged_profile.sudo_username == "root"
    assert staged_profile.sudo_password == "S3cret!"
    assert staged_profile.privileged_db_username == "dbadmin"
    assert staged_profile.privileged_db_password == "dbpass"
    assert controller.set_privileged_credentials_calls == [
        ("dbadmin", "dbpass"),
        ("dbadmin", "dbpass"),
    ]
    assert result["date_of_birth"] == "1815-12-10"
    assert result["domain"] == "example.com"
    assert result["privileged_credentials"]["sudo_username"] == "root"
    assert result["privileged_credentials"]["sudo_password"] == "S3cret!"


def test_install_postgresql_reuses_staged_sudo_password(monkeypatch):
    controller = DummyController()
    controller.set_user_profile(AdminProfile(sudo_password="secret"))

    responses = iter(["y"])
    captured_runs: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        captured_runs.append((command, kwargs))
        return types.SimpleNamespace(returncode=0)

    def fail_getpass(prompt):
        raise AssertionError("sudo password prompt should be skipped")

    utility = SetupUtility(
        controller=controller,
        input_func=lambda prompt: next(responses),
        getpass_func=fail_getpass,
        print_func=lambda message: None,
        run=fake_run,
        platform_system=lambda: "Linux",
    )

    monkeypatch.setattr(utility, "_postgres_commands", lambda system: [["sudo", "true"]])

    utility.install_postgresql()

    assert captured_runs
    command, kwargs = captured_runs[0]
    assert command == ["sudo", "true"]
    assert kwargs["input"] == "secret\n"
    assert kwargs["text"] is True


def test_finalize_writes_marker(monkeypatch, tmp_path: Path):
    controller = DummyController()
    utility = SetupUtility(
        controller=controller,
        input_func=lambda prompt: "",
        getpass_func=lambda prompt: "",
        print_func=lambda message: None,
    )

    marker_path = tmp_path / "marker.json"
    captured = {}

    def fake_write(payload):
        captured.update(payload)
        return marker_path

    monkeypatch.setattr("ATLAS.setup.cli.write_setup_marker", fake_write)

    summary = {"database": {"dsn": "postgresql://example"}}
    result = utility.finalize(summary)

    assert captured["setup_complete"] is True
    assert captured["summary"] == summary
    assert result == marker_path


def test_run_skips_virtualenv_and_focuses_on_configuration(monkeypatch):
    controller = DummyController()
    utility = SetupUtility(
        controller=controller,
        input_func=lambda prompt: "",
        getpass_func=lambda prompt: "",
        print_func=lambda message: None,
    )

    order: list[str] = []

    def recorder(name: str, return_value):
        def _(*args, **kwargs):
            order.append(name)
            return return_value

        return _

    monkeypatch.setattr(utility, "ensure_virtualenv", Mock(side_effect=AssertionError("should not be called")))
    monkeypatch.setattr(
        utility,
        "choose_setup_type",
        recorder("choose_setup_type", SetupTypeState(mode="personal", applied=True)),
    )
    monkeypatch.setattr(utility, "configure_user", recorder("configure_user", {}))
    monkeypatch.setattr(utility, "install_postgresql", recorder("install_postgresql", None))
    monkeypatch.setattr(utility, "configure_database", recorder("configure_database", "dsn"))
    monkeypatch.setattr(utility, "configure_kv_store", recorder("configure_kv_store", {}))
    monkeypatch.setattr(utility, "configure_job_scheduling", recorder("configure_job_scheduling", {}))
    monkeypatch.setattr(utility, "configure_message_bus", recorder("configure_message_bus", {}))
    monkeypatch.setattr(utility, "configure_providers", recorder("configure_providers", {}))
    monkeypatch.setattr(utility, "configure_speech", recorder("configure_speech", {}))
    monkeypatch.setattr(utility, "configure_optional_settings", recorder("configure_optional_settings", {}))
    monkeypatch.setattr(controller, "register_user", recorder("register_user", {}))

    sentinel = Path("/tmp/setup.json")
    monkeypatch.setattr(utility, "finalize", recorder("finalize", sentinel))

    result = utility.run()

    assert result == sentinel
    assert order == [
        "choose_setup_type",
        "configure_user",
        "install_postgresql",
        "configure_database",
        "configure_kv_store",
        "configure_job_scheduling",
        "configure_message_bus",
        "configure_providers",
        "configure_speech",
        "configure_optional_settings",
        "register_user",
        "finalize",
    ]


def test_run_non_interactive_consumes_environment(monkeypatch, tmp_path):
    controller = DummyController()
    env = {
        "ATLAS_SETUP_TYPE": "enterprise",
        "ATLAS_ADMIN_USERNAME": "admin",
        "ATLAS_ADMIN_EMAIL": "admin@example.com",
        "ATLAS_ADMIN_PASSWORD": "secret",
        "ATLAS_ADMIN_FULL_NAME": "Admin User",
        "ATLAS_ADMIN_DISPLAY_NAME": "Administrator",
        "ATLAS_ADMIN_DOMAIN": "example.com",
        "ATLAS_ADMIN_DOB": "2000-01-02",
        "ATLAS_SUDO_USERNAME": "root",
        "ATLAS_SUDO_PASSWORD": "sudo-secret",
        "ATLAS_DATABASE_HOST": "db.example.com",
        "ATLAS_DATABASE_PORT": "5433",
        "ATLAS_DATABASE_NAME": "atlas_prod",
        "ATLAS_DATABASE_USER": "atlas_user",
        "ATLAS_DATABASE_PASSWORD": "db-pass",
        "ATLAS_DATABASE_PRIVILEGED_USER": "postgres",
        "ATLAS_DATABASE_PRIVILEGED_PASSWORD": "pg-pass",
        "ATLAS_OPTIONAL_TENANT_ID": "tenant-123",
        "ATLAS_OPTIONAL_HTTP_AUTO_START": "true",
        "ATLAS_PROVIDER_DEFAULT": "openai",
        "ATLAS_PROVIDER_KEY_OPENAI": "api-key",
    }

    sentinel = tmp_path / "sentinel.json"

    utility = SetupUtility(
        controller=controller,
        print_func=lambda message: None,
        env=env,
    )

    def register_user_stub(self, state=None):
        self.registered_users.append(dataclasses.replace(self.state.user))
        return {"username": self.state.user.username}

    controller.register_user = types.MethodType(register_user_stub, controller)
    monkeypatch.setattr(utility, "finalize", lambda summary: sentinel)

    result = utility.run(non_interactive=True)

    assert result == sentinel
    assert controller.applied_setup_modes == ["enterprise"]
    assert controller.registered_users and controller.registered_users[0].username == "admin"

    db_state = controller.applied_database_states[-1]
    assert db_state.host == "db.example.com"
    assert db_state.port == 5433
    assert db_state.database == "atlas_prod"
    assert db_state.user == "atlas_user"
    assert db_state.password == "db-pass"
    assert controller.privileged_credentials[-1] == ("postgres", "pg-pass")

    provider_state = controller.state.providers
    assert provider_state.default_provider == "openai"
    assert provider_state.api_keys["openai"] == "api-key"

    optional_state = controller.state.optional
    assert optional_state.tenant_id == "tenant-123"
    assert optional_state.http_auto_start is True


def test_run_non_interactive_missing_required_fields():
    controller = DummyController()
    env = {
        "ATLAS_ADMIN_USERNAME": "admin",
        "ATLAS_ADMIN_EMAIL": "admin@example.com",
        # password intentionally omitted
    }
    utility = SetupUtility(
        controller=controller,
        print_func=lambda message: None,
        env=env,
    )

    with pytest.raises(RuntimeError, match="ATLAS_ADMIN_PASSWORD"):
        utility.run(non_interactive=True)


def test_setup_script_non_interactive_invokes_utility(monkeypatch):
    from scripts import setup_atlas

    calls: dict[str, object] = {}

    class StubUtility:
        def __init__(self):
            calls["instantiated"] = True

        def run(self, *, non_interactive: bool = False):
            calls["non_interactive"] = non_interactive
            return Path("/tmp/sentinel.json")

    monkeypatch.setattr(setup_atlas, "SetupUtility", StubUtility)

    exit_code = setup_atlas.main(["--non-interactive"])

    assert exit_code == 0
    assert calls == {"instantiated": True, "non_interactive": True}
