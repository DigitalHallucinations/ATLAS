import sys
import types


pytest_plugins = ("pytest_postgresql",)

if "pygame" not in sys.modules:
    music = types.SimpleNamespace(
        load=lambda *_args, **_kwargs: None,
        play=lambda *_args, **_kwargs: None,
        get_busy=lambda: False,
    )
    mixer = types.SimpleNamespace(init=lambda *_args, **_kwargs: None, music=music)

    class _Clock:
        def tick(self, *_args, **_kwargs):
            return None

    time_module = types.SimpleNamespace(Clock=lambda: _Clock())

    pygame_module = types.ModuleType("pygame")
    pygame_module.mixer = mixer
    pygame_module.time = time_module

    sys.modules["pygame"] = pygame_module


try:
    import requests  # type: ignore  # noqa: F401
except Exception:
    requests = None  # pragma: no cover - ensures stub path executes

if "requests" not in sys.modules and requests is None:
    class _DummyResponse:
        ok = True
        text = ""

        def json(self):
            return {}

        def iter_content(self, chunk_size=1024):
            return iter(())

    def _dummy_request(*_args, **_kwargs):
        return _DummyResponse()

    requests_module = types.ModuleType("requests")
    requests_module.get = _dummy_request
    requests_module.post = _dummy_request
    requests_module.__path__ = []

    class _Session:
        def request(self, *args, **kwargs):  # pragma: no cover - stubbed session
            return _dummy_request(*args, **kwargs)

    requests_module.Session = _Session

    exceptions_module = types.ModuleType("requests.exceptions")

    class _RequestException(Exception):
        """Stub base class mirroring requests.exceptions.RequestException."""

    class _HTTPError(_RequestException):
        """Stub HTTP error matching requests.exceptions.HTTPError."""

    exceptions_module.RequestException = _RequestException
    exceptions_module.HTTPError = _HTTPError

    requests_module.exceptions = exceptions_module
    class _DummyResponse:
        status_code = 200
        text = ""
        content = b""

        def json(self):
            return {}

    requests_module.HTTPError = exceptions_module.HTTPError
    requests_module.RequestException = exceptions_module.RequestException
    requests_module.Response = _DummyResponse

    adapters_module = types.ModuleType("requests.adapters")

    class _HTTPAdapter:
        """Stub adapter mirroring requests.adapters.HTTPAdapter."""

        def __init__(self, *args, **kwargs):
            pass

        def send(self, request, **kwargs):  # pragma: no cover - stub helper
            return request

    adapters_module.HTTPAdapter = _HTTPAdapter


try:
    import jsonschema  # type: ignore  # noqa: F401
except Exception:
    jsonschema = None  # pragma: no cover - ensures stub path executes

if "jsonschema" not in sys.modules and jsonschema is None:
    jsonschema_stub = types.ModuleType("jsonschema")
    jsonschema_stub.__file__ = __file__
    
    class _Draft202012Validator:
        """Lightweight stub matching jsonschema.Draft202012Validator."""

        def __init__(self, *args, **kwargs):  # pragma: no cover - stub initialiser
            self.schema = args[0] if args else None

        def iter_errors(self, *_args, **_kwargs):  # pragma: no cover - stub helper
            return iter(())

    jsonschema_stub.Draft202012Validator = _Draft202012Validator
    sys.modules["jsonschema"] = jsonschema_stub


try:
    import sqlalchemy  # type: ignore  # noqa: F401
except Exception:
    sqlalchemy = None  # pragma: no cover - ensures stub path executes

if "sqlalchemy" not in sys.modules and sqlalchemy is None:
    sqlalchemy_stub = types.ModuleType("sqlalchemy")
    sqlalchemy_stub.__file__ = __file__
    sqlalchemy_stub.__path__ = []

    def _marker(name):
        class _Type:
            def __init__(self, *args, **kwargs):  # pragma: no cover - stub initialiser
                self.args = args
                self.kwargs = kwargs

        _Type.__name__ = name
        return _Type

    def _noop(*_args, **_kwargs):  # pragma: no cover - lightweight stub
        return None

    for attr in [
        "Boolean",
        "Enum",
        "DateTime",
        "Float",
        "ForeignKey",
        "Index",
        "Integer",
        "String",
        "Text",
        "UniqueConstraint",
    ]:
        setattr(sqlalchemy_stub, attr, _marker(attr))

    def _column(*args, **kwargs):
        name = args[0] if args else kwargs.get("name")
        return types.SimpleNamespace(args=args, kwargs=kwargs, name=name, key=name)

    def _table(name, metadata, *columns, **kwargs):
        column_map = {}
        for column in columns:
            key = getattr(column, "key", None)
            if isinstance(key, str):
                column_map[key] = column
        return types.SimpleNamespace(
            name=name,
            metadata=metadata,
            columns=columns,
            kwargs=kwargs,
            c=types.SimpleNamespace(**column_map),
        )

    sqlalchemy_stub.Column = _column
    sqlalchemy_stub.Table = _table
    sqlalchemy_stub.MetaData = lambda *args, **kwargs: types.SimpleNamespace(create_all=_noop)
    sqlalchemy_stub.inspect = _noop
    sqlalchemy_stub.create_engine = _noop
    sqlalchemy_stub.delete = lambda *args, **kwargs: ("delete", args, kwargs)

    class _FuncProxy:
        def __getattr__(self, name):  # pragma: no cover - stub helper
            return lambda *args, **kwargs: (name, args, kwargs)

    sqlalchemy_stub.func = _FuncProxy()
    sqlalchemy_stub.select = lambda *args, **kwargs: ("select", args, kwargs)

    event_module = types.ModuleType("sqlalchemy.event")

    def _listen_decorator(*_args, **_kwargs):  # pragma: no cover - stub decorator
        def _wrapper(func):
            return func

        return _wrapper

    event_module.listens_for = _listen_decorator
    sqlalchemy_stub.event = event_module

    dialects_module = types.ModuleType("sqlalchemy.dialects")
    postgres_module = types.ModuleType("sqlalchemy.dialects.postgresql")
    for attr in ["ARRAY", "JSONB", "UUID", "TSVECTOR"]:
        setattr(postgres_module, attr, _marker(attr))
    postgres_module.insert = lambda *args, **kwargs: ("insert", args, kwargs)

    orm_module = types.ModuleType("sqlalchemy.orm")
    exc_module = types.ModuleType("sqlalchemy.exc")

    class _IntegrityError(Exception):
        pass

    exc_module.IntegrityError = _IntegrityError

    def _declarative_base():  # pragma: no cover - lightweight declarative base stub
        metadata = types.SimpleNamespace(create_all=_noop)
        return type("Base", (), {"metadata": metadata})

    orm_module.declarative_base = _declarative_base
    orm_module.relationship = _noop
    orm_module.sessionmaker = lambda *args, **kwargs: _noop

    engine_module = types.ModuleType("sqlalchemy.engine")
    engine_module.Engine = _marker("Engine")

    sqlalchemy_stub.Engine = engine_module.Engine

    sys.modules["sqlalchemy"] = sqlalchemy_stub
    sys.modules["sqlalchemy.dialects"] = dialects_module
    sys.modules["sqlalchemy.dialects.postgresql"] = postgres_module
    sys.modules["sqlalchemy.orm"] = orm_module
    sys.modules["sqlalchemy.exc"] = exc_module
    sys.modules["sqlalchemy.engine"] = engine_module
    sys.modules["sqlalchemy.event"] = event_module
    types_module = types.ModuleType("sqlalchemy.types")
    types_module.JSON = _marker("JSON")
    types_module.TypeDecorator = _marker("TypeDecorator")
    sys.modules["sqlalchemy.types"] = types_module
    sql_module = types.ModuleType("sqlalchemy.sql")
    sql_module.Select = _marker("Select")
    sys.modules["sqlalchemy.sql"] = sql_module
    engine_url_module = types.ModuleType("sqlalchemy.engine.url")

    class _StubURL:
        def __init__(self, url: str):
            self._url = url
            self.drivername = url.split(":", 1)[0]

        def set(self, *, drivername: str):
            self.drivername = drivername
            return self

        def __str__(self) -> str:  # pragma: no cover - deterministic repr
            return self._url

    def _stub_make_url(url: str):  # pragma: no cover - lightweight helper
        return _StubURL(url)

    engine_url_module.URL = _StubURL
    engine_url_module.make_url = _stub_make_url
    sys.modules["sqlalchemy.engine.url"] = engine_url_module
    requests_module.adapters = adapters_module

    sys.modules["requests"] = requests_module
    sys.modules["requests.exceptions"] = exceptions_module
    sys.modules["requests.adapters"] = adapters_module

if "jsonschema" not in sys.modules:
    jsonschema_module = types.ModuleType("jsonschema")

    class _DummyValidator:
        def __init__(self, *args, **kwargs):
            pass

        def validate(self, *args, **kwargs):
            return None

        def iter_errors(self, *args, **kwargs):  # pragma: no cover - stub
            return []

    class _DummyValidationError(Exception):
        def __init__(self, message: str = "", path=None):
            super().__init__(message)
            self.message = message
            self.absolute_path = list(path or [])

    jsonschema_module.Draft7Validator = _DummyValidator
    jsonschema_module.Draft202012Validator = _DummyValidator
    jsonschema_module.ValidationError = _DummyValidationError
    jsonschema_module.exceptions = types.SimpleNamespace(ValidationError=_DummyValidationError)
    sys.modules["jsonschema"] = jsonschema_module

if "yaml" not in sys.modules:
    yaml_module = types.ModuleType("yaml")

    def _safe_load_fallback(stream=None, *_args, **_kwargs):
        """Lightweight JSON-based ``yaml.safe_load`` drop-in."""

        import json as _json

        if hasattr(stream, "read"):
            content = stream.read()
        else:
            content = stream or ""

        content = (content or "").strip()
        if not content:
            return {}

        try:
            return _json.loads(content)
        except _json.JSONDecodeError:
            return {}

    def _safe_dump_fallback(data, stream=None, *_args, **_kwargs):
        """Mirror ``yaml.safe_dump`` using JSON for tests."""

        import json as _json

        text = _json.dumps(data or {})
        if stream is None:
            return text

        stream.write(text)
        return text

    yaml_module.safe_load = _safe_load_fallback
    yaml_module.safe_dump = _safe_dump_fallback
    yaml_module.dump = _safe_dump_fallback
    sys.modules["yaml"] = yaml_module

if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")

    def _noop(*_args, **_kwargs):  # pragma: no cover - dotenv stub
        return None

    dotenv_module.load_dotenv = _noop
    dotenv_module.set_key = _noop
    dotenv_module.find_dotenv = lambda *_args, **_kwargs: ""
    sys.modules["dotenv"] = dotenv_module

if "pytz" not in sys.modules:
    sys.modules["pytz"] = types.SimpleNamespace(timezone=lambda *_args, **_kwargs: None, utc=None)

if "aiohttp" not in sys.modules:
    aiohttp_module = types.ModuleType("aiohttp")

    class _DummySession:
        async def __aenter__(self):  # pragma: no cover - stub
            return self

        async def __aexit__(self, *exc_info):  # pragma: no cover - stub
            return False

        async def get(self, *args, **kwargs):  # pragma: no cover - stub
            raise RuntimeError("aiohttp stub is not implemented")

    aiohttp_module.ClientSession = _DummySession
    sys.modules["aiohttp"] = aiohttp_module

if "apscheduler" not in sys.modules:
    apscheduler_module = types.ModuleType("apscheduler")
    events_module = types.ModuleType("apscheduler.events")

    events_module.EVENT_JOB_ERROR = 1
    events_module.EVENT_JOB_EXECUTED = 2
    events_module.EVENT_JOB_MISSED = 3

    class JobEvent:  # pragma: no cover - stub container
        def __init__(self, *args, **kwargs):
            self.job_id = kwargs.get("job_id")

    events_module.JobEvent = JobEvent

    sys.modules["apscheduler"] = apscheduler_module
    sys.modules["apscheduler.events"] = events_module
    executors_module = types.ModuleType("apscheduler.executors")
    pool_module = types.ModuleType("apscheduler.executors.pool")

    class ThreadPoolExecutor:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs):
            pass

    pool_module.ThreadPoolExecutor = ThreadPoolExecutor

    sys.modules["apscheduler.executors"] = executors_module
    sys.modules["apscheduler.executors.pool"] = pool_module
    jobstores_module = types.ModuleType("apscheduler.jobstores.base")

    class _StubJobstoreError(Exception):
        pass

    class ConflictingIdError(_StubJobstoreError):
        pass

    class JobLookupError(_StubJobstoreError):
        pass

    jobstores_module.ConflictingIdError = ConflictingIdError
    jobstores_module.JobLookupError = JobLookupError

    sys.modules["apscheduler.jobstores.base"] = jobstores_module
    sql_module = types.ModuleType("apscheduler.jobstores.sqlalchemy")

    class SQLAlchemyJobStore:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs):
            pass

    sql_module.SQLAlchemyJobStore = SQLAlchemyJobStore
    sys.modules["apscheduler.jobstores.sqlalchemy"] = sql_module
    schedulers_module = types.ModuleType("apscheduler.schedulers")
    background_module = types.ModuleType("apscheduler.schedulers.background")

    class BackgroundScheduler:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs):
            pass

    background_module.BackgroundScheduler = BackgroundScheduler

    sys.modules["apscheduler.schedulers"] = schedulers_module
    sys.modules["apscheduler.schedulers.background"] = background_module
    triggers_module = types.ModuleType("apscheduler.triggers")
    cron_module = types.ModuleType("apscheduler.triggers.cron")
    date_module = types.ModuleType("apscheduler.triggers.date")

    class CronTrigger:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs):
            pass

    class DateTrigger:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs):
            pass

    cron_module.CronTrigger = CronTrigger
    date_module.DateTrigger = DateTrigger

    sys.modules["apscheduler.triggers"] = triggers_module
    sys.modules["apscheduler.triggers.cron"] = cron_module
    sys.modules["apscheduler.triggers.date"] = date_module

if "huggingface_hub" not in sys.modules:
    hf_module = types.ModuleType("huggingface_hub")

    class _StubHfApi:
        """Lightweight stub for Hugging Face client used in tests."""

        def __init__(self, *args, **kwargs):
            pass

        def whoami(self, *args, **kwargs):  # pragma: no cover - stub helper
            return {}

    hf_module.HfApi = _StubHfApi
    hf_module.hf_hub_download = lambda *args, **kwargs: ""
    sys.modules["huggingface_hub"] = hf_module

if "modules.Providers.HuggingFace.HF_gen_response" not in sys.modules:
    hf_provider_module = types.ModuleType("modules.Providers.HuggingFace.HF_gen_response")

    class _StubModelManager:
        def __init__(self):
            self.installed = []

        def remove_installed_model(self, model_name):
            self.installed = [name for name in self.installed if name != model_name]

        def get_installed_models(self):
            return list(self.installed)

    class HuggingFaceGenerator:
        def __init__(self, *_args, **_kwargs):
            self.model_manager = _StubModelManager()

        async def load_model(self, model_name, *_args, **_kwargs):
            self.model_manager.installed.append(model_name)

        async def generate_response(self, messages, model, stream=True):  # pragma: no cover - stub helper
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - stub helper
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

        def unload_model(self):  # pragma: no cover - stub helper
            return None

        def get_installed_models(self):
            return self.model_manager.get_installed_models()

    async def search_models(*_args, **_kwargs):  # pragma: no cover - stub helper
        return []

    async def download_model(generator, model_id, force=False):  # pragma: no cover - stub helper
        generator.model_manager.installed.append(model_id)

    def update_model_settings(*_args, **_kwargs):  # pragma: no cover - stub helper
        return {}

    def clear_cache(*_args, **_kwargs):  # pragma: no cover - stub helper
        return None

    hf_provider_module.HuggingFaceGenerator = HuggingFaceGenerator
    hf_provider_module.search_models = search_models
    hf_provider_module.download_model = download_model
    hf_provider_module.update_model_settings = update_model_settings
    hf_provider_module.clear_cache = clear_cache

    sys.modules["modules.Providers.HuggingFace.HF_gen_response"] = hf_provider_module

if "modules.Providers.Grok.grok_generate_response" not in sys.modules:
    grok_module = types.ModuleType("modules.Providers.Grok.grok_generate_response")

    class GrokGenerator:
        def __init__(self, *_args, **_kwargs):
            pass

        async def generate_response(self, *_args, **_kwargs):  # pragma: no cover - stub helper
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - stub helper
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

        async def unload_model(self):  # pragma: no cover - stub helper
            return None

    grok_module.GrokGenerator = GrokGenerator
    sys.modules["modules.Providers.Grok.grok_generate_response"] = grok_module

if "modules.Providers.OpenAI.OA_gen_response" not in sys.modules:
    openai_module = types.ModuleType("modules.Providers.OpenAI.OA_gen_response")

    from weakref import WeakKeyDictionary

    class _StubModelManager:
        def __init__(self):
            self.current_model = "gpt-3.5-turbo"

        def set_model(self, model, provider):
            self.current_model = model

        def get_current_model(self):
            return self.current_model

    class OpenAIGenerator:
        def __init__(self, config_manager):
            self.config_manager = config_manager
            self.model_manager = _StubModelManager()

        async def generate_response(self, **_kwargs):  # pragma: no cover - stub helper
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - stub helper
            chunks = []
            async for entry in response:
                chunks.append(entry)
            return "".join(chunks)

        async def aclose(self):  # pragma: no cover - stub helper
            return None

        async def close(self):  # pragma: no cover - stub helper
            return None

    _GENERATOR_CACHE = WeakKeyDictionary()

    def get_generator(config_manager, _module=openai_module, _cache=_GENERATOR_CACHE):
        generator = _cache.get(config_manager)
        if generator is None:
            generator_cls = _module.OpenAIGenerator
            generator = generator_cls(config_manager)
            _cache[config_manager] = generator
        return generator

    openai_module.OpenAIGenerator = OpenAIGenerator
    openai_module._GENERATOR_CACHE = _GENERATOR_CACHE
    openai_module.get_generator = get_generator
    sys.modules["modules.Providers.OpenAI.OA_gen_response"] = openai_module

if "modules.Providers.Mistral.Mistral_gen_response" not in sys.modules:
    mistral_module = types.ModuleType("modules.Providers.Mistral.Mistral_gen_response")

    from weakref import WeakKeyDictionary

    class MistralGenerator:
        def __init__(self, config_manager):
            self.config_manager = config_manager

        async def generate_response(self, **_kwargs):  # pragma: no cover - stub helper
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - stub helper
            chunks = []
            async for entry in response:
                chunks.append(entry)
            return "".join(chunks)

    _GENERATOR_CACHE = WeakKeyDictionary()

    def get_generator(config_manager, _module=mistral_module, _cache=_GENERATOR_CACHE):
        generator = _cache.get(config_manager)
        if generator is None:
            generator_cls = _module.MistralGenerator
            generator = generator_cls(config_manager)
            _cache[config_manager] = generator
        return generator

    mistral_module.MistralGenerator = MistralGenerator
    mistral_module._GENERATOR_CACHE = _GENERATOR_CACHE
    mistral_module.get_generator = get_generator
    sys.modules["modules.Providers.Mistral.Mistral_gen_response"] = mistral_module

if "modules.Providers.Google.GG_gen_response" not in sys.modules:
    google_module = types.ModuleType("modules.Providers.Google.GG_gen_response")

    from weakref import WeakKeyDictionary

    class GoogleGeminiGenerator:
        def __init__(self, config_manager):
            self.config_manager = config_manager

        async def generate_response(self, **_kwargs):  # pragma: no cover - stub helper
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - stub helper
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

    _GENERATOR_CACHE = WeakKeyDictionary()

    def get_generator(config_manager, _module=google_module, _cache=_GENERATOR_CACHE):
        generator = _cache.get(config_manager)
        if generator is None:
            generator_cls = _module.GoogleGeminiGenerator
            generator = generator_cls(config_manager)
            _cache[config_manager] = generator
        return generator

    google_module.GoogleGeminiGenerator = GoogleGeminiGenerator
    google_module._GENERATOR_CACHE = _GENERATOR_CACHE
    google_module.get_generator = get_generator
    sys.modules["modules.Providers.Google.GG_gen_response"] = google_module

if "modules.Providers.Anthropic.Anthropic_gen_response" not in sys.modules:
    anthropic_provider_module = types.ModuleType("modules.Providers.Anthropic.Anthropic_gen_response")

    class AnthropicGenerator:
        def __init__(self, config_manager):
            self.config_manager = config_manager

        async def generate_response(self, **_kwargs):  # pragma: no cover - stub helper
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - stub helper
            chunks = []
            async for entry in response:
                chunks.append(entry)
            return "".join(chunks)

    anthropic_provider_module.AnthropicGenerator = AnthropicGenerator
    sys.modules["modules.Providers.Anthropic.Anthropic_gen_response"] = anthropic_provider_module

if "anthropic" not in sys.modules:
    anthropic_module = types.ModuleType("anthropic")

    class _StubAsyncAnthropic:
        def __init__(self, *args, **kwargs):
            pass

        class _Messages:
            async def create(self, *args, **kwargs):  # pragma: no cover - stub helper
                return types.SimpleNamespace(content=[])

            def stream(self, *args, **kwargs):  # pragma: no cover - stub helper
                raise RuntimeError("stream not supported in stub")

        @property
        def messages(self):  # pragma: no cover - stub helper
            return _StubAsyncAnthropic._Messages()

    anthropic_module.AsyncAnthropic = _StubAsyncAnthropic
    anthropic_module.APIError = Exception
    anthropic_module.RateLimitError = Exception
    sys.modules["anthropic"] = anthropic_module


if "google" not in sys.modules:
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    speech_module = types.ModuleType("google.cloud.speech_v1p1beta1")

    class _DummyVoiceSelectionParams:
        def __init__(self, *args, **kwargs):
            pass

    class _DummySpeechClient:
        def synthesize_speech(self, *args, **kwargs):
            return types.SimpleNamespace(audio_content=b"")

        def list_voices(self):
            return types.SimpleNamespace(voices=[])

    class _DummySynthesisInput:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyAudioConfig:
        def __init__(self, *args, **kwargs):
            pass

    class _DummySsmlVoiceGender:
        MALE = 1
        FEMALE = 2
        NEUTRAL = 3

        def __call__(self, *_args, **_kwargs):
            return types.SimpleNamespace(name="NEUTRAL")

    speech_module.VoiceSelectionParams = _DummyVoiceSelectionParams
    speech_module.SpeechClient = _DummySpeechClient
    speech_module.SynthesisInput = _DummySynthesisInput
    speech_module.AudioConfig = _DummyAudioConfig
    speech_module.AudioEncoding = types.SimpleNamespace(MP3=1)
    speech_module.SsmlVoiceGender = _DummySsmlVoiceGender()

    texttospeech_module = types.ModuleType("google.cloud.texttospeech")
    texttospeech_module.VoiceSelectionParams = _DummyVoiceSelectionParams
    texttospeech_module.TextToSpeechClient = _DummySpeechClient
    texttospeech_module.SynthesisInput = _DummySynthesisInput
    texttospeech_module.AudioConfig = _DummyAudioConfig
    texttospeech_module.AudioEncoding = types.SimpleNamespace(MP3=1)
    texttospeech_module.SsmlVoiceGender = _DummySsmlVoiceGender()

    cloud_module.speech_v1p1beta1 = speech_module
    cloud_module.texttospeech = texttospeech_module
    google_module.cloud = cloud_module

    sys.modules["google"] = google_module
    sys.modules["google.cloud"] = cloud_module
    sys.modules["google.cloud.speech_v1p1beta1"] = speech_module
    sys.modules["google.cloud.texttospeech"] = texttospeech_module


if "sounddevice" not in sys.modules:
    class _DummyInputStream:
        def __init__(self, callback=None, *args, **kwargs):
            self.callback = callback

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    sounddevice_module = types.ModuleType("sounddevice")
    sounddevice_module.InputStream = _DummyInputStream

    sys.modules["sounddevice"] = sounddevice_module


if "soundfile" not in sys.modules:
    soundfile_module = types.ModuleType("soundfile")
    soundfile_module.write = lambda *_args, **_kwargs: None

    sys.modules["soundfile"] = soundfile_module


if "numpy" not in sys.modules:
    def _concatenate(items):
        result = []
        for entry in items:
            if hasattr(entry, "tolist"):
                result.extend(entry.tolist())
            elif isinstance(entry, (list, tuple)):
                result.extend(entry)
            else:
                result.append(entry)
        return result

    numpy_module = types.ModuleType("numpy")
    numpy_module.concatenate = _concatenate
    numpy_module.array = lambda value: value
    numpy_module.isscalar = lambda value: isinstance(value, (int, float, complex, bool, str))
    numpy_module.bool_ = bool

    sys.modules["numpy"] = numpy_module
