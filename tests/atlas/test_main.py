import importlib
import sys
import types


if 'gi' not in sys.modules:
    gi = types.ModuleType('gi')

    def require_version(_namespace: str, _version: str) -> None:
        return None

    gi.require_version = require_version

    repository = types.ModuleType('gi.repository')
    Gtk = types.ModuleType('Gtk')

    class Application:
        last_instance = None

        def __init__(self, *args, **kwargs):
            self._signals = {}
            Application.last_instance = self

        def connect(self, signal, handler):
            self._signals[signal] = handler

        def run(self, *args, **kwargs):
            handler = self._signals.get('activate')
            if handler is None:
                raise AssertionError('activate handler not connected')
            handler(self)

    Gtk.Application = Application

    repository.Gtk = Gtk
    gi.repository = repository

    sys.modules['gi'] = gi
    sys.modules['gi.repository'] = repository
    sys.modules['gi.repository.Gtk'] = Gtk


def test_main_launches_setup_when_incomplete(monkeypatch):
    stub_sidebar = types.ModuleType('GTKUI.sidebar')

    class DummyMainWindow:
        def __init__(self, atlas):
            self.atlas = atlas

    stub_sidebar.MainWindow = DummyMainWindow
    monkeypatch.setitem(sys.modules, 'GTKUI.sidebar', stub_sidebar)

    stub_atlas_module = types.ModuleType('ATLAS.ATLAS')

    class DummyAtlas:
        async def initialize(self):
            return None

    stub_atlas_module.ATLAS = DummyAtlas
    monkeypatch.setitem(sys.modules, 'ATLAS.ATLAS', stub_atlas_module)

    sys.modules.pop('main', None)
    module = importlib.import_module('main')

    class DummyCoordinator:
        instance = None

        def __init__(self, *, application, atlas_provider, main_window_cls, **kwargs):
            DummyCoordinator.instance = self
            self.application = application
            self.atlas_provider = atlas_provider
            self.main_window_cls = main_window_cls
            self.kwargs = kwargs
            self.atlas = None
            self.main_window = None
            self.setup_window = None

        def activate(self):
            try:
                self.atlas = self.atlas_provider.get_atlas()
            except RuntimeError:
                self.setup_window = object()
            else:
                self.main_window = self.main_window_cls(self.atlas)

    monkeypatch.setattr(module, 'FirstRunCoordinator', DummyCoordinator)

    setup_checks = []

    class DummyAtlasProvider:
        instance = None

        def __init__(self, *, atlas_cls, setup_check):
            self.atlas_cls = atlas_cls
            self.setup_check = setup_check
            self.atlas = None
            DummyAtlasProvider.instance = self

        def get_atlas(self):
            if not self.setup_check():
                raise RuntimeError("ATLAS setup is incomplete.")
            if self.atlas is None:
                self.atlas = self.atlas_cls()
            return self.atlas

    def fake_is_setup_complete():
        setup_checks.append(True)
        return False

    monkeypatch.setattr(module, 'is_setup_complete', fake_is_setup_complete)
    monkeypatch.setattr(module, 'AtlasProvider', DummyAtlasProvider)

    module.Gtk.Application.last_instance = None

    module.main()

    app = module.Gtk.Application.last_instance
    assert app is not None

    coordinator = getattr(app, '_first_run_coordinator', None)
    assert coordinator is DummyCoordinator.instance
    assert coordinator.application is app
    assert coordinator.main_window is None
    assert coordinator.setup_window is not None

    assert getattr(app, '_setup_window', None) is coordinator.setup_window
    assert getattr(app, '_atlas_instance', None) is None
    assert getattr(app, '_main_window', None) is None
    assert getattr(app, '_atlas_provider', None) is DummyAtlasProvider.instance
    assert setup_checks

    try:
        DummyAtlasProvider.instance.get_atlas()
    except RuntimeError:
        pass
    else:
        raise AssertionError('atlas_provider should raise when setup is incomplete')
