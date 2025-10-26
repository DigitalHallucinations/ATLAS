import importlib
import logging
import sys
import types

import pytest


if 'gi' not in sys.modules:
    gi = types.ModuleType('gi')

    def require_version(_namespace: str, _version: str) -> None:
        return None

    gi.require_version = require_version

    repository = types.ModuleType('gi.repository')
    Gtk = types.ModuleType('Gtk')

    class Application:
        def __init__(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            pass

    Gtk.Application = Application

    repository.Gtk = Gtk
    gi.repository = repository

    sys.modules['gi'] = gi
    sys.modules['gi.repository'] = repository
    sys.modules['gi.repository.Gtk'] = Gtk


def test_main_exits_when_setup_incomplete(monkeypatch, caplog):
    stub_sidebar = types.ModuleType('GTKUI.sidebar')

    class DummyMainWindow:
        def __init__(self, atlas):
            self.atlas = atlas

        def set_application(self, application):
            self.application = application

        def present(self):
            self.presented = True

    stub_sidebar.MainWindow = DummyMainWindow
    monkeypatch.setitem(sys.modules, 'GTKUI.sidebar', stub_sidebar)

    stub_atlas = types.ModuleType('ATLAS.ATLAS')

    class DummyAtlas:
        async def initialize(self):
            return None

    stub_atlas.ATLAS = DummyAtlas
    monkeypatch.setitem(sys.modules, 'ATLAS.ATLAS', stub_atlas)

    sys.modules.pop('main', None)
    module = importlib.import_module('main')

    def fail_is_setup_complete():
        return False

    monkeypatch.setattr(module, 'is_setup_complete', fail_is_setup_complete)

    class FailingApplication:
        def __init__(self, *args, **kwargs):
            raise AssertionError('GTK application should not be created when setup is incomplete.')

    monkeypatch.setattr(module.Gtk, 'Application', FailingApplication)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            module.main()

    assert excinfo.value.code == 1
    assert 'setup' in caplog.text.lower()
