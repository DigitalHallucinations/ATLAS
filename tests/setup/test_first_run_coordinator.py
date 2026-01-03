import asyncio


import sys
import types

if 'gi' not in sys.modules:
    gi = types.ModuleType('gi')

    def require_version(_namespace: str, _version: str) -> None:
        return None

    gi.require_version = require_version

    repository = types.ModuleType('gi.repository')
    Gtk = types.ModuleType('Gtk')

    class _BaseWidget:
        def __init__(self, *args, **kwargs):
            pass

        def set_application(self, *args, **kwargs):
            pass

        def set_default_size(self, *args, **kwargs):
            pass

        def set_child(self, *args, **kwargs):
            pass

        def set_margin_top(self, *args, **kwargs):
            pass

        def set_margin_bottom(self, *args, **kwargs):
            pass

        def set_margin_start(self, *args, **kwargs):
            pass

        def set_margin_end(self, *args, **kwargs):
            pass

        def set_wrap(self, *args, **kwargs):
            pass

        def set_justify(self, *args, **kwargs):
            pass

        def set_text(self, *args, **kwargs):
            pass

        def set_visible(self, *args, **kwargs):
            pass

        def set_css_classes(self, *args, **kwargs):
            pass

        def append(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

    class Application:
        pass

    class Window(_BaseWidget):
        pass

    class Box(_BaseWidget):
        pass

    class Label(_BaseWidget):
        pass

    class Button(_BaseWidget):
        pass

    class Orientation:
        VERTICAL = 'vertical'
        HORIZONTAL = 'horizontal'

    class Justification:
        FILL = 'fill'

    Gtk.Application = Application
    Gtk.Window = Window
    Gtk.Box = Box
    Gtk.Label = Label
    Gtk.Button = Button
    Gtk.Orientation = Orientation
    Gtk.Justification = Justification

    repository.Gtk = Gtk
    gi.repository = repository
    sys.modules['gi'] = gi
    sys.modules['gi.repository'] = repository
    sys.modules['gi.repository.Gtk'] = Gtk

from GTKUI.Setup.first_run import FirstRunCoordinator


class DummyAtlas:
    def __init__(self):
        self.initialize_calls = 0

    async def initialize(self):
        self.initialize_calls += 1


def run_coroutine(coro):
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro)
    finally:
        loop.close()


class RecordingMainWindow:
    instances = []

    def __init__(self, atlas):
        self.atlas = atlas
        self.present_called = False
        self.application = None
        RecordingMainWindow.instances.append(self)

    def set_application(self, application):
        self.application = application

    def present(self):
        self.present_called = True


class RecordingSetupWindow:
    def __init__(
        self,
        *,
        application,
        atlas,
        on_success,
        on_error,
        error=None,
    ):
        self.application = application
        self.atlas = atlas
        self.on_success = on_success
        self.on_error = on_error
        self.error_messages = []
        self.present_count = 0
        self.closed = False
        if error is not None:
            self.display_error(error)

    def set_application(self, application):
        self.application = application

    def present(self):
        self.present_count += 1

    def close(self):
        self.closed = True

    def display_error(self, error):
        self.error_messages.append(error)

    def trigger_success(self):
        self.on_success()

    def trigger_error(self, error):
        self.on_error(error)


class NoopSetupWindow:
    created = False

    def __init__(self, **_kwargs):
        NoopSetupWindow.created = True


class FactoryAtlasProvider:
    def __init__(self, factory):
        self.factory = factory
        self.calls = 0

    def get_atlas(self):
        self.calls += 1
        return self.factory()


def test_coordinator_launches_main_window_on_success():
    RecordingMainWindow.instances.clear()
    NoopSetupWindow.created = False
    provider = FactoryAtlasProvider(DummyAtlas)
    coordinator = FirstRunCoordinator(
        application=object(),
        atlas_provider=provider,
        main_window_cls=RecordingMainWindow,
        setup_window_cls=NoopSetupWindow,
        loop_runner=run_coroutine,
    )

    coordinator.activate()

    assert coordinator.setup_window is None
    assert RecordingMainWindow.instances
    assert RecordingMainWindow.instances[0].present_called
    assert NoopSetupWindow.created is False


def test_coordinator_shows_setup_when_initialize_fails():
    atlas = DummyAtlas()
    provider = FactoryAtlasProvider(lambda: atlas)

    class FailingRunner:
        def __init__(self):
            self.calls = 0

        def __call__(self, coro):
            self.calls += 1
            raise RuntimeError("bootstrap failed")

    runner = FailingRunner()
    coordinator = FirstRunCoordinator(
        application=object(),
        atlas_provider=provider,
        main_window_cls=RecordingMainWindow,
        setup_window_cls=RecordingSetupWindow,
        loop_runner=runner,
    )

    coordinator.activate()

    assert coordinator.main_window is None
    assert isinstance(coordinator.setup_window, RecordingSetupWindow)
    assert coordinator.setup_window.present_count == 1
    assert runner.calls == 1
    assert coordinator.setup_window.error_messages
    assert isinstance(coordinator.setup_window.error_messages[0], RuntimeError)


def test_setup_retry_closes_wizard_after_success():
    atlas = DummyAtlas()
    provider = FactoryAtlasProvider(lambda: atlas)

    class ToggleRunner:
        def __init__(self):
            self.calls = 0

        def __call__(self, coro):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("still failing")
            run_coroutine(coro)

    runner = ToggleRunner()
    coordinator = FirstRunCoordinator(
        application=object(),
        atlas_provider=provider,
        main_window_cls=RecordingMainWindow,
        setup_window_cls=RecordingSetupWindow,
        loop_runner=runner,
    )

    coordinator.activate()
    assert isinstance(coordinator.setup_window, RecordingSetupWindow)
    assert runner.calls == 1

    coordinator.setup_window.trigger_success()

    assert runner.calls == 2
    assert coordinator.setup_window is None
    assert isinstance(coordinator.main_window, RecordingMainWindow)
    assert coordinator.main_window.present_called


def test_setup_retry_recreates_atlas_on_factory_failure():
    created = []

    class Factory:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("factory broke")
            atlas = DummyAtlas()
            created.append(atlas)
            return atlas

    factory = Factory()
    provider = FactoryAtlasProvider(factory)
    coordinator = FirstRunCoordinator(
        application=object(),
        atlas_provider=provider,
        main_window_cls=RecordingMainWindow,
        setup_window_cls=RecordingSetupWindow,
        loop_runner=run_coroutine,
    )

    coordinator.activate()
    assert isinstance(coordinator.setup_window, RecordingSetupWindow)
    assert factory.calls == 1

    coordinator.setup_window.trigger_success()

    assert factory.calls == 2
    assert created
    assert isinstance(coordinator.main_window, RecordingMainWindow)
    assert coordinator.main_window.present_called
