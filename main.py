# main.py

import gi

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from ATLAS.ATLAS import ATLAS
from GTKUI.Setup.first_run import FirstRunCoordinator
from GTKUI.sidebar import MainWindow


def main():
    app = Gtk.Application(application_id='com.example.atlas')

    def on_activate(app: Gtk.Application) -> None:
        coordinator = FirstRunCoordinator(
            application=app,
            atlas_factory=ATLAS,
            main_window_cls=MainWindow,
        )
        # Retain a reference to avoid premature garbage collection while the
        # GTK application is running.
        app._first_run_coordinator = coordinator  # type: ignore[attr-defined]
        coordinator.activate()

    app.connect('activate', on_activate)
    app.run()


if __name__ == "__main__":
    main()
