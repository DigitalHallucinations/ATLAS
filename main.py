# main.py

import asyncio
import logging
import sys

import gi

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from ATLAS.ATLAS import ATLAS
from ATLAS.setup_marker import is_setup_complete
from GTKUI.sidebar import MainWindow

logger = logging.getLogger(__name__)


def main() -> None:
    if not is_setup_complete():
        logger.error(
            "ATLAS setup is incomplete. Run the standalone setup utility before launching the UI."
        )
        sys.exit(1)

    app = Gtk.Application(application_id='com.example.atlas')

    def on_activate(app: Gtk.Application) -> None:
        atlas = ATLAS()
        asyncio.run(atlas.initialize())

        window = MainWindow(atlas)
        if hasattr(window, "set_application"):
            window.set_application(app)
        if hasattr(window, "present"):
            window.present()

        # Retain references while the GTK application is running.
        app._atlas_instance = atlas  # type: ignore[attr-defined]
        app._main_window = window  # type: ignore[attr-defined]

    app.connect('activate', on_activate)
    app.run()


if __name__ == "__main__":
    main()
