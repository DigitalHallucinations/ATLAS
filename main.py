# main.py



import gi
import asyncio
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from ATLAS.ATLAS import ATLAS
from GTKUI.sidebar import MainWindow

def main():
    atlas = ATLAS()
    asyncio.run(atlas.initialize())

    app = Gtk.Application(application_id='com.example.atlas')

    def on_activate(app):
        window = MainWindow(atlas)
        window.set_application(app)
        window.present()

    app.connect('activate', on_activate)
    app.run()

if __name__ == "__main__":
    main()

