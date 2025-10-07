# main.py



import gi
import asyncio
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from ATLAS.ATLAS import ATLAS
from GTKUI.sidebar import MainWindow

async def initialize_atlas():
    atlas = ATLAS()
    await atlas.initialize()
    return atlas

def main():
    atlas = asyncio.run(initialize_atlas())

    app = Gtk.Application(application_id='com.example.atlas')

    def on_activate(app):
        window = MainWindow(atlas)
        window.set_application(app)
        window.present()

    app.connect('activate', on_activate)
    app.run()

if __name__ == "__main__":
    main()

