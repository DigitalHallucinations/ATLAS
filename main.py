# main.py

import gi
import asyncio
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from ATLAS.ATLAS import ATLAS
from UI.sidebar import Sidebar

async def initialize_atlas():
    atlas = ATLAS()
    await atlas.initialize()
    return atlas

def main():
    atlas = asyncio.run(initialize_atlas())

    app = Gtk.Application(application_id='com.example.sidebar')

    def on_activate(app):
        sidebar = Sidebar(atlas)
        sidebar.set_application(app)
        sidebar.present()

    app.connect('activate', on_activate)
    app.run()

if __name__ == "__main__":
    main()

