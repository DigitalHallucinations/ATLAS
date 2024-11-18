# main.py

import gi
import asyncio
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

from ATLAS.ATLAS import ATLAS
from UI.sidebar import Sidebar

async def initialize_atlas():
    atlas = ATLAS()
    await atlas.initialize()
    return atlas

def main():
    loop = asyncio.get_event_loop()
    atlas = loop.run_until_complete(initialize_atlas())
    
    sidebar = Sidebar(atlas)
    sidebar.show_all()
    Gtk.main()  # Start the GTK main loop

if __name__ == "__main__":
    main()
