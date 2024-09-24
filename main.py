# main.py

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import asyncio
import gbulb
gbulb.install()

from ATLAS.ATLAS import ATLAS
from UI.sidebar import Sidebar

async def main():
    atlas = await ATLAS.create()
    sidebar = Sidebar(atlas)
    sidebar.show_all()

if __name__ == "__main__":
    asyncio.ensure_future(main())
    Gtk.main()
