# PS6_main.py

import sys
import asyncio
from PySide6.QtWidgets import QApplication

from ATLAS.ATLAS import ATLAS
from PS6UI.sidebar import Sidebar

async def initialize_atlas():
    atlas = ATLAS()
    await atlas.initialize()
    return atlas

def main():
    # Initialize the Qt application
    app = QApplication(sys.argv)

    # Since we are using asyncio, we need to run the async function outside the Qt loop
    loop = asyncio.get_event_loop()
    atlas = loop.run_until_complete(initialize_atlas())

    sidebar = Sidebar(atlas)
    sidebar.show()  # In Qt, use show() to display the window

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
