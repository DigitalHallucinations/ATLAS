# main.py

import asyncio
import threading
from ATLAS.ATLAS import ATLAS
from UI.sidebar import Sidebar
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

def run_ui(atlas):
    """
    Initialize and run the GTK UI components.

    Args:
        atlas (ATLAS): The initialized ATLAS instance.
    """
    sidebar = Sidebar(atlas)
    sidebar.show_all()
    Gtk.main()

async def initialize_atlas():
    """
    Asynchronously initialize the ATLAS instance.

    Returns:
        ATLAS: The initialized ATLAS instance.
    """
    atlas = await ATLAS.create()
    return atlas

async def main():
    """
    Main asynchronous function to initialize ATLAS and run the UI.
    """
    atlas = await initialize_atlas()
    # Start the GTK main loop in a separate thread to prevent blocking
    ui_thread = threading.Thread(target=run_ui, args=(atlas,), daemon=True)
    ui_thread.start()
    # Keep the main thread alive to allow asynchronous tasks to run
    try:
        while True:
            await asyncio.sleep(3600)  # Sleep for an hour, can be adjusted as needed
    except KeyboardInterrupt:
        print("Shutting down application.")

if __name__ == "__main__":
    asyncio.run(main())