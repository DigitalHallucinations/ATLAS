# main.py

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

from ATLAS.ATLAS import ATLAS
from UI.sidebar import Sidebar

def main():
    atlas = ATLAS()  # Initialize ATLAS synchronously
    sidebar = Sidebar(atlas)
    sidebar.show_all()
    Gtk.main()  # Start the GTK main loop

if __name__ == "__main__":
    main()
