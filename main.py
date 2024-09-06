from UI.sidebar import Sidebar
from gi.repository import Gtk

if __name__ == "__main__":
    sidebar = Sidebar()
    sidebar.connect("destroy", Gtk.main_quit)
    sidebar.show_all()
    Gtk.main()
