# GTKUI/sidebar.py

"""Main application window with navigation sidebar and central workspace."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from GTKUI.Budget_manager import BudgetManagement
from GTKUI.Calendar_manager import CalendarManagement
from GTKUI.Chat.chat_page import ChatPage
from GTKUI.Chat.conversation_history_page import ConversationHistoryPage
from GTKUI.Docs.docs_page import DocsPage
from GTKUI.Persona_manager.persona_management import PersonaManagement
from GTKUI.Provider_manager.provider_management import ProviderManagement
from GTKUI.Settings.Speech.speech_settings import SpeechSettings
from GTKUI.Settings.backup_settings import BackupSettings
from GTKUI.Tool_manager import ToolManagement
from GTKUI.Skill_manager import SkillManagement
from GTKUI.Job_manager import JobManagement
from GTKUI.Task_manager import TaskManagement
from GTKUI.UserAccounts.account_dialog import AccountDialog
from GTKUI.Utils.styled_window import AtlasWindow
from GTKUI.Utils.utils import apply_css

logger = logging.getLogger(__name__)


class MainWindow(AtlasWindow):
    """Top-level Atlas window with a navigation column and notebook workspace."""

    def __init__(self, atlas) -> None:
        super().__init__(title="ATLAS")
        safe_width, safe_height = self._calculate_safe_size(1200, 800)
        self.set_default_size(safe_width, safe_height)
        self.ATLAS = atlas
        self._pages: Dict[str, Gtk.Widget] = {}
        self._page_controllers: Dict[str, Any] = {}

        apply_css()

        self.persona_management = PersonaManagement(self.ATLAS, self)
        self.provider_management = ProviderManagement(self.ATLAS, self)
        self.tool_management = ToolManagement(self.ATLAS, self)
        self.skill_management = SkillManagement(self.ATLAS, self)
        self.task_management = TaskManagement(self.ATLAS, self)
        self.job_management = JobManagement(self.ATLAS, self)
        self.budget_management = BudgetManagement(self.ATLAS, self)
        self.calendar_management = CalendarManagement(self.ATLAS, self)
        self.tool_management.on_open_in_persona = self._open_tool_in_persona
        self.skill_management.on_open_in_persona = self._open_skill_in_persona

        self._page_factories: Dict[str, Callable[[], Gtk.Widget | Tuple[Gtk.Widget, Any]]] = {}
        self._register_page_factories()

        layout = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        layout.set_hexpand(True)
        layout.set_vexpand(True)
        self.set_child(layout)
        self.connect("close-request", self.close_application)

        self.sidebar = _NavigationSidebar(self)
        layout.append(self.sidebar)

        separator = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        separator.add_css_class("sidebar-divider")
        layout.append(separator)

        self.notebook = Gtk.Notebook()
        self.notebook.set_hexpand(True)
        self.notebook.set_vexpand(True)
        self.notebook.set_scrollable(True)
        layout.append(self.notebook)

    def _register_page_factories(self) -> None:
        self._page_factories = {
            "chat": self._build_chat_page,
            "providers": self.provider_management.get_embeddable_widget,
            "personas": self.persona_management.get_embeddable_widget,
            "tools": self.tool_management.get_embeddable_widget,
            "tasks": self.task_management.get_embeddable_widget,
            "jobs": self.job_management.get_embeddable_widget,
            "budget": self.budget_management.get_embeddable_widget,
            "calendar": self.calendar_management.get_embeddable_widget,
            "skills": self.skill_management.get_embeddable_widget,
            "speech": self._build_speech_settings_page,
            "settings": self._build_backup_settings_page,
            "conversation-history": self._build_conversation_history_page,
            "accounts": self._build_accounts_page,
            "docs": self._build_docs_page,
        }

    def _build_chat_page(self) -> Gtk.Widget:
        return ChatPage(self.ATLAS)

    def _build_conversation_history_page(self) -> Gtk.Widget:
        return ConversationHistoryPage(self.ATLAS)

    def _build_speech_settings_page(self) -> Gtk.Widget:
        return SpeechSettings(self.ATLAS)

    def _build_backup_settings_page(self) -> Gtk.Widget:
        return BackupSettings(self.ATLAS)

    def _build_docs_page(self) -> Gtk.Widget:
        return DocsPage(self.ATLAS)

    def _build_accounts_page(self) -> Tuple[Gtk.Widget, AccountDialog]:
        dialog = AccountDialog(
            self.ATLAS,
            parent=self,
            on_close=lambda: self._close_page("accounts"),
        )
        content = dialog.get_child()
        if content is None:
            fallback = Gtk.Label(label="Account management is unavailable.")
            fallback.set_hexpand(True)
            fallback.set_vexpand(True)
            return fallback, dialog

        dialog.set_child(None)
        content.set_hexpand(True)
        content.set_vexpand(True)

        scroller = Gtk.ScrolledWindow()
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_child(content)

        return scroller, dialog

    # ------------------------------------------------------------------
    # Navigation actions
    # ------------------------------------------------------------------
    def show_chat_page(self) -> None:
        if not self._ensure_initialized():
            return

        page = self._open_or_focus_page("chat", "Chat")
        if page is not None:
            self.sidebar.set_active_item("chat")

    def show_provider_menu(self) -> None:
        if not self._ensure_initialized():
            return

        page = self._open_or_focus_page("providers", "Providers")
        if page is not None:
            self.sidebar.set_active_item("providers")

    def show_persona_menu(self) -> None:
        if not self._ensure_initialized():
            return

        page = self._open_or_focus_page("personas", "Personas")
        if page is not None:
            self.sidebar.set_active_item("personas")

    def show_tools_menu(self, tool_name: str | None = None) -> None:
        if not self._ensure_initialized():
            return

        page = self._open_or_focus_page("tools", "Tools")
        if page is not None:
            self.sidebar.set_active_item("tools")
            self._focus_tool_if_requested(tool_name)

    def show_task_workspace(self, task_id: str | None = None) -> None:
        if not self._ensure_initialized():
            return

        page = self._open_or_focus_page("tasks", "Tasks")
        if page is not None:
            self.sidebar.set_active_item("tasks")
            self._focus_task_if_requested(task_id)

    def show_job_workspace(self, job_id: str | None = None) -> None:
        if not self._ensure_initialized():
            return

        page = self._open_or_focus_page("jobs", "Jobs")
        if page is not None:
            self.sidebar.set_active_item("jobs")
            self._focus_job_if_requested(job_id)

    def show_budget_workspace(self) -> None:
        if not self._ensure_initialized():
            return

        page = self._open_or_focus_page("budget", "Budget")
        if page is not None:
            self.sidebar.set_active_item("budget")

    def show_calendar_workspace(self, selected_date: str | None = None) -> None:
        if not self._ensure_initialized():
            return

        page = self._open_or_focus_page("calendar", "Calendar")
        if page is not None:
            self.sidebar.set_active_item("calendar")
            if selected_date and hasattr(self.calendar_management, "select_date"):
                self.calendar_management.select_date(selected_date)

    def create_new_job(self) -> None:
        if not self._ensure_initialized():
            return
        self.show_job_workspace()
        creator = getattr(self.job_management, "prompt_new_job", None)
        if callable(creator):
            creator()

    def show_skills_menu(self, skill_name: str | None = None) -> None:
        if not self._ensure_initialized():
            return

        page = self._open_or_focus_page("skills", "Skills")
        if page is not None:
            self.sidebar.set_active_item("skills")
            self._focus_skill_if_requested(skill_name)

    def show_speech_settings(self) -> None:
        if not self._ensure_initialized():
            return

        page = self._open_or_focus_page("speech", "Speech")
        if page is not None:
            self.sidebar.set_active_item("speech")

    def show_docs_page(self) -> None:
        if not self._ensure_initialized():
            return

        page = self._open_or_focus_page("docs", "Docs")
        if page is not None:
            self.sidebar.set_active_item("docs")

    def show_settings_page(self) -> None:
        page = self._open_or_focus_page("settings", "Settings")
        if page is not None:
            self.sidebar.set_active_item("settings")

    def show_conversation_history_page(
        self, conversation_id: str | None = None
    ) -> None:
        if not self._ensure_initialized():
            return

        page = self._open_or_focus_page("conversation-history", "History")
        if page is not None:
            self._focus_conversation_if_requested(page, conversation_id)

    def handle_history_button(self) -> None:
        if not self._ensure_initialized():
            return
        try:
            self.ATLAS.log_history()
        except Exception:  # pragma: no cover - defensive logging only
            logger.debug("History logger hook failed", exc_info=True)
        self.show_conversation_history_page()

    def _focus_job_in_manager(self, job_id: str) -> None:
        focus = getattr(self.job_management, "focus_job", None)
        if callable(focus):
            focus(job_id)

    def _transition_job(
        self,
        job_id: str,
        target_status: str,
        *,
        updated_at: str | None = None,
    ) -> Mapping[str, Any]:
        server = getattr(self.ATLAS, "server", None)
        transition = getattr(server, "transition_job", None)
        if not callable(transition):
            raise RuntimeError("Job transitions are unavailable.")

        context = {"tenant_id": getattr(self.ATLAS, "tenant_id", "default")}
        try:
            payload = transition(
                job_id,
                target_status,
                context=context,
                expected_updated_at=updated_at,
            )
        except Exception:
            logger.error("Failed to transition job %s to %s", job_id, target_status, exc_info=True)
            raise

        notifier = getattr(self, "show_success_toast", None)
        if callable(notifier):
            notifier(f"Job moved to {target_status.replace('_', ' ').title()}")
        return dict(payload) if isinstance(payload, Mapping) else {}

    def start_job(
        self,
        job_id: str,
        current_status: str,
        updated_at: str | None,
        *,
        mode: str = "auto",
    ) -> Mapping[str, Any]:
        status = (current_status or "").lower()
        normalized_mode = (mode or "auto").lower()
        server = getattr(self.ATLAS, "server", None)
        if normalized_mode == "run_now":
            scheduled_payload: Mapping[str, Any] | None = None
            if status == "draft":
                scheduled_payload = self._transition_job(
                    job_id,
                    "scheduled",
                    updated_at=updated_at,
                )
                if isinstance(scheduled_payload, Mapping):
                    updated_at = scheduled_payload.get("updated_at")  # type: ignore[assignment]
            run_now = getattr(server, "run_job_now", None)
            if callable(run_now):
                context = {"tenant_id": getattr(self.ATLAS, "tenant_id", "default")}
                try:
                    payload = run_now(
                        job_id,
                        context=context,
                        expected_updated_at=updated_at,
                    )
                except Exception:
                    logger.error("Failed to enqueue immediate run for job %s", job_id, exc_info=True)
                    raise

                notifier = getattr(self, "show_success_toast", None)
                if callable(notifier):
                    notifier("Job run queued")
                return dict(payload) if isinstance(payload, Mapping) else {}

            if scheduled_payload is not None:
                return scheduled_payload

            target = "running"
        elif status == "draft":
            target = "scheduled"
        elif normalized_mode == "resume":
            return self.resume_job(job_id, current_status, updated_at)
        else:
            target = "running"

        return self._transition_job(job_id, target, updated_at=updated_at)

    def resume_job(
        self, job_id: str, current_status: str, updated_at: str | None
    ) -> Mapping[str, Any]:
        server = getattr(self.ATLAS, "server", None)
        resume_schedule = getattr(server, "resume_job_schedule", None)
        if callable(resume_schedule):
            context = {"tenant_id": getattr(self.ATLAS, "tenant_id", "default")}
            try:
                payload = resume_schedule(
                    job_id,
                    context=context,
                    expected_updated_at=updated_at,
                )
            except Exception:
                logger.error("Failed to resume job schedule %s", job_id, exc_info=True)
                raise

            notifier = getattr(self, "show_success_toast", None)
            if callable(notifier):
                notifier("Job schedule resumed")
            return dict(payload) if isinstance(payload, Mapping) else {}

        return self._transition_job(job_id, "scheduled", updated_at=updated_at)

    def pause_job(
        self, job_id: str, current_status: str, updated_at: str | None
    ) -> Mapping[str, Any]:
        server = getattr(self.ATLAS, "server", None)
        pause_schedule = getattr(server, "pause_job_schedule", None)
        if callable(pause_schedule):
            context = {"tenant_id": getattr(self.ATLAS, "tenant_id", "default")}
            try:
                payload = pause_schedule(
                    job_id,
                    context=context,
                    expected_updated_at=updated_at,
                )
            except Exception:
                logger.error("Failed to pause job schedule %s", job_id, exc_info=True)
                raise

            notifier = getattr(self, "show_success_toast", None)
            if callable(notifier):
                notifier("Job schedule paused")
            return dict(payload) if isinstance(payload, Mapping) else {}

        status = (current_status or "").lower()
        if status == "scheduled":
            target = "cancelled"
        elif status == "running":
            target = "cancelled"
        else:
            target = "cancelled"
        return self._transition_job(job_id, target, updated_at=updated_at)

    def rerun_job(
        self, job_id: str, current_status: str, updated_at: str | None
    ) -> Mapping[str, Any]:
        server = getattr(self.ATLAS, "server", None)
        rerun = getattr(server, "rerun_job", None)
        if not callable(rerun):
            return self._transition_job(job_id, "running", updated_at=updated_at)

        context = {"tenant_id": getattr(self.ATLAS, "tenant_id", "default")}
        try:
            payload = rerun(
                job_id,
                context=context,
                expected_updated_at=updated_at,
            )
        except Exception:
            logger.error("Failed to rerun job %s", job_id, exc_info=True)
            raise

        notifier = getattr(self, "show_success_toast", None)
        if callable(notifier):
            notifier("Job rerun queued")
        return dict(payload) if isinstance(payload, Mapping) else {}

    def show_accounts_page(self) -> None:
        if not self._ensure_initialized():
            return

        result = self._open_or_focus_page("accounts", "Account")
        if result is not None:
            self.sidebar.set_active_item("accounts")

    def _open_tool_in_persona(self, tool_name: str) -> None:
        persona_name = self._get_active_persona_name()
        if not persona_name:
            self._notify_persona_warning("Select a persona before configuring tools.")
            return

        if not self.persona_management.ensure_persona_settings(persona_name):
            self._notify_persona_warning("Unable to open persona settings.")
            return

        GLib.idle_add(self._focus_persona_tool, tool_name)

    def _open_skill_in_persona(self, skill_name: str) -> None:
        persona_name = self._get_active_persona_name()
        if not persona_name:
            self._notify_persona_warning("Select a persona before configuring skills.")
            return

        if not self.persona_management.ensure_persona_settings(persona_name):
            self._notify_persona_warning("Unable to open persona settings.")
            return

        GLib.idle_add(self._focus_persona_skill, skill_name)

    def close_application(self, *_args) -> bool:
        logger.debug("Closing application")
        app = self.get_application()
        if app:
            app.quit()
        return False

    # ------------------------------------------------------------------
    # Page management helpers
    # ------------------------------------------------------------------
    def _open_or_focus_page(self, page_id: str, title: str) -> Gtk.Widget | None:
        widget = self._pages.get(page_id)
        if widget is not None:
            self._focus_page(widget)
            return widget

        factory = self._page_factories.get(page_id)
        if not callable(factory):
            logger.error("No factory registered for %s", page_id)
            self.show_error_dialog(f"Unable to open {title} page")
            return None

        try:
            result = factory()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to build %s page: %s", page_id, exc, exc_info=True)
            self.show_error_dialog(str(exc) or f"Unable to open {title} page")
            return None

        controller: Any | None = None
        if isinstance(result, tuple):
            widget, controller = result
        else:
            widget = result
            controller = None

        if widget is None:
            return None

        widget.set_hexpand(True)
        widget.set_vexpand(True)
        self._reset_page_scroll(widget)
        tab_label = self._create_tab_header(title, page_id)
        page_index = self.notebook.append_page(widget, tab_label)
        self.notebook.set_tab_reorderable(widget, True)
        self.notebook.set_current_page(page_index)
        self._pages[page_id] = widget
        if controller is not None:
            self._page_controllers[page_id] = controller
        return widget

    def _reset_page_scroll(self, widget: Gtk.Widget) -> None:
        scrollers: list[Gtk.ScrolledWindow] = []

        def _collect_scrollers(root: Gtk.Widget | None) -> None:
            if root is None:
                return
            if isinstance(root, Gtk.ScrolledWindow):
                scrollers.append(root)
                return
            get_first_child = getattr(root, "get_first_child", None)
            get_next_sibling = getattr(root, "get_next_sibling", None)
            if not callable(get_first_child) or not callable(get_next_sibling):
                return
            child = get_first_child()
            while child is not None:
                _collect_scrollers(child)
                child = get_next_sibling()

        _collect_scrollers(widget)

        for scroller in scrollers:
            def _reset_scroll(sc=scroller) -> bool:
                vadj = sc.get_vadjustment()
                if vadj is not None:
                    vadj.set_value(vadj.get_lower())
                return False

            GLib.idle_add(_reset_scroll)

    def _focus_page(self, widget: Gtk.Widget) -> None:
        page_index = self.notebook.page_num(widget)
        if page_index != -1:
            self.notebook.set_current_page(page_index)

    def _close_page(self, page_id: str) -> None:
        widget = self._pages.pop(page_id, None)
        if widget is None:
            return
        try:
            self.notebook.remove(widget)
        except Exception:  # pragma: no cover - fallback for older GTK stubs
            page_index = self.notebook.page_num(widget)
            if page_index != -1:
                self.notebook.remove_page(page_index)

        controller = self._page_controllers.pop(page_id, None)
        if controller is not None:
            close_request = getattr(controller, "_on_close_request", None)
            if callable(close_request):
                try:
                    close_request()
                except Exception:  # pragma: no cover - defensive cleanup
                    logger.debug("Error during controller close for %s", page_id, exc_info=True)
        if page_id == "providers":
            self.provider_management._provider_page = None
        elif page_id == "personas":
            self.persona_management._persona_page = None

    def _create_tab_header(self, title: str, page_id: str) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        label = Gtk.Label(label=title)
        label.set_xalign(0.0)
        label.set_yalign(0.5)
        box.append(label)

        close_btn = Gtk.Button()
        close_btn.add_css_class("flat")
        close_btn.set_tooltip_text(f"Close {title} tab")
        icon = Gtk.Image.new_from_icon_name("window-close")
        icon.set_pixel_size(12)
        close_btn.set_child(icon)
        close_btn.connect("clicked", lambda _btn: self._close_page(page_id))
        box.append(close_btn)

        return box

    def _ensure_initialized(self) -> bool:
        if self.ATLAS.is_initialized():
            return True
        self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")
        return False

    def _focus_tool_if_requested(self, tool_name: str | None) -> None:
        if tool_name:
            GLib.idle_add(self._focus_tool_in_manager, tool_name)

    def _focus_task_if_requested(self, task_id: str | None) -> None:
        if task_id:
            GLib.idle_add(self._focus_task_in_manager, task_id)

    def _focus_job_if_requested(self, job_id: str | None) -> None:
        if job_id:
            GLib.idle_add(self._focus_job_in_manager, job_id)

    def _focus_skill_if_requested(self, skill_name: str | None) -> None:
        if skill_name:
            GLib.idle_add(self._focus_skill_in_manager, skill_name)

    def _focus_conversation_if_requested(
        self, page: Gtk.Widget, conversation_id: str | None
    ) -> None:
        if not conversation_id:
            return
        focus = getattr(page, "focus_conversation", None)
        if callable(focus):
            GLib.idle_add(focus, conversation_id)

    def _focus_tool_in_manager(self, tool_name: str) -> bool:
        if not self.tool_management.focus_tool(tool_name):
            logger.debug("Tool '%s' could not be focused in the workspace", tool_name)
        return False

    def _focus_task_in_manager(self, task_id: str) -> bool:
        if not self.task_management.focus_task(task_id):
            logger.debug("Task '%s' could not be focused in the workspace", task_id)
        return False

    def _focus_skill_in_manager(self, skill_name: str) -> bool:
        if not self.skill_management.focus_skill(skill_name):
            logger.debug("Skill '%s' could not be focused in the workspace", skill_name)
        return False

    def _focus_persona_tool(self, tool_name: str) -> bool:
        if not self.persona_management.focus_tools_tab(tool_name):
            self._notify_persona_warning(
                f"Tool '{tool_name}' is not available in the active persona."
            )
        return False

    def _focus_persona_skill(self, skill_name: str) -> bool:
        if not self.persona_management.focus_skills_tab(skill_name):
            self._notify_persona_warning(
                f"Skill '{skill_name}' is not available in the active persona."
            )
        return False

    def _get_active_persona_name(self) -> str | None:
        getter = getattr(self.ATLAS, "get_active_persona_name", None)
        if not callable(getter):
            return None

        try:
            persona = getter()
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.error("Failed to determine active persona: %s", exc, exc_info=True)
            return None

        if persona:
            return str(persona)
        return None

    def _notify_persona_warning(self, message: str) -> None:
        notifier = getattr(self.ATLAS, "show_persona_message", None)
        if callable(notifier):
            try:
                notifier("warning", message)
                return
            except Exception:  # pragma: no cover - defensive logging only
                logger.debug("Failed to send persona warning: %s", message, exc_info=True)

        self.show_error_dialog(message)

    # ------------------------------------------------------------------
    # Dialog helpers
    # ------------------------------------------------------------------
    def show_error_dialog(self, message: str) -> None:
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Initialization Error",
        )
        self._apply_shared_styles(dialog)
        # ``Gtk.MessageDialog.format_secondary_text`` was removed in GTK4.
        # Prefer ``set_secondary_text`` when available and fall back to
        # directly assigning the property for compatibility with GTK 4.x.
        set_secondary = getattr(dialog, "set_secondary_text", None)
        if callable(set_secondary):
            set_secondary(message)
        else:
            try:
                dialog.props.secondary_text = message
            except Exception:  # pragma: no cover - property unavailable
                pass
        dialog.connect("response", lambda d, r: d.destroy())
        dialog.present()

    def _apply_shared_styles(self, widget: Gtk.Widget) -> None:
        try:
            apply_css()
        except Exception:  # pragma: no cover
            pass

        get_context = getattr(widget, "get_style_context", None)
        if not callable(get_context):
            return

        context = get_context()
        if context is None:
            return

        add_class = getattr(context, "add_class", None)
        if not callable(add_class):
            return

        for css_class in ("chat-page", "sidebar"):
            try:
                add_class(css_class)
            except Exception:
                continue


class _NavigationSidebar(Gtk.Box):
    """Vertical navigation used along the left edge of the main window."""

    def __init__(self, main_window: MainWindow) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.main_window = main_window
        self.ATLAS = main_window.ATLAS
        self.persona_management = main_window.persona_management
        self.provider_management = main_window.provider_management
        self.tool_management = main_window.tool_management
        self.skill_management = main_window.skill_management
        self.task_management = main_window.task_management
        self.job_management = main_window.job_management
        self.calendar_management = main_window.calendar_management

        self.set_margin_top(4)
        self.set_margin_bottom(4)
        self.set_margin_start(4)
        self.set_margin_end(4)
        self.set_valign(Gtk.Align.FILL)
        self.set_size_request(96, -1)
        self.set_hexpand(False)
        self.set_halign(Gtk.Align.START)

        self._nav_items: Dict[str, Gtk.ListBoxRow] = {}
        self._row_actions: Dict[Gtk.ListBoxRow, Callable[[], None]] = {}
        self._active_nav_id: str | None = None
        self._history_limit = 20
        self._history_rows: Dict[str, Gtk.ListBoxRow] = {}
        self._history_placeholder_row: Gtk.ListBoxRow | None = None
        self._history_header_row: Gtk.ListBoxRow | None = None
        self._history_listener: Optional[Callable[[Dict[str, Any]], None]] = None
        self.connect("unrealize", self._on_unrealize)

        primary_listbox = Gtk.ListBox()
        primary_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        primary_listbox.set_accessible_role(Gtk.AccessibleRole.LIST)
        primary_listbox.add_css_class("sidebar-nav")
        primary_listbox.set_hexpand(False)
        primary_listbox.set_halign(Gtk.Align.START)
        self._primary_listbox = primary_listbox
        primary_listbox.connect("row-activated", self._on_row_activated)
        self.append(primary_listbox)

        self._create_nav_item(
            "providers",
            "Providers",
            self.main_window.show_provider_menu,
            tooltip="Providers",
        )
        self._create_nav_item(
            "chat",
            "Chat",
            self.main_window.show_chat_page,
            tooltip="Chat",
        )
        self._create_nav_item(
            "docs",
            "Docs",
            self.main_window.show_docs_page,
            tooltip="Documentation",
        )
        self._create_nav_item(
            "tools",
            "Tools",
            self.main_window.show_tools_menu,
            tooltip="Tools",
        )
        self._create_nav_item(
            "tasks",
            "Tasks",
            self.main_window.show_task_workspace,
            tooltip="Tasks",
        )
        self._create_nav_item(
            "jobs",
            "Jobs",
            self.main_window.show_job_workspace,
            tooltip="Jobs",
        )
        self._create_nav_item(
            None,
            "New job",
            self.main_window.create_new_job,
            tooltip="Create a new job",
        )
        self._create_nav_item(
            "budget",
            "Budget",
            self.main_window.show_budget_workspace,
            tooltip="Budget Manager",
        )
        self._create_nav_item(
            "calendar",
            "Calendar",
            self.main_window.show_calendar_workspace,
            tooltip="Calendar Manager",
        )
        self._create_nav_item(
            "skills",
            "Skills",
            self.main_window.show_skills_menu,
            tooltip="Skills",
        )
        self._create_nav_item(
            "speech",
            "Speech",
            self.main_window.show_speech_settings,
            tooltip="Speech",
        )
        self._create_nav_item(
            "personas",
            "Personas",
            self.main_window.show_persona_menu,
            tooltip="Personas",
        )

        personas_history_divider = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        personas_history_divider.add_css_class("sidebar-divider")
        personas_history_divider.set_margin_top(10)
        personas_history_divider.set_margin_bottom(1)
        self.append(personas_history_divider)

        history_listbox = Gtk.ListBox()
        history_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        history_listbox.set_accessible_role(Gtk.AccessibleRole.LIST)
        history_listbox.add_css_class("sidebar-nav")
        history_listbox.set_hexpand(False)
        history_listbox.set_halign(Gtk.Align.START)
        history_listbox.connect("row-activated", self._on_row_activated)
        self._history_listbox = history_listbox

        history_scroller = Gtk.ScrolledWindow()
        history_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        history_scroller.set_hexpand(False)
        history_scroller.set_halign(Gtk.Align.FILL)
        history_scroller.set_vexpand(True)
        history_scroller.set_child(history_listbox)
        self.append(history_scroller)

        self._history_header_row = self._create_nav_item(
            None,
            "History",
            lambda: self.main_window.show_conversation_history_page(),
            tooltip="History",
            container=history_listbox,
            margin_start=3,
            margin_top=3,
            label_css_classes=["history-nav-label"],
        )

        self._refresh_history_sidebar()
        listener = getattr(self.ATLAS, "add_conversation_history_listener", None)
        if callable(listener):
            callback = lambda event: GLib.idle_add(self._handle_history_event, event)
            listener(callback)
            self._history_listener = callback

        # RAG quick toggle section
        rag_separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        rag_separator.add_css_class("sidebar-divider")
        rag_separator.set_margin_top(10)
        rag_separator.set_margin_bottom(4)
        self.append(rag_separator)

        rag_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        rag_box.set_margin_start(12)
        rag_box.set_margin_end(12)
        rag_box.set_margin_bottom(4)
        
        rag_label = Gtk.Label(label="RAG")
        rag_label.set_xalign(0.0)
        rag_label.set_hexpand(True)
        rag_label.set_tooltip_text("Enable knowledge base context retrieval")
        rag_box.append(rag_label)
        
        # KB Manager button
        kb_manager_btn = Gtk.Button.new_from_icon_name("folder-documents-symbolic")
        kb_manager_btn.add_css_class("flat")
        kb_manager_btn.add_css_class("circular")
        kb_manager_btn.set_tooltip_text("Open Knowledge Base Manager")
        kb_manager_btn.connect("clicked", self._on_kb_manager_clicked)
        rag_box.append(kb_manager_btn)
        
        self._rag_switch = Gtk.Switch()
        self._rag_switch.set_valign(Gtk.Align.CENTER)
        self._rag_switch.set_tooltip_text("Toggle RAG on/off")
        self._rag_switch.connect("notify::active", self._on_rag_toggled)
        rag_box.append(self._rag_switch)
        
        self.append(rag_box)
        
        # Load initial RAG state
        self._load_rag_state()

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.add_css_class("sidebar-divider")
        separator.set_margin_top(6)
        separator.set_margin_bottom(10)
        self.append(separator)

        footer_listbox = Gtk.ListBox()
        footer_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        footer_listbox.set_accessible_role(Gtk.AccessibleRole.LIST)
        footer_listbox.add_css_class("sidebar-nav")
        footer_listbox.set_hexpand(False)
        footer_listbox.set_halign(Gtk.Align.START)
        footer_listbox.connect("row-activated", self._on_row_activated)
        self._footer_listbox = footer_listbox
        self.append(footer_listbox)

        self._create_nav_item(
            "accounts",
            "Account",
            self.main_window.show_accounts_page,
            tooltip="Account",
            container=footer_listbox,
        )
        self._create_nav_item(
            "settings",
            "Settings",
            self.main_window.show_settings_page,
            tooltip="Settings",
            container=footer_listbox,
        )

    # ------------------------------------------------------------------
    def _create_nav_item(
        self,
        nav_id: str | None,
        label: str,
        callback: Callable[[], None],
        tooltip: str | None = None,
        container: Gtk.ListBox | None = None,
        *,
        margin_top: int = 3,
        margin_bottom: int = 3,
        margin_start: int = 12,
        margin_end: int = 12,
        label_css_classes: Sequence[str] | None = None,
    ) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        row.set_accessible_role(Gtk.AccessibleRole.LIST_ITEM)
        if hasattr(row, "set_accessible_name"):
            row.set_accessible_name(label)
        elif hasattr(row, "update_property"):
            try:
                row.update_property(
                    Gtk.AccessibleProperty.LABEL, GLib.Variant.new_string(label)
                )
            except TypeError as exc:  # pragma: no cover - defensive fallback
                logger.debug(
                    "Gtk.ListBoxRow.update_property failed; skipping accessible label: %s",
                    exc,
                )
        else:
            logger.debug(
                "Gtk.ListBoxRow does not support accessible names; skipping label assignment"
            )
        row.set_focusable(True)
        row.add_css_class("sidebar-nav-item")
        row.set_tooltip_text(tooltip)

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        box.set_margin_top(margin_top)
        box.set_margin_bottom(margin_bottom)
        box.set_margin_start(margin_start)
        box.set_margin_end(margin_end)
        box.set_hexpand(False)

        text = Gtk.Label(label=label)
        text.set_xalign(0.0)
        text.set_halign(Gtk.Align.START)
        text.set_hexpand(False)
        if label_css_classes:
            for css_class in label_css_classes:
                text.add_css_class(css_class)
        box.append(text)

        row.set_child(box)
        row.set_activatable(True)

        gesture = Gtk.GestureClick()
        gesture.connect("released", lambda _gesture, _n_press, _x, _y: callback())
        row.add_controller(gesture)

        self._row_actions[row] = callback
        target = container if container is not None else self._primary_listbox
        target.append(row)

        if nav_id:
            self._nav_items[nav_id] = row

        return row

    def _load_rag_state(self) -> None:
        """Load the current RAG enabled state from ConfigManager."""
        try:
            from core.config import ConfigManager
            config = ConfigManager()
            enabled = config.is_rag_enabled()
            self._rag_switch.set_active(enabled)
        except Exception as exc:
            logger.debug("Failed to load RAG state: %s", exc)

    def _on_rag_toggled(self, switch: Gtk.Switch, _pspec) -> None:
        """Handle RAG toggle switch changes."""
        enabled = switch.get_active()
        try:
            from core.config import ConfigManager
            config = ConfigManager()
            config.set_rag_enabled(enabled)
            logger.info("RAG %s via sidebar toggle", "enabled" if enabled else "disabled")
        except Exception as exc:
            logger.warning("Failed to toggle RAG: %s", exc)
            # Revert the switch on error
            GLib.idle_add(lambda: switch.set_active(not enabled))

    def _on_kb_manager_clicked(self, button: Gtk.Button) -> None:
        """Handle Knowledge Base Manager button click."""
        try:
            from GTKUI.KnowledgeBase import KnowledgeBaseManager

            # Get config manager from ATLAS
            config_manager = getattr(self.ATLAS, "config_manager", None)
            
            # Try to get knowledge store from storage manager
            knowledge_store = None
            storage_manager = getattr(self.ATLAS, "_storage_manager", None)
            if storage_manager is not None:
                knowledge_store = getattr(storage_manager, "knowledge_store", None)
            
            # Try to get RAG service
            rag_service = getattr(self.ATLAS, "rag_service", None)

            manager = KnowledgeBaseManager(
                config_manager=config_manager,
                knowledge_store=knowledge_store,
                rag_service=rag_service,
                parent=self.main_window,
            )
            manager.present()
        except Exception as exc:
            logger.error("Failed to open KB Manager: %s", exc, exc_info=True)

    def _handle_history_event(self, _event: Mapping[str, Any]) -> bool:
        self._refresh_history_sidebar()
        return False

    def _refresh_history_sidebar(self) -> None:
        listbox = getattr(self, "_history_listbox", None)
        if listbox is None:
            return

        fetcher = getattr(self.ATLAS, "get_recent_conversations", None)
        repository = getattr(self.ATLAS, "conversation_repository", None)
        records: List[Dict[str, Any]] = []
        fetched_successfully = False
        if callable(fetcher):
            try:
                result = fetcher(limit=self._history_limit)
                if isinstance(result, list):
                    records = result
                fetched_successfully = True
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Failed to refresh history sidebar: %s", exc, exc_info=True)

        needs_fallback = repository is None or not fetched_successfully or not records
        if needs_fallback:
            synthetic = self._build_session_history_record()
            if synthetic:
                records = [synthetic]
        self._populate_history_sidebar(records)

    def _build_session_history_record(self) -> Optional[Dict[str, Any]]:
        session = getattr(self.ATLAS, "chat_session", None)
        if session is None:
            return None

        conversation_id: Optional[str] = None
        getter = getattr(session, "get_conversation_id", None)
        if callable(getter):
            try:
                result = getter()
                if isinstance(result, str):
                    conversation_id = result
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Failed to determine active conversation id: %s", exc, exc_info=True)
        if not conversation_id and hasattr(session, "conversation_id"):
            attr = getattr(session, "conversation_id", None)
            if isinstance(attr, str):
                conversation_id = attr

        text_id = str(conversation_id).strip() if conversation_id else ""
        if not text_id:
            return None

        created_at: Optional[str] = None
        updated_at: Optional[str] = None
        history = getattr(session, "conversation_history", None)
        if isinstance(history, Sequence):
            for entry in history:
                if isinstance(entry, Mapping):
                    timestamp = entry.get("timestamp")
                    if timestamp:
                        created_at = str(timestamp)
                        break
            for entry in reversed(history):
                if isinstance(entry, Mapping):
                    timestamp = entry.get("timestamp")
                    if timestamp:
                        updated_at = str(timestamp)
                        break

        if not created_at:
            created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        record: Dict[str, Any] = {
            "id": text_id,
            "title": "Current conversation",
            "created_at": created_at,
            "metadata": {"source": "session"},
        }
        if updated_at:
            record["updated_at"] = updated_at
        return record

    def _populate_history_sidebar(self, records: Sequence[Mapping[str, Any]]) -> None:
        listbox = getattr(self, "_history_listbox", None)
        if listbox is None:
            return

        children_getter = getattr(listbox, "get_children", None)
        if callable(children_getter):
            result = children_getter()
            rows: List[Any] = list(result) if isinstance(result, (list, tuple)) else []
        else:
            rows = []
            child = listbox.get_first_child()
            while child is not None:
                rows.append(child)
                child = child.get_next_sibling()

        for row in rows:
            if row is self._history_header_row:
                continue
            listbox.remove(row)

        self._history_rows.clear()
        self._history_placeholder_row = None

        if not records:
            placeholder_row = Gtk.ListBoxRow()
            placeholder_row.set_activatable(False)
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            box.set_margin_top(4)
            box.set_margin_bottom(4)
            box.set_margin_start(12)
            box.set_margin_end(12)
            label = Gtk.Label(label="No saved conversations yet.")
            label.set_xalign(0.0)
            label.add_css_class("history-placeholder")
            box.append(label)
            placeholder_row.set_child(box)
            listbox.append(placeholder_row)
            self._history_placeholder_row = placeholder_row
            return

        for record in records[: self._history_limit]:
            if not isinstance(record, Mapping):
                continue

            identifier = str(record.get("id") or "")
            row = Gtk.ListBoxRow()
            row.set_accessible_role(Gtk.AccessibleRole.LIST_ITEM)
            row.add_css_class("sidebar-nav-item")
            row.set_activatable(True)

            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            box.set_margin_top(4)
            box.set_margin_bottom(4)
            box.set_margin_start(12)
            box.set_margin_end(12)

            title_label = Gtk.Label(label=self._format_history_title(record))
            title_label.set_xalign(0.0)
            title_label.add_css_class("history-nav-label")
            box.append(title_label)

            timestamp_value = record.get("created_at") or record.get("timestamp")
            timestamp_text = self._format_history_timestamp(timestamp_value)
            if not timestamp_text:
                fallback_label = record.get("timestamp_label")
                if isinstance(fallback_label, str) and fallback_label.strip():
                    timestamp_text = fallback_label.strip()
            if timestamp_text:
                subtitle = Gtk.Label(label=timestamp_text)
                subtitle.set_xalign(0.0)
                subtitle.add_css_class("history-nav-subtitle")
                box.append(subtitle)

            row.set_child(box)

            action = lambda conv_id=identifier: self.main_window.show_conversation_history_page(
                conv_id
            )
            self._row_actions[row] = action

            gesture = Gtk.GestureClick()
            gesture.connect(
                "released",
                lambda _gesture, _n, _x, _y, conv_id=identifier: self.main_window.show_conversation_history_page(
                    conv_id
                ),
            )
            row.add_controller(gesture)

            listbox.append(row)
            if identifier:
                self._history_rows[identifier] = row

    def _format_history_title(self, record: Mapping[str, Any]) -> str:
        title = record.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()

        metadata = record.get("metadata")
        if isinstance(metadata, Mapping):
            candidate = metadata.get("title") or metadata.get("name")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        identifier = record.get("id")
        text = str(identifier) if identifier is not None else ""
        return f"Conversation {text[:8]}" if text else "Conversation"

    def _format_history_timestamp(self, value: Any) -> str:
        if not value:
            return ""
        text = str(value)
        if text.endswith("Z"):
            text = text.replace("Z", "+00:00")
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return ""
        if moment.tzinfo is not None:
            moment = moment.astimezone()
        return moment.strftime("%Y-%m-%d %H:%M")

    def _on_unrealize(self, *_args) -> None:
        remover = getattr(self.ATLAS, "remove_conversation_history_listener", None)
        if callable(remover) and self._history_listener is not None:
            remover(self._history_listener)
            self._history_listener = None

    def _on_row_activated(self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow) -> None:
        callback = self._row_actions.get(row)
        if callable(callback):
            callback()

    def set_active_item(self, nav_id: str | None) -> None:
        if nav_id == self._active_nav_id:
            return

        if self._active_nav_id and self._active_nav_id in self._nav_items:
            previous = self._nav_items[self._active_nav_id]
            previous.remove_css_class("active")

        self._active_nav_id = nav_id if nav_id in self._nav_items else None

        if self._active_nav_id:
            current = self._nav_items[self._active_nav_id]
            current.add_css_class("active")


# Backwards compatibility for imports expecting ``Sidebar``
Sidebar = MainWindow
