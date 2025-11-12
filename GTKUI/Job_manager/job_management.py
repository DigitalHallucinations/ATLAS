"""GTK job management workspace used by the sidebar."""

from __future__ import annotations

import copy
import json
import logging
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from GTKUI.Task_manager.widgets import (
    append_badge,
    clear_container,
    create_badge,
    create_badge_container,
    sync_badge_section,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _JobEntry:
    """Normalized job record returned by the ATLAS server."""

    job_id: str
    name: str
    description: str
    status: str
    owner_id: Optional[str]
    conversation_id: Optional[str]
    personas: Tuple[str, ...]
    recurrence: Mapping[str, Any]
    metadata: Dict[str, Any]
    created_at: Optional[str]
    updated_at: Optional[str]


class JobManagement:
    """Controller responsible for rendering the job management workspace."""

    _PERSONA_UNASSIGNED = "__unassigned__"
    _RECURRENCE_RECURRING = "__recurring__"
    _RECURRENCE_ONE_OFF = "__one_off__"
    _STATUS_SEQUENCE: Tuple[str, ...] = (
        "draft",
        "scheduled",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    )

    def __init__(self, atlas: Any, parent_window: Any) -> None:
        self.ATLAS = atlas
        self.parent_window = parent_window

        self._widget: Optional[Gtk.Widget] = None
        self._job_list: Optional[Gtk.ListBox] = None
        self._persona_filter_combo: Optional[Gtk.ComboBoxText] = None
        self._status_filter_combo: Optional[Gtk.ComboBoxText] = None
        self._recurrence_filter_combo: Optional[Gtk.ComboBoxText] = None
        self._title_label: Optional[Gtk.Label] = None
        self._status_label: Optional[Gtk.Label] = None
        self._description_label: Optional[Gtk.Label] = None
        self._persona_label: Optional[Gtk.Label] = None
        self._owner_label: Optional[Gtk.Label] = None
        self._conversation_label: Optional[Gtk.Label] = None
        self._recurrence_label: Optional[Gtk.Label] = None
        self._next_run_label: Optional[Gtk.Label] = None
        self._updated_label: Optional[Gtk.Label] = None
        self._persona_badges: Optional[Gtk.Widget] = None
        self._schedule_box: Optional[Gtk.Widget] = None
        self._linked_tasks_box: Optional[Gtk.Widget] = None
        self._link_task_button: Optional[Gtk.Button] = None
        self._escalation_box: Optional[Gtk.Widget] = None
        self._action_box: Optional[Gtk.Box] = None
        self._start_button: Optional[Gtk.Button] = None
        self._pause_button: Optional[Gtk.Button] = None
        self._rerun_button: Optional[Gtk.Button] = None
        self._new_job_button: Optional[Gtk.Button] = None
        self._job_creation_dialog: Optional[Gtk.Widget] = None

        self._entries: List[_JobEntry] = []
        self._entry_lookup: Dict[str, _JobEntry] = {}
        self._display_entries: List[_JobEntry] = []
        self._row_lookup: Dict[str, Gtk.ListBoxRow] = {}
        self._persona_filter: Optional[str] = None
        self._status_filter: Optional[str] = None
        self._recurrence_filter: Optional[str] = None
        self._persona_option_lookup: List[Optional[str]] = []
        self._status_option_lookup: List[Optional[str]] = []
        self._recurrence_option_lookup: List[Optional[str]] = []
        self._active_job: Optional[str] = None
        self._pending_focus_job: Optional[str] = None
        self._refresh_pending = False
        self._bus_subscriptions: List[Any] = []
        self._start_confirmation_widget: Optional[Gtk.Widget] = None
        self._start_confirmation_handler: Optional[Callable[[str], None]] = None
        self._start_confirmation_choices: Dict[str, Callable[[], None]] = {}
        self._suppress_filter_refresh = False

        # Cached detail data for tests and scripted validation
        self._current_schedule_badges: List[str] = []
        self._current_linked_task_badges: List[str] = []
        self._current_escalation_badges: List[str] = []
        self._linked_task_action_lookup: List[Tuple[Mapping[str, Any], Gtk.Button]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_embeddable_widget(self) -> Gtk.Widget:
        """Return the workspace widget, creating it on first use."""

        if self._widget is None:
            self._widget = self._build_workspace()
            self._subscribe_to_bus()
            self._refresh_state()
        else:
            self._refresh_state()
        return self._widget

    def focus_job(self, job_id: str) -> bool:
        """Ensure ``job_id`` is visible and selected."""

        identifier = str(job_id or "").strip()
        if not identifier:
            return False

        if identifier not in self._entry_lookup:
            self._pending_focus_job = identifier
            self._refresh_state()
        if identifier not in self._entry_lookup:
            return False

        self._pending_focus_job = identifier
        self._select_job(identifier)
        return True

    def _on_close_request(self) -> None:
        """Clean up resources when the workspace tab is closed."""

        for subscription in list(self._bus_subscriptions):
            try:
                subscription.cancel()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.debug("Failed to cancel job bus subscription", exc_info=True)
            finally:
                self._bus_subscriptions.remove(subscription)

    # ------------------------------------------------------------------
    # Workspace construction
    # ------------------------------------------------------------------
    def _build_workspace(self) -> Gtk.Widget:
        root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        for setter_name in ("set_margin_top", "set_margin_bottom", "set_margin_start", "set_margin_end"):
            setter = getattr(root, setter_name, None)
            if callable(setter):
                try:
                    setter(12)
                except Exception:  # pragma: no cover - GTK fallback
                    continue

        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        sidebar.set_hexpand(False)
        sidebar.set_vexpand(True)
        root.append(sidebar)

        filter_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        sidebar.append(filter_box)

        persona_combo = Gtk.ComboBoxText()
        persona_combo.connect("changed", self._on_persona_filter_changed)
        filter_box.append(persona_combo)
        self._persona_filter_combo = persona_combo

        status_combo = Gtk.ComboBoxText()
        status_combo.connect("changed", self._on_status_filter_changed)
        filter_box.append(status_combo)
        self._status_filter_combo = status_combo
        self._populate_status_filter_options()

        recurrence_combo = Gtk.ComboBoxText()
        recurrence_combo.connect("changed", self._on_recurrence_filter_changed)
        filter_box.append(recurrence_combo)
        self._recurrence_filter_combo = recurrence_combo
        self._populate_recurrence_filter_options()

        new_job_button = Gtk.Button(label="New job…")
        new_job_button.set_hexpand(True)
        new_job_button.connect("clicked", self._on_new_job_clicked)
        sidebar.append(new_job_button)
        self._new_job_button = new_job_button

        scroller = Gtk.ScrolledWindow()
        scroller.set_hexpand(False)
        scroller.set_vexpand(True)
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        sidebar.append(scroller)

        listbox = Gtk.ListBox()
        listbox.set_selection_mode(Gtk.SelectionMode.SINGLE)
        listbox.connect("row-selected", self._on_row_selected)
        scroller.set_child(listbox)
        self._job_list = listbox

        detail_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        detail_panel.set_hexpand(True)
        detail_panel.set_vexpand(True)
        root.append(detail_panel)

        title_label = Gtk.Label()
        title_label.set_xalign(0.0)
        title_label.set_wrap(True)
        title_label.add_css_class("heading") if hasattr(title_label, "add_css_class") else None
        detail_panel.append(title_label)
        self._title_label = title_label

        status_label = Gtk.Label()
        status_label.set_xalign(0.0)
        detail_panel.append(status_label)
        self._status_label = status_label

        info_grid = Gtk.Grid(column_spacing=12, row_spacing=6)
        detail_panel.append(info_grid)

        self._persona_label = self._add_info_row(info_grid, 0, "Personas")
        self._owner_label = self._add_info_row(info_grid, 1, "Owner")
        self._conversation_label = self._add_info_row(info_grid, 2, "Conversation")
        self._recurrence_label = self._add_info_row(info_grid, 3, "Recurrence")
        self._next_run_label = self._add_info_row(info_grid, 4, "Next run")
        self._updated_label = self._add_info_row(info_grid, 5, "Last updated")

        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        detail_panel.append(action_box)
        self._action_box = action_box

        start_button = Gtk.Button(label="Start")
        start_button.connect("clicked", self._on_start_clicked)
        action_box.append(start_button)
        self._start_button = start_button

        pause_button = Gtk.Button(label="Pause")
        pause_button.connect("clicked", self._on_pause_clicked)
        action_box.append(pause_button)
        self._pause_button = pause_button

        rerun_button = Gtk.Button(label="Rerun")
        rerun_button.connect("clicked", self._on_rerun_clicked)
        action_box.append(rerun_button)
        self._rerun_button = rerun_button

        description_header = Gtk.Label(label="Description")
        description_header.set_xalign(0.0)
        detail_panel.append(description_header)

        description_label = Gtk.Label()
        description_label.set_wrap(True)
        description_label.set_xalign(0.0)
        detail_panel.append(description_label)
        self._description_label = description_label

        self._persona_badges = self._add_badge_section(detail_panel, "Persona roster")
        self._schedule_box = self._add_badge_section(detail_panel, "Schedule")
        self._linked_tasks_box = self._add_linked_tasks_section(detail_panel)
        self._escalation_box = self._add_badge_section(detail_panel, "Escalation policy")

        detail_panel.connect("destroy", lambda *_args: self._on_close_request())
        return root

    def _on_new_job_clicked(self, _button: Gtk.Button) -> None:
        self.prompt_new_job()

    def prompt_new_job(self) -> None:
        details = self._collect_new_job_details()
        if details is not None:
            self._handle_new_job_submission(details)
            return
        self._show_new_job_dialog()

    def _collect_new_job_details(self) -> Optional[Mapping[str, Any]]:
        prompt = getattr(self.parent_window, "prompt_new_job_details", None)
        if callable(prompt):
            try:
                result = prompt()
            except Exception as exc:
                logger.error("Failed to collect new job details: %s", exc, exc_info=True)
                self._handle_backend_error("Unable to gather job details.")
                return None
            if isinstance(result, Mapping):
                return result
        return None

    def _show_new_job_dialog(self) -> None:
        if self._job_creation_dialog is not None:
            presenter = getattr(self._job_creation_dialog, "present", None)
            if callable(presenter):
                presenter()
            return
        dialog = _NewJobDialog(self)
        dialog.connect("destroy", lambda *_args: self._on_job_dialog_closed(dialog))
        presenter = getattr(dialog, "present", None)
        if callable(presenter):
            presenter()
        self._job_creation_dialog = dialog

    def _on_job_dialog_closed(self, dialog: Gtk.Widget) -> None:
        if dialog is self._job_creation_dialog:
            self._job_creation_dialog = None

    def _handle_new_job_submission(self, details: Mapping[str, Any]) -> bool:
        payload = self._normalize_new_job_payload(details)
        if payload is None:
            return False

        create_job = getattr(self.ATLAS, "create_job", None)
        if not callable(create_job):
            self._handle_backend_error("Job creation is not supported by this backend.")
            return False

        try:
            response = create_job(**payload)
        except (TypeError, ValueError) as exc:
            logger.error("Invalid job payload: %s", exc, exc_info=True)
            self._handle_backend_error(str(exc) or "Unable to create the job.")
            return False
        except Exception as exc:
            logger.error("Failed to create job: %s", exc, exc_info=True)
            self._handle_backend_error("Unable to create the job.")
            return False

        job_id: Optional[str] = None
        if isinstance(response, Mapping):
            raw_id = response.get("id") or response.get("job_id")
            if raw_id:
                job_id = str(raw_id).strip() or None

        if job_id:
            self._pending_focus_job = job_id
        self._refresh_state()
        if job_id:
            self.focus_job(job_id)
        self._notify_success("Job created.")
        return True

    def _normalize_new_job_payload(
        self, details: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        name = str(details.get("name") or "").strip()
        if not name:
            self._handle_backend_error("Job name is required.")
            return None

        description = str(details.get("description") or "").strip() or None

        metadata_value = details.get("metadata")
        metadata_payload = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}

        roster_value = details.get("personas") or details.get("persona_roster")
        roster, roster_error = self._parse_persona_roster_field(roster_value)
        if roster_error:
            self._handle_backend_error(roster_error)
            return None

        schedule_source = details.get("schedule") or details.get("schedule_metadata")
        if schedule_source is None and "schedule" in metadata_payload:
            schedule_source = metadata_payload.pop("schedule")
        schedule, schedule_error = self._parse_schedule_metadata_field(schedule_source)
        if schedule_error:
            self._handle_backend_error(schedule_error)
            return None

        payload: Dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if roster:
            payload["personas"] = roster
        if schedule is not None:
            payload["schedule"] = schedule
        if metadata_payload:
            payload["metadata"] = metadata_payload
        return payload

    def _parse_persona_roster_field(
        self, value: Any
    ) -> Tuple[Tuple[str, ...], Optional[str]]:
        if value is None:
            return tuple(), None

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return tuple(), None
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                items = [segment.strip() for segment in text.split(",") if segment.strip()]
                return tuple(dict.fromkeys(items)), None
            return self._parse_persona_roster_field(decoded)

        if isinstance(value, Mapping):
            raw = value.get("name") or value.get("persona") or value.get("id")
            text = str(raw).strip() if raw else ""
            return ((text,) if text else tuple(), None)

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            combined: List[str] = []
            for entry in value:
                roster, error = self._parse_persona_roster_field(entry)
                if error:
                    return tuple(), error
                combined.extend(roster)
            deduped: List[str] = []
            seen: set[str] = set()
            for persona in combined:
                if persona not in seen:
                    deduped.append(persona)
                    seen.add(persona)
            return tuple(deduped), None

        text = str(value).strip()
        return ((text,) if text else tuple(), None)

    def _parse_schedule_metadata_field(
        self, value: Any
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if value is None:
            return None, None

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None, None
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                return None, f"Schedule metadata must be valid JSON: {exc.msg}"
            value = decoded

        if isinstance(value, Mapping):
            return dict(value), None

        return None, "Schedule metadata must be a mapping or JSON object."



    def _add_info_row(self, grid: Gtk.Grid, row: int, title: str) -> Gtk.Label:
        label_widget = Gtk.Label(label=title)
        label_widget.set_xalign(0.0)
        label_widget.set_yalign(0.5)
        grid.attach(label_widget, 0, row, 1, 1)

        value_label = Gtk.Label()
        value_label.set_xalign(0.0)
        value_label.set_yalign(0.5)
        value_label.set_wrap(True)
        grid.attach(value_label, 1, row, 1, 1)
        return value_label

    def _add_badge_section(self, container: Gtk.Box, title: str) -> Gtk.Widget:
        header = Gtk.Label(label=title)
        header.set_xalign(0.0)
        container.append(header)

        badge_container = create_badge_container()
        badge_container.set_visible(False)
        container.append(badge_container)
        return badge_container

    def _add_linked_tasks_section(self, container: Gtk.Box) -> Gtk.Widget:
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        header_label = Gtk.Label(label="Linked tasks")
        header_label.set_xalign(0.0)
        header_box.append(header_label)

        link_button = Gtk.Button(label="Link task…")
        link_button.connect("clicked", self._on_link_task_clicked)
        header_box.append(link_button)
        self._link_task_button = link_button

        container.append(header_box)

        badge_container = create_badge_container()
        badge_container.set_visible(False)
        container.append(badge_container)
        return badge_container

    # ------------------------------------------------------------------
    # Data loading and synchronization
    # ------------------------------------------------------------------
    def _refresh_state(self) -> None:
        entries = self._load_job_entries()
        self._entries = entries
        self._entry_lookup = {entry.job_id: entry for entry in entries}
        self._suppress_filter_refresh = True
        try:
            self._populate_persona_filter_options()
            self._populate_status_filter_options()
            self._populate_recurrence_filter_options()
        finally:
            self._suppress_filter_refresh = False
        self._rebuild_job_list()

    def _schedule_refresh(self) -> None:
        if self._refresh_pending:
            return
        self._refresh_pending = True

        def _callback() -> bool:
            self._refresh_pending = False
            if self._widget is None:
                return False
            self._refresh_state()
            return False

        try:
            GLib.idle_add(_callback)
        except Exception:  # pragma: no cover - fallback when GLib idle is unavailable
            _callback()

    def _load_job_entries(self) -> List[_JobEntry]:
        server = getattr(self.ATLAS, "server", None)
        list_jobs = getattr(server, "list_jobs", None)
        if not callable(list_jobs):
            logger.warning("ATLAS server does not expose list_jobs; returning empty workspace")
            return []

        params: Dict[str, Any] = {}

        context = {"tenant_id": getattr(self.ATLAS, "tenant_id", "default")}
        try:
            response = list_jobs(params, context=context)
        except Exception as exc:
            logger.error("Failed to load job list: %s", exc, exc_info=True)
            self._handle_backend_error("Unable to load jobs from ATLAS.")
            return []

        items: Iterable[Any]
        if isinstance(response, Mapping):
            items = response.get("items", [])
        elif isinstance(response, Iterable):
            items = response
        else:
            items = []

        entries: List[_JobEntry] = []
        for raw_entry in items:
            normalized = self._normalize_entry(raw_entry)
            if normalized is not None:
                entries.append(normalized)
        return entries

    def _normalize_entry(self, payload: Any) -> Optional[_JobEntry]:
        if not isinstance(payload, Mapping):
            return None

        raw_id = payload.get("id")
        identifier = str(raw_id).strip() if raw_id else ""
        if not identifier:
            return None

        metadata_value = payload.get("metadata")
        if isinstance(metadata_value, Mapping):
            metadata = dict(metadata_value)
        else:
            metadata = {}

        schedule_detail = payload.get("schedule")
        if isinstance(schedule_detail, Mapping):
            metadata["schedule"] = dict(schedule_detail)
            schedule_meta = schedule_detail.get("metadata")
            if isinstance(schedule_meta, Mapping):
                state = schedule_meta.get("state")
                if state is not None:
                    metadata["schedule_state"] = state

        personas = self._extract_personas(metadata)
        recurrence = self._extract_recurrence(metadata)

        description_source = payload.get("description") or metadata.get("description")
        description = str(description_source).strip() if description_source else "No description provided."

        status_value = str(payload.get("status") or "draft").strip().lower()

        return _JobEntry(
            job_id=identifier,
            name=str(payload.get("name") or metadata.get("name") or "Untitled job").strip() or "Untitled job",
            description=description,
            status=status_value,
            owner_id=str(payload.get("owner_id") or "").strip() or None,
            conversation_id=str(payload.get("conversation_id") or "").strip() or None,
            personas=personas,
            recurrence=recurrence,
            metadata=metadata,
            created_at=str(payload.get("created_at") or "").strip() or None,
            updated_at=str(payload.get("updated_at") or "").strip() or None,
        )

    def _extract_personas(self, metadata: Mapping[str, Any]) -> Tuple[str, ...]:
        roster: Iterable[Any]
        if "personas" in metadata:
            roster = metadata.get("personas", [])
        elif "persona_roster" in metadata:
            roster = metadata.get("persona_roster", [])
        elif "roster" in metadata:
            roster = metadata.get("roster", [])
        else:
            single = metadata.get("persona") or metadata.get("persona_name")
            roster = [single] if single else []

        personas: List[str] = []
        if isinstance(roster, Iterable) and not isinstance(roster, (str, bytes)):
            for entry in roster:
                if isinstance(entry, Mapping):
                    name = entry.get("name") or entry.get("persona") or entry.get("id")
                    if name:
                        text = str(name).strip()
                        if text:
                            personas.append(text)
                elif entry is not None:
                    text = str(entry).strip()
                    if text:
                        personas.append(text)
        return tuple(personas)

    def _extract_recurrence(self, metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        value = metadata.get("recurrence")
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------
    def _populate_persona_filter_options(self) -> None:
        combo = self._persona_filter_combo
        if combo is None:
            return
        previous_guard = self._suppress_filter_refresh
        self._suppress_filter_refresh = True
        try:
            if hasattr(combo, "remove_all"):
                combo.remove_all()

            personas = sorted({persona for entry in self._entries for persona in entry.personas})
            has_unassigned = any(not entry.personas for entry in self._entries)

            options: List[Tuple[Optional[str], str]] = [(None, "All personas")]
            options.extend((name, name) for name in personas)
            if has_unassigned:
                options.append((self._PERSONA_UNASSIGNED, "Unassigned"))

            self._persona_option_lookup = []
            for index, (value, label) in enumerate(options):
                combo.append_text(label)
                self._persona_option_lookup.append(value)
                if value == self._persona_filter:
                    combo.set_active(index)

            if combo.get_active_text() is None:
                combo.set_active(0)
                self._persona_filter = None
        finally:
            self._suppress_filter_refresh = previous_guard

    def _populate_status_filter_options(self) -> None:
        combo = self._status_filter_combo
        if combo is None:
            return
        previous_guard = self._suppress_filter_refresh
        self._suppress_filter_refresh = True
        try:
            if hasattr(combo, "remove_all"):
                combo.remove_all()

            options: List[Tuple[Optional[str], str]] = [(None, "All statuses")]
            options.extend((status, self._format_status(status)) for status in self._STATUS_SEQUENCE)

            self._status_option_lookup = []
            for index, (value, label) in enumerate(options):
                combo.append_text(label)
                self._status_option_lookup.append(value)
                if value == self._status_filter:
                    combo.set_active(index)

            if combo.get_active_text() is None:
                combo.set_active(0)
                self._status_filter = None
        finally:
            self._suppress_filter_refresh = previous_guard

    def _populate_recurrence_filter_options(self) -> None:
        combo = self._recurrence_filter_combo
        if combo is None:
            return
        previous_guard = self._suppress_filter_refresh
        self._suppress_filter_refresh = True
        try:
            if hasattr(combo, "remove_all"):
                combo.remove_all()

            options: List[Tuple[Optional[str], str]] = [(None, "All recurrence patterns")]
            if any(self._entry_is_recurring(entry) for entry in self._entries):
                options.append((self._RECURRENCE_RECURRING, "Recurring"))
            if any(not self._entry_is_recurring(entry) for entry in self._entries):
                options.append((self._RECURRENCE_ONE_OFF, "One-off"))

            self._recurrence_option_lookup = []
            for index, (value, label) in enumerate(options):
                combo.append_text(label)
                self._recurrence_option_lookup.append(value)
                if value == self._recurrence_filter:
                    combo.set_active(index)

            if combo.get_active_text() is None:
                combo.set_active(0)
                self._recurrence_filter = None
        finally:
            self._suppress_filter_refresh = previous_guard

    def _entry_is_recurring(self, entry: _JobEntry) -> bool:
        return bool(entry.recurrence)

    def _on_persona_filter_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._suppress_filter_refresh:
            return
        index = self._combo_active_index(combo)
        if 0 <= index < len(self._persona_option_lookup):
            self._persona_filter = self._persona_option_lookup[index]
        else:
            self._persona_filter = None
        self._rebuild_job_list()

    def _on_status_filter_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._suppress_filter_refresh:
            return
        index = self._combo_active_index(combo)
        if 0 <= index < len(self._status_option_lookup):
            self._status_filter = self._status_option_lookup[index]
        else:
            self._status_filter = None
        self._refresh_state()
        persona_combo = self._persona_filter_combo
        if persona_combo is not None:
            self._persona_filter = None
            persona_combo.set_active(0)
            self._on_persona_filter_changed(persona_combo)

    def _on_recurrence_filter_changed(self, combo: Gtk.ComboBoxText) -> None:
        if self._suppress_filter_refresh:
            return
        index = self._combo_active_index(combo)
        if 0 <= index < len(self._recurrence_option_lookup):
            self._recurrence_filter = self._recurrence_option_lookup[index]
        else:
            self._recurrence_filter = None
        status_combo = self._status_filter_combo
        if status_combo is not None:
            self._status_filter = None
            status_combo.set_active(0)
        self._rebuild_job_list()

    def _combo_active_index(self, combo: Gtk.ComboBoxText) -> int:
        getter = getattr(combo, "get_active", None)
        index: Optional[int]
        if callable(getter):
            try:
                value = getter()
            except Exception:  # pragma: no cover - GTK fallback
                value = None
            index = value if isinstance(value, int) else None
        else:
            index = None

        if index is not None:
            return index

        active_text = getattr(combo, "get_active_text", lambda: None)()
        items = list(getattr(combo, "_items", []))
        if active_text in items:
            return items.index(active_text)
        return -1

    # ------------------------------------------------------------------
    # List management
    # ------------------------------------------------------------------
    def _rebuild_job_list(self) -> None:
        listbox = self._job_list
        if listbox is None:
            return
        clear_container(listbox)
        setattr(listbox, "_selected_row", None)
        self._row_lookup.clear()

        filtered: List[_JobEntry] = []
        for entry in self._entries:
            if not self._passes_filters(entry):
                continue
            filtered.append(entry)

        self._display_entries = filtered
        for entry in filtered:
            row = self._create_list_row(entry)
            listbox.append(row)
            self._row_lookup[entry.job_id] = row

        target = self._pending_focus_job or self._active_job
        self._pending_focus_job = None
        if target and target in {entry.job_id for entry in filtered}:
            self._select_job(target)
        elif filtered:
            self._select_job(filtered[0].job_id)
        else:
            self._select_job(None)

    def _passes_filters(self, entry: _JobEntry) -> bool:
        persona_filter = self._persona_filter
        if persona_filter == self._PERSONA_UNASSIGNED:
            if entry.personas:
                return False
        elif persona_filter and persona_filter not in entry.personas:
            return False

        status_filter = self._status_filter
        if status_filter and entry.status != status_filter:
            return False

        recurrence_filter = self._recurrence_filter
        if recurrence_filter == self._RECURRENCE_RECURRING and not self._entry_is_recurring(entry):
            return False
        if recurrence_filter == self._RECURRENCE_ONE_OFF and self._entry_is_recurring(entry):
            return False
        return True

    def _create_list_row(self, entry: _JobEntry) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        row._job_id = entry.job_id  # type: ignore[attr-defined]

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        box.set_margin_top(6)
        box.set_margin_bottom(6)
        box.set_margin_start(6)
        box.set_margin_end(6)

        title = Gtk.Label(label=entry.name or "Untitled job")
        title.set_xalign(0.0)
        title.set_wrap(True)
        box.append(title)

        info = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        info.set_hexpand(True)
        box.append(info)

        status_badge = create_badge(self._format_status(entry.status), self._status_css(entry.status))
        info.append(status_badge)

        if entry.personas:
            persona_label = Gtk.Label(label=", ".join(entry.personas))
            persona_label.set_xalign(0.0)
            info.append(persona_label)
        else:
            persona_label = Gtk.Label(label="Unassigned")
            persona_label.set_xalign(0.0)
            persona_label.add_css_class("dim-label") if hasattr(persona_label, "add_css_class") else None
            info.append(persona_label)

        recurrence_text = self._format_recurrence(entry)
        recurrence_label = Gtk.Label(label=recurrence_text)
        recurrence_label.set_xalign(0.0)
        recurrence_label.add_css_class("dim-label") if hasattr(recurrence_label, "add_css_class") else None
        box.append(recurrence_label)

        summary = Gtk.Label(label=entry.description)
        summary.set_xalign(0.0)
        summary.set_wrap(True)
        box.append(summary)

        row.set_child(box)
        return row

    def _format_recurrence(self, entry: _JobEntry) -> str:
        recurrence = entry.recurrence
        if not recurrence:
            return "One-off"
        frequency = str(recurrence.get("frequency") or recurrence.get("type") or "").strip()
        interval = str(recurrence.get("interval") or recurrence.get("cadence") or "").strip()
        if frequency and interval:
            return f"{frequency.title()} ({interval})"
        if frequency:
            return frequency.title()
        if interval:
            return interval
        return "Recurring"

    def _on_row_selected(self, listbox: Gtk.ListBox, row: Optional[Gtk.ListBoxRow]) -> None:
        setattr(listbox, "_selected_row", row)
        if row is None:
            self._select_job(None)
            return
        job_id = getattr(row, "_job_id", None)
        if isinstance(job_id, str):
            self._select_job(job_id)

    def _select_job(self, job_id: Optional[str]) -> None:
        if not job_id:
            self._active_job = None
            self._update_detail(None)
            return
        entry = self._entry_lookup.get(job_id)
        if entry is None:
            return
        self._active_job = job_id
        row = self._row_lookup.get(job_id)
        listbox = self._job_list
        if row is not None and listbox is not None:
            getter = getattr(listbox, "get_selected_row", None)
            if callable(getter):
                try:
                    current = getter()
                except Exception:  # pragma: no cover - GTK fallback
                    current = None
            else:
                current = getattr(listbox, "_selected_row", None)
            if current is not row:
                selector = getattr(listbox, "select_row", None)
                if callable(selector):
                    try:
                        selector(row)
                    except Exception:  # pragma: no cover - GTK fallback
                        setattr(listbox, "_selected_row", row)
                else:
                    setattr(listbox, "_selected_row", row)
        self._update_detail(entry)

    # ------------------------------------------------------------------
    # Detail panel
    # ------------------------------------------------------------------
    def _update_detail(self, entry: Optional[_JobEntry]) -> None:
        if entry is None:
            self._set_label(self._title_label, "No job selected")
            self._set_label(self._status_label, "")
            self._set_label(self._description_label, "Select a job to view details.")
            self._set_label(self._persona_label, "-")
            self._set_label(self._owner_label, "-")
            self._set_label(self._conversation_label, "-")
            self._set_label(self._recurrence_label, "-")
            self._set_label(self._next_run_label, "-")
            self._set_label(self._updated_label, "-")
            sync_badge_section(self._persona_badges, [])
            sync_badge_section(self._schedule_box, [])
            sync_badge_section(self._linked_tasks_box, [])
            sync_badge_section(self._escalation_box, [])
            self._current_schedule_badges = []
            self._current_linked_task_badges = []
            self._current_escalation_badges = []
            self._update_action_buttons(None)
            return

        self._set_label(self._title_label, entry.name)
        self._set_label(self._status_label, f"Status: {self._format_status(entry.status)}")
        self._set_label(self._description_label, entry.description)

        roster_text = ", ".join(entry.personas) if entry.personas else "Unassigned"
        self._set_label(self._persona_label, roster_text)
        self._set_label(self._owner_label, entry.owner_id or "-")
        self._set_label(self._conversation_label, entry.conversation_id or "-")
        self._set_label(self._recurrence_label, self._format_recurrence(entry))
        self._set_label(self._updated_label, entry.updated_at or "-")

        detail, linked_tasks = self._load_job_detail(entry)
        schedule_info = detail.get("schedule") if isinstance(detail, Mapping) else None
        next_run = "-"
        schedule_badges: List[Tuple[str, Sequence[str]]] = []
        if isinstance(schedule_info, Mapping):
            schedule_type = str(schedule_info.get("schedule_type") or schedule_info.get("type") or "").strip()
            if schedule_type:
                schedule_badges.append((f"Type: {schedule_type}", ("tag-badge", "status-unknown")))
            expression = str(schedule_info.get("expression") or "").strip()
            if expression:
                schedule_badges.append((f"Expression: {expression}", ("tag-badge", "status-warning")))
            timezone = str(schedule_info.get("timezone") or "").strip()
            if timezone:
                schedule_badges.append((f"Timezone: {timezone}", ("tag-badge", "status-unknown")))
            next_run_value = str(schedule_info.get("next_run_at") or "").strip()
            if next_run_value:
                schedule_badges.append((f"Next run: {next_run_value}", ("tag-badge", "status-ok")))
                next_run = next_run_value
        self._set_label(self._next_run_label, next_run)
        sync_badge_section(self._schedule_box, schedule_badges, fallback="No schedule configured")
        self._current_schedule_badges = [text for text, _css in schedule_badges]

        persona_badges = []
        for persona in entry.personas:
            persona_badges.append((persona, ("tag-badge", "status-warning")))
        sync_badge_section(self._persona_badges, persona_badges, fallback="No personas assigned")

        linked_badges: List[Tuple[str, Sequence[str], Mapping[str, Any]]] = []
        for record in linked_tasks:
            if not isinstance(record, Mapping):
                continue
            task = record.get("task")
            relationship = str(record.get("relationship_type") or "").strip()
            metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
            summary = None
            if isinstance(metadata, Mapping):
                summary = metadata.get("summary")

            if isinstance(task, Mapping):
                title = str(task.get("title") or task.get("name") or "Linked task").strip()
                status = str(task.get("status") or "").strip().lower()
                badge_text = title
                if status:
                    badge_text = f"{title} ({self._format_status(status)})"
                if relationship:
                    badge_text = f"{badge_text} – {relationship}"
                if summary:
                    badge_text = f"{badge_text} – {summary}"
                linked_badges.append((badge_text, self._task_status_css(status), record))
            else:
                task_id = str(record.get("task_id") or "").strip() or "Unknown task"
                linked_badges.append((task_id, ("tag-badge", "status-unknown"), record))
        self._sync_linked_task_badges(linked_badges)
        self._current_linked_task_badges = [text for text, _css, _record in linked_badges]

        metadata = detail.get("metadata") if isinstance(detail, Mapping) else entry.metadata
        if not isinstance(metadata, Mapping):
            metadata = entry.metadata
        elif isinstance(detail, Mapping):
            normalized = self._normalize_entry(detail)
            if normalized is not None:
                entry.status = normalized.status
                entry.metadata = normalized.metadata
                entry.updated_at = normalized.updated_at
                entry.personas = normalized.personas
                entry.recurrence = normalized.recurrence
                metadata = entry.metadata
        escalation = metadata.get("escalation_policy") if isinstance(metadata, Mapping) else None
        escalation_badges: List[Tuple[str, Sequence[str]]] = []
        if isinstance(escalation, Mapping):
            level = str(escalation.get("level") or "").strip()
            contact = str(escalation.get("contact") or "").strip()
            timeframe = str(escalation.get("timeframe") or escalation.get("sla") or "").strip()
            if level:
                escalation_badges.append((f"Level: {level}", ("tag-badge", "status-warning")))
            if contact:
                escalation_badges.append((f"Contact: {contact}", ("tag-badge", "status-unknown")))
            if timeframe:
                escalation_badges.append((f"Timeframe: {timeframe}", ("tag-badge", "status-unknown")))
        sync_badge_section(self._escalation_box, escalation_badges, fallback="No escalation policy configured")
        self._current_escalation_badges = [text for text, _css in escalation_badges]

        self._update_action_buttons(entry)

    def _load_job_detail(self, entry: _JobEntry) -> Tuple[Mapping[str, Any], List[Mapping[str, Any]]]:
        server = getattr(self.ATLAS, "server", None)
        get_job = getattr(server, "get_job", None)
        list_tasks = getattr(server, "list_job_tasks", None)
        context = {"tenant_id": getattr(self.ATLAS, "tenant_id", "default")}

        detail: Mapping[str, Any] = {}
        if callable(get_job):
            try:
                payload = get_job(
                    entry.job_id,
                    context=context,
                    include_schedule=True,
                    include_runs=False,
                    include_events=False,
                )
                if isinstance(payload, Mapping):
                    detail = payload
            except Exception as exc:
                logger.debug("Failed to load job detail for %s: %s", entry.job_id, exc, exc_info=True)

        linked: List[Mapping[str, Any]] = []
        if callable(list_tasks):
            try:
                payload = list_tasks(entry.job_id, context=context)
                if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
                    for item in payload:
                        if isinstance(item, Mapping):
                            linked.append(dict(item))
            except Exception as exc:
                logger.debug("Failed to load linked tasks for %s: %s", entry.job_id, exc, exc_info=True)
        return detail, linked

    def _sync_linked_task_badges(
        self,
        badges: Sequence[Tuple[str, Sequence[str], Mapping[str, Any]]],
    ) -> None:
        container = self._linked_tasks_box
        if container is None:
            return

        clear_container(container)
        self._linked_task_action_lookup = []

        has_badges = False
        for text, css_classes, record in badges:
            badge_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            badge = create_badge(text, css_classes)
            badge_row.append(badge)

            record_copy: Mapping[str, Any] = copy.deepcopy(record)

            unlink_button = Gtk.Button(label="Unlink")
            unlink_button.connect(
                "clicked",
                lambda _button, payload=record_copy: self._on_unlink_task_clicked(_button, payload),
            )
            setattr(unlink_button, "_link_record", record_copy)
            badge_row.append(unlink_button)

            append_badge(container, badge_row)
            self._linked_task_action_lookup.append((record_copy, unlink_button))
            has_badges = True

        if not has_badges:
            fallback = create_badge("No linked tasks", ("tag-badge", "status-unknown"))
            append_badge(container, fallback)
            has_badges = True

        if hasattr(container, "set_visible"):
            container.set_visible(has_badges)

    def _set_label(self, widget: Optional[Gtk.Label], text: str) -> None:
        if widget is None:
            return
        try:
            widget.set_text(text)
        except Exception:  # pragma: no cover - GTK fallback
            setattr(widget, "label", text)

    def _format_status(self, status: str) -> str:
        if not status:
            return "Unknown"
        text = status.replace("_", " ").strip()
        if not text:
            return "Unknown"
        return text.capitalize()

    def _status_css(self, status: str) -> Tuple[str, ...]:
        normalized = (status or "").lower()
        if normalized in {"succeeded"}:
            return ("tag-badge", "status-ok")
        if normalized in {"failed", "cancelled"}:
            return ("tag-badge", "status-error")
        if normalized in {"running", "scheduled"}:
            return ("tag-badge", "status-warning")
        return ("tag-badge", "status-unknown")

    def _task_status_css(self, status: str) -> Tuple[str, ...]:
        normalized = (status or "").lower()
        if normalized in {"done", "succeeded", "complete"}:
            return ("tag-badge", "status-ok")
        if normalized in {"cancelled", "failed"}:
            return ("tag-badge", "status-error")
        if normalized in {"in_progress", "running", "review", "ready"}:
            return ("tag-badge", "status-warning")
        return ("tag-badge", "status-unknown")

    def _update_action_buttons(self, entry: Optional[_JobEntry]) -> None:
        start = self._start_button
        pause = self._pause_button
        rerun = self._rerun_button
        for button in (start, pause, rerun):
            if button is not None:
                button.set_visible(False)
                button.set_sensitive(False)
        if entry is None:
            return

        status = entry.status
        if start is not None and status in {"draft", "scheduled"}:
            start.set_visible(True)
            start.set_sensitive(True)
        if pause is not None and status in {"scheduled", "running"}:
            pause.set_visible(True)
            pause.set_sensitive(True)
        if rerun is not None and status in {"succeeded", "failed", "cancelled"}:
            rerun.set_visible(True)
            rerun.set_sensitive(True)

    def _on_start_clicked(self, button: Gtk.Button) -> None:
        entry = None
        if self._active_job and self._active_job in self._entry_lookup:
            entry = self._entry_lookup[self._active_job]
        if entry is None:
            return

        if self._should_offer_resume(entry):
            self._present_start_confirmation(button, entry)
            return

        self._trigger_action("start")

    def _on_pause_clicked(self, _button: Gtk.Button) -> None:
        self._trigger_action("pause")

    def _on_rerun_clicked(self, _button: Gtk.Button) -> None:
        self._trigger_action("rerun")

    def _on_link_task_clicked(self, _button: Gtk.Button) -> None:
        if not self._active_job or self._active_job not in self._entry_lookup:
            return

        entry = self._entry_lookup[self._active_job]
        payload = self._prompt_link_job_task(entry.job_id)
        if not isinstance(payload, Mapping):
            return

        task_id = payload.get("task_id")
        if not task_id:
            return

        try:
            self._call_link_job_task(entry, payload)
        except Exception as exc:
            logger.error(
                "Failed to link task %s to job %s: %s",
                task_id,
                entry.job_id,
                exc,
                exc_info=True,
            )
            self._handle_backend_error("Unable to link the task to this job.")
            return

        self._notify_success("Task linked to job.")
        self._pending_focus_job = entry.job_id
        self._refresh_state()

    def _on_unlink_task_clicked(self, _button: Gtk.Button, record: Mapping[str, Any]) -> None:
        if not self._active_job or self._active_job not in self._entry_lookup:
            return

        entry = self._entry_lookup[self._active_job]
        link_id = record.get("id")
        task_id = record.get("task_id")

        try:
            self._call_unlink_job_task(entry, link_id=link_id, task_id=task_id)
        except Exception as exc:
            logger.error(
                "Failed to unlink task %s from job %s: %s",
                task_id,
                entry.job_id,
                exc,
                exc_info=True,
            )
            self._handle_backend_error("Unable to unlink the task from this job.")
            return

        self._notify_success("Task unlinked from job.")
        self._pending_focus_job = entry.job_id
        self._refresh_state()

    def _should_offer_resume(self, entry: _JobEntry) -> bool:
        status = (entry.status or "").strip().lower()
        if status == "scheduled":
            return True

        metadata = entry.metadata or {}
        schedule_info = metadata.get("schedule") if isinstance(metadata.get("schedule"), Mapping) else {}
        schedule_meta = schedule_info.get("metadata") if isinstance(schedule_info.get("metadata"), Mapping) else {}
        schedule_state = (
            str(
                schedule_meta.get("state")
                or schedule_info.get("state")
                or schedule_info.get("status")
                or metadata.get("schedule_state")
                or metadata.get("schedule_status")
                or ""
            )
            .strip()
            .lower()
        )
        if schedule_state in {"paused", "suspended"}:
            return True

        paused_flag = schedule_info.get("paused")
        if paused_flag is None:
            paused_flag = metadata.get("schedule_paused")
        if isinstance(paused_flag, bool):
            return paused_flag
        if isinstance(paused_flag, str):
            return paused_flag.strip().lower() in {"true", "1", "yes", "paused"}
        return False

    def _present_start_confirmation(self, anchor: Gtk.Widget, entry: _JobEntry) -> None:
        self._dismiss_start_confirmation()

        def handler(choice: str) -> None:
            if choice == "resume":
                self._trigger_action("start", variant="resume")
            elif choice == "run_now":
                self._trigger_action("start", variant="run_now")
            else:
                self._trigger_action("start")

        self._start_confirmation_handler = handler
        self._start_confirmation_choices = {
            "resume": lambda: self._confirm_start_choice("resume"),
            "run_now": lambda: self._confirm_start_choice("run_now"),
        }
        widget = self._build_start_confirmation_widget(anchor, entry)
        self._start_confirmation_widget = widget
        if widget is not None:
            presenter = getattr(widget, "present", None)
            if callable(presenter):
                try:
                    presenter()
                except Exception:  # pragma: no cover - GTK fallback
                    logger.debug("Failed to present start confirmation widget", exc_info=True)

    def _build_start_confirmation_widget(
        self, anchor: Gtk.Widget, entry: _JobEntry
    ) -> Optional[Gtk.Widget]:
        message = "This job already has a schedule."
        if self._should_offer_resume(entry):
            message = (
                "This job is scheduled. Resume the schedule or run immediately?"
            )

        popover: Optional[Gtk.Widget] = None
        popover_class = getattr(Gtk, "Popover", None)
        if callable(popover_class):
            try:
                popover = popover_class()
            except Exception:  # pragma: no cover - GTK fallback
                popover = None
        if popover is not None:
            try:
                popover.set_autohide(True)
            except Exception:  # pragma: no cover - GTK compatibility
                pass
            try:
                popover.set_parent(anchor)
            except Exception:  # pragma: no cover - GTK compatibility
                pass
            content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            text = Gtk.Label(label=message)
            text.set_wrap(True)
            text.set_xalign(0.0)
            content.append(text)
            resume_btn = Gtk.Button(label="Resume schedule")
            resume_btn.connect("clicked", lambda *_: self._confirm_start_choice("resume"))
            run_btn = Gtk.Button(label="Run now")
            run_btn.connect("clicked", lambda *_: self._confirm_start_choice("run_now"))
            content.append(resume_btn)
            content.append(run_btn)
            try:
                popover.set_child(content)
            except Exception:  # pragma: no cover - GTK compatibility
                pass
            return popover

        dialog_class = getattr(Gtk, "MessageDialog", None)
        if callable(dialog_class):
            transient_parent = getattr(self.parent_window, "get_root", None)
            if callable(transient_parent):
                try:
                    parent_widget = transient_parent()
                except Exception:  # pragma: no cover - GTK fallback
                    parent_widget = None
            else:
                parent_widget = self.parent_window if isinstance(self.parent_window, Gtk.Widget) else None
            message_type = getattr(Gtk, "MessageType", types.SimpleNamespace(INFO=None))
            buttons_type = getattr(Gtk, "ButtonsType", types.SimpleNamespace(NONE=None, OK=None))
            try:
                dialog = dialog_class(
                    transient_for=parent_widget,
                    modal=True,
                    message_type=getattr(message_type, "INFO", None),
                    buttons=getattr(buttons_type, "NONE", None),
                    text=message,
                )
            except Exception:  # pragma: no cover - GTK fallback
                dialog = None
            if dialog is not None:
                response_type = getattr(Gtk, "ResponseType", types.SimpleNamespace())
                resume_response = getattr(
                    response_type,
                    "YES",
                    getattr(response_type, "ACCEPT", 1),
                )
                run_response = getattr(
                    response_type,
                    "APPLY",
                    getattr(response_type, "OK", 2),
                )
                add_button = getattr(dialog, "add_button", None)
                if callable(add_button):
                    try:
                        add_button("Resume schedule", resume_response)
                        add_button("Run now", run_response)
                    except Exception:  # pragma: no cover - GTK fallback
                        pass
                connect = getattr(dialog, "connect", None)
                if callable(connect):
                    try:
                        connect(
                            "response",
                            lambda _dlg, response: self._confirm_start_choice(
                                "resume" if response == resume_response else "run_now"
                            ),
                        )
                    except Exception:  # pragma: no cover - GTK fallback
                        pass
                return dialog

        return None

    def _confirm_start_choice(self, choice: str) -> None:
        handler = self._start_confirmation_handler
        self._dismiss_start_confirmation()
        if callable(handler):
            handler(choice)

    def _dismiss_start_confirmation(self) -> None:
        widget = self._start_confirmation_widget
        self._start_confirmation_widget = None
        self._start_confirmation_handler = None
        self._start_confirmation_choices = {}
        if widget is None:
            return
        for method_name in ("popdown", "hide", "close"):
            closer = getattr(widget, method_name, None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # pragma: no cover - GTK fallback
                    continue

    def _trigger_action(self, action: str, *, variant: Optional[str] = None) -> None:
        if not self._active_job or self._active_job not in self._entry_lookup:
            return
        entry = self._entry_lookup[self._active_job]
        handler_name = {
            "start": "start_job",
            "pause": "pause_job",
            "rerun": "rerun_job",
        }.get(action)
        if action == "start" and variant == "resume":
            handler_name = "resume_job"
        if not handler_name:
            return

        handler = getattr(self.parent_window, handler_name, None)
        if not callable(handler):
            self._handle_backend_error("Job actions are unavailable.")
            return

        try:
            if action == "start" and variant == "run_now":
                payload = handler(entry.job_id, entry.status, entry.updated_at, mode="run_now")
            else:
                payload = handler(entry.job_id, entry.status, entry.updated_at)
        except Exception as exc:
            logger.error(
                "Failed to invoke job action %s for %s: %s", action, entry.job_id, exc, exc_info=True
            )
            self._handle_backend_error("Unable to update the job status.")
            return

        if isinstance(payload, Mapping):
            normalized = self._normalize_entry(payload)
            if normalized is not None:
                self._entry_lookup[normalized.job_id] = normalized
                if normalized.job_id == self._active_job:
                    entry = normalized
        self._pending_focus_job = entry.job_id
        self._refresh_state()

    # ------------------------------------------------------------------
    # Message bus integration
    # ------------------------------------------------------------------
    def _subscribe_to_bus(self) -> None:
        if self._bus_subscriptions:
            return
        events = (
            "jobs.created",
            "jobs.updated",
            "jobs.completed",
            "job.created",
            "job.updated",
            "job.status_changed",
        )
        for event_name in events:
            try:
                handle = self.ATLAS.subscribe_event(event_name, self._handle_bus_event)
            except Exception as exc:  # pragma: no cover - subscription fallback
                logger.debug("Unable to subscribe to %s events: %s", event_name, exc, exc_info=True)
                continue
            self._bus_subscriptions.append(handle)

    async def _handle_bus_event(self, payload: Any, *_args: Any) -> None:
        job_id = None
        if isinstance(payload, Mapping):
            job_id = payload.get("job_id") or payload.get("id")
        if job_id:
            logger.debug("Received job event for %s; scheduling refresh", job_id)
        self._schedule_refresh()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    def _handle_backend_error(self, message: str) -> None:
        handler = getattr(self.parent_window, "show_error_dialog", None)
        if callable(handler):
            handler(message)
        else:
            logger.error("Job management error: %s", message)

    def _notify_success(self, message: str) -> None:
        handler = getattr(self.parent_window, "show_success_toast", None)
        if callable(handler):
            handler(message)

    def _prompt_link_job_task(self, job_id: str) -> Optional[Mapping[str, Any]]:
        prompt = getattr(self.parent_window, "prompt_link_job_task", None)
        if callable(prompt):
            try:
                result = prompt(job_id)
            except Exception as exc:
                logger.error("Failed to prompt for task link: %s", exc, exc_info=True)
                self._handle_backend_error("Unable to gather task link details.")
                return None
            if isinstance(result, Mapping):
                return result
        return None

    def _call_link_job_task(
        self,
        entry: _JobEntry,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        atlas = self.ATLAS
        handler = getattr(atlas, "link_job_task", None)
        if callable(handler):
            return handler(
                entry.job_id,
                payload.get("task_id"),
                relationship_type=payload.get("relationship_type"),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else None,
            )

        server = getattr(atlas, "server", None)
        link_route = getattr(server, "link_job_task", None)
        if not callable(link_route):
            raise RuntimeError("Task linking is unavailable.")

        request: Dict[str, Any] = {"task_id": payload.get("task_id")}
        relationship = payload.get("relationship_type")
        if relationship:
            request["relationship_type"] = relationship
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            request["metadata"] = dict(metadata)

        context = {"tenant_id": getattr(atlas, "tenant_id", "default")}
        return link_route(entry.job_id, request, context=context)

    def _call_unlink_job_task(
        self,
        entry: _JobEntry,
        *,
        link_id: Any | None,
        task_id: Any | None,
    ) -> Mapping[str, Any]:
        atlas = self.ATLAS
        handler = getattr(atlas, "unlink_job_task", None)
        if callable(handler):
            return handler(entry.job_id, link_id=link_id, task_id=task_id)

        server = getattr(atlas, "server", None)
        unlink_route = getattr(server, "unlink_job_task", None)
        if not callable(unlink_route):
            raise RuntimeError("Task unlinking is unavailable.")

        context = {"tenant_id": getattr(atlas, "tenant_id", "default")}
        return unlink_route(
            entry.job_id,
            context=context,
            link_id=link_id,
            task_id=task_id,
        )


class _NewJobDialog(Gtk.Window):
    """Simple modal dialog to gather basic job information."""

    def __init__(self, controller: JobManagement) -> None:
        super().__init__(title="New job")
        self._controller = controller

        if hasattr(self, "set_modal"):
            try:
                self.set_modal(True)
            except Exception:  # pragma: no cover - GTK fallback
                pass
        if hasattr(self, "set_default_size"):
            try:
                self.set_default_size(420, 280)
            except Exception:  # pragma: no cover - GTK fallback
                pass

        parent = getattr(controller, "parent_window", None)
        transient = getattr(self, "set_transient_for", None)
        if callable(transient) and isinstance(parent, Gtk.Widget):
            try:
                transient(parent)
            except Exception:  # pragma: no cover - GTK fallback
                pass

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        for setter_name in (
            "set_margin_top",
            "set_margin_bottom",
            "set_margin_start",
            "set_margin_end",
        ):
            setter = getattr(root, setter_name, None)
            if callable(setter):
                try:
                    setter(12)
                except Exception:  # pragma: no cover - GTK fallback
                    continue
        self.set_child(root)

        self._name_entry = self._add_entry_row(root, "Job name", placeholder="Weekly sync")
        self._description_entry = self._add_entry_row(
            root, "Description", placeholder="Describe the outcome"
        )
        self._personas_entry = self._add_entry_row(
            root,
            "Persona roster",
            placeholder="Comma-separated names or JSON list",
        )
        self._schedule_entry = self._add_entry_row(
            root,
            "Schedule metadata",
            placeholder="JSON (e.g., {\"frequency\": \"weekly\"})",
        )

        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        root.append(action_box)

        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.connect("clicked", self._on_cancel_clicked)
        action_box.append(cancel_button)

        create_button = Gtk.Button(label="Create job")
        create_button.connect("clicked", self._on_create_clicked)
        action_box.append(create_button)

    def _add_entry_row(
        self, container: Gtk.Box, label: str, *, placeholder: str = ""
    ) -> Gtk.Entry:
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        container.append(row)

        label_widget = Gtk.Label(label=label)
        label_widget.set_xalign(0.0)
        row.append(label_widget)

        entry = Gtk.Entry()
        if placeholder:
            entry.set_placeholder_text(placeholder)
        row.append(entry)
        return entry

    def _on_cancel_clicked(self, _button: Gtk.Button) -> None:
        self._close()

    def _on_create_clicked(self, _button: Gtk.Button) -> None:
        payload: Dict[str, Any] = {
            "name": self._name_entry.get_text(),
            "description": self._description_entry.get_text(),
        }
        personas = self._personas_entry.get_text()
        if personas.strip():
            payload["personas"] = personas
        schedule = self._schedule_entry.get_text()
        if schedule.strip():
            payload["schedule"] = schedule
        success = self._controller._handle_new_job_submission(payload)
        if success:
            self._close()

    def _close(self) -> None:
        closer = getattr(self, "close", None)
        if callable(closer):
            try:
                closer()
                return
            except Exception:  # pragma: no cover - GTK fallback
                pass
        destroyer = getattr(self, "destroy", None)
        if callable(destroyer):
            try:
                destroyer()
                return
            except Exception:  # pragma: no cover - GTK fallback
                pass
        self.set_visible(False)


__all__ = ["JobManagement"]

