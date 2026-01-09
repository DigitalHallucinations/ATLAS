"""Budget policy editor dialog for creating and editing budget policies."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class PolicyEditorDialog(Gtk.Dialog):
    """Dialog for creating or editing a budget policy."""

    def __init__(
        self,
        atlas: Any,
        parent: Gtk.Window,
        *,
        policy: Optional[Any] = None,
        on_save: Optional[Callable[[Any], None]] = None,
    ) -> None:
        super().__init__(
            title="Edit Policy" if policy else "New Budget Policy",
            transient_for=parent,
            modal=True,
        )
        self.ATLAS = atlas
        self._policy = policy
        self._on_save = on_save

        self.set_default_size(450, 500)
        self.add_button("Cancel", Gtk.ResponseType.CANCEL)
        self.add_button("Save", Gtk.ResponseType.OK)

        # Form widgets (initialized in _build_form)
        self._name_entry: Gtk.Entry
        self._scope_combo: Gtk.ComboBoxText
        self._scope_target_entry: Gtk.Entry
        self._period_combo: Gtk.ComboBoxText
        self._limit_entry: Gtk.Entry
        self._action_combo: Gtk.ComboBoxText
        self._warn_pct_entry: Gtk.Entry
        self._critical_pct_entry: Gtk.Entry
        self._enabled_switch: Gtk.Switch

        self._build_form()
        if policy:
            self._populate_from_policy(policy)

        self.connect("response", self._on_response)

    def _build_form(self) -> None:
        """Build the policy editor form."""
        content = self.get_content_area()
        content.set_margin_top(16)
        content.set_margin_bottom(16)
        content.set_margin_start(16)
        content.set_margin_end(16)
        content.set_spacing(12)

        # Name field
        name_box = self._create_field_row("Policy Name")
        self._name_entry = Gtk.Entry()
        self._name_entry.set_placeholder_text("e.g., Monthly Budget")
        self._name_entry.set_hexpand(True)
        name_box.append(self._name_entry)
        content.append(name_box)

        # Scope field
        scope_box = self._create_field_row("Scope")
        self._scope_combo = Gtk.ComboBoxText()
        self._scope_combo.append("global", "Global (All providers)")
        self._scope_combo.append("provider", "Provider (Specific provider)")
        self._scope_combo.append("model", "Model (Specific model)")
        self._scope_combo.append("user", "User (Specific user)")
        self._scope_combo.set_active(0)
        self._scope_combo.connect("changed", self._on_scope_changed)
        scope_box.append(self._scope_combo)
        content.append(scope_box)

        # Scope target (shown for non-global scopes)
        target_box = self._create_field_row("Scope Target")
        self._scope_target_entry = Gtk.Entry()
        self._scope_target_entry.set_placeholder_text("Provider/Model/User identifier")
        self._scope_target_entry.set_hexpand(True)
        self._scope_target_entry.set_sensitive(False)
        target_box.append(self._scope_target_entry)
        content.append(target_box)
        self._scope_target_box = target_box

        # Period field
        period_box = self._create_field_row("Budget Period")
        self._period_combo = Gtk.ComboBoxText()
        self._period_combo.append("daily", "Daily")
        self._period_combo.append("weekly", "Weekly")
        self._period_combo.append("monthly", "Monthly")
        self._period_combo.append("quarterly", "Quarterly")
        self._period_combo.append("yearly", "Yearly")
        self._period_combo.set_active(2)  # Default to monthly
        period_box.append(self._period_combo)
        content.append(period_box)

        # Limit amount field
        limit_box = self._create_field_row("Budget Limit ($)")
        self._limit_entry = Gtk.Entry()
        self._limit_entry.set_placeholder_text("100.00")
        self._limit_entry.set_input_purpose(Gtk.InputPurpose.NUMBER)
        self._limit_entry.set_hexpand(True)
        limit_box.append(self._limit_entry)
        content.append(limit_box)

        # Limit action field
        action_box = self._create_field_row("When Limit Reached")
        self._action_combo = Gtk.ComboBoxText()
        self._action_combo.append("warn", "Warn Only")
        self._action_combo.append("soft_block", "Soft Block (Allow override)")
        self._action_combo.append("hard_block", "Hard Block (Deny requests)")
        self._action_combo.set_active(0)
        action_box.append(self._action_combo)
        content.append(action_box)

        # Alert thresholds section
        threshold_header = Gtk.Label(label="Alert Thresholds")
        threshold_header.set_xalign(0.0)
        threshold_header.add_css_class("heading")
        threshold_header.set_margin_top(12)
        content.append(threshold_header)

        # Warning threshold
        warn_box = self._create_field_row("Warning (%)")
        self._warn_pct_entry = Gtk.Entry()
        self._warn_pct_entry.set_text("80")
        self._warn_pct_entry.set_max_width_chars(5)
        self._warn_pct_entry.set_input_purpose(Gtk.InputPurpose.NUMBER)
        warn_box.append(self._warn_pct_entry)
        content.append(warn_box)

        # Critical threshold
        critical_box = self._create_field_row("Critical (%)")
        self._critical_pct_entry = Gtk.Entry()
        self._critical_pct_entry.set_text("95")
        self._critical_pct_entry.set_max_width_chars(5)
        self._critical_pct_entry.set_input_purpose(Gtk.InputPurpose.NUMBER)
        critical_box.append(self._critical_pct_entry)
        content.append(critical_box)

        # Enabled switch
        enabled_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        enabled_box.set_margin_top(12)
        enabled_label = Gtk.Label(label="Policy Enabled")
        enabled_label.set_xalign(0.0)
        enabled_label.set_hexpand(True)
        enabled_box.append(enabled_label)

        self._enabled_switch = Gtk.Switch()
        self._enabled_switch.set_active(True)
        self._enabled_switch.set_valign(Gtk.Align.CENTER)
        enabled_box.append(self._enabled_switch)
        content.append(enabled_box)

    def _create_field_row(self, label_text: str) -> Gtk.Box:
        """Create a labeled field row."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        label = Gtk.Label(label=label_text)
        label.set_xalign(0.0)
        label.add_css_class("field-label")
        box.append(label)

        return box

    def _on_scope_changed(self, combo: Gtk.ComboBoxText) -> None:
        """Handle scope selection change."""
        active = combo.get_active_id()
        is_global = active == "global"

        if self._scope_target_entry:
            self._scope_target_entry.set_sensitive(not is_global)
            if is_global:
                self._scope_target_entry.set_text("")

    def _populate_from_policy(self, policy: Any) -> None:
        """Populate form fields from an existing policy."""
        if self._name_entry and hasattr(policy, "name"):
            self._name_entry.set_text(policy.name or "")

        if self._scope_combo and hasattr(policy, "scope"):
            scope_id = str(policy.scope.value).lower() if hasattr(policy.scope, "value") else str(policy.scope).lower()
            self._scope_combo.set_active_id(scope_id)

        if self._scope_target_entry and hasattr(policy, "scope_target"):
            self._scope_target_entry.set_text(policy.scope_target or "")

        if self._period_combo and hasattr(policy, "period"):
            period_id = str(policy.period.value).lower() if hasattr(policy.period, "value") else str(policy.period).lower()
            self._period_combo.set_active_id(period_id)

        if self._limit_entry and hasattr(policy, "limit_amount"):
            self._limit_entry.set_text(str(policy.limit_amount))

        if self._action_combo and hasattr(policy, "limit_action"):
            action_id = str(policy.limit_action.value).lower() if hasattr(policy.limit_action, "value") else str(policy.limit_action).lower()
            self._action_combo.set_active_id(action_id)

        if self._warn_pct_entry and hasattr(policy, "warning_threshold_pct"):
            self._warn_pct_entry.set_text(str(policy.warning_threshold_pct))

        if self._critical_pct_entry and hasattr(policy, "critical_threshold_pct"):
            self._critical_pct_entry.set_text(str(policy.critical_threshold_pct))

        if self._enabled_switch and hasattr(policy, "enabled"):
            self._enabled_switch.set_active(policy.enabled)

    def _on_response(self, dialog: Gtk.Dialog, response: int) -> None:
        """Handle dialog response."""
        if response == Gtk.ResponseType.OK:
            policy_data = self._collect_form_data()
            if policy_data and self._on_save:
                self._on_save(policy_data)
        self.close()

    def _collect_form_data(self) -> Optional[dict]:
        """Collect and validate form data."""
        data = {}

        # Validate name
        name = self._name_entry.get_text().strip() if self._name_entry else ""
        if not name:
            self._show_validation_error("Policy name is required")
            return None
        data["name"] = name

        # Collect scope
        if self._scope_combo:
            data["scope"] = self._scope_combo.get_active_id()
            if data["scope"] != "global" and self._scope_target_entry:
                target = self._scope_target_entry.get_text().strip()
                if not target:
                    self._show_validation_error("Scope target is required for non-global scopes")
                    return None
                data["scope_target"] = target

        # Collect period
        if self._period_combo:
            data["period"] = self._period_combo.get_active_id()

        # Validate and collect limit
        limit_text = self._limit_entry.get_text().strip() if self._limit_entry else ""
        try:
            limit = Decimal(limit_text)
            if limit <= 0:
                raise ValueError("Limit must be positive")
            data["limit_amount"] = limit
        except (InvalidOperation, ValueError):
            self._show_validation_error("Please enter a valid budget limit")
            return None

        # Collect action
        if self._action_combo:
            data["limit_action"] = self._action_combo.get_active_id()

        # Validate thresholds
        try:
            warn_pct = int(self._warn_pct_entry.get_text()) if self._warn_pct_entry else 80
            critical_pct = int(self._critical_pct_entry.get_text()) if self._critical_pct_entry else 95

            if not (0 <= warn_pct <= 100) or not (0 <= critical_pct <= 100):
                raise ValueError("Thresholds must be between 0 and 100")
            if warn_pct >= critical_pct:
                raise ValueError("Warning threshold must be less than critical threshold")

            data["warning_threshold_pct"] = warn_pct
            data["critical_threshold_pct"] = critical_pct
        except ValueError as exc:
            self._show_validation_error(str(exc))
            return None

        # Collect enabled state
        data["enabled"] = self._enabled_switch.get_active() if self._enabled_switch else True

        # Preserve ID if editing
        if self._policy and hasattr(self._policy, "policy_id"):
            data["policy_id"] = self._policy.policy_id

        return data

    def _show_validation_error(self, message: str) -> None:
        """Show a validation error message."""
        error_dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Validation Error",
        )
        error_dialog.format_secondary_text(message)
        error_dialog.connect("response", lambda d, _: d.close())
        error_dialog.present()


class PolicyListPanel(Gtk.Box):
    """Panel displaying list of budget policies with management actions."""

    def __init__(self, atlas: Any, parent_controller: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas
        self._parent = parent_controller

        self.set_margin_top(16)
        self.set_margin_bottom(16)
        self.set_margin_start(16)
        self.set_margin_end(16)

        # Widget (initialized in _build_ui)
        self._policy_list: Gtk.ListBox
        self._policies: list = []

        self._build_ui()
        self._refresh_policies()

    def _build_ui(self) -> None:
        """Build the policy list panel."""
        # Header with add button
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        title = Gtk.Label(label="Budget Policies")
        title.set_xalign(0.0)
        title.add_css_class("title-2")
        title.set_hexpand(True)
        header.append(title)

        add_btn = Gtk.Button.new_from_icon_name("list-add-symbolic")
        add_btn.add_css_class("suggested-action")
        add_btn.set_tooltip_text("Add new policy")
        add_btn.connect("clicked", self._on_add_policy)
        header.append(add_btn)

        self.append(header)

        # Policy list in scrolled window
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_vexpand(True)

        self._policy_list = Gtk.ListBox()
        self._policy_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self._policy_list.add_css_class("boxed-list")
        scroller.set_child(self._policy_list)

        self.append(scroller)

    def _refresh_policies(self) -> None:
        """Refresh the policy list from the budget service."""
        async def fetch_policies() -> None:
            try:
                from core.services.budget import get_policy_service
                from core.services.common import Actor

                actor = Actor(
                    type="system",
                    id="gtkui-policy-editor",
                    tenant_id="local",
                    permissions={"budget:read"},
                )

                policy_service = await get_policy_service()
                result = await policy_service.list_policies(actor, enabled_only=False)
                
                if result.success and result.data:
                    policy_dicts = []
                    for p in result.data:
                        policy_dict = {
                            "id": p.id,
                            "name": p.name,
                            "scope": p.scope.value if hasattr(p.scope, 'value') else str(p.scope),
                            "period": p.period.value if hasattr(p.period, 'value') else str(p.period),
                            "limit_amount": str(p.limit_amount),
                            "enabled": p.enabled,
                        }
                        policy_dicts.append(policy_dict)
                    GLib.idle_add(self._on_policies_loaded, policy_dicts)
                else:
                    GLib.idle_add(self._on_policies_loaded, [])
            except Exception as exc:
                logger.warning("Failed to fetch policies: %s", exc)
                GLib.idle_add(self._on_policies_loaded, [])

        asyncio.create_task(fetch_policies())

    def _on_policies_loaded(self, policies: list) -> bool:
        """Handle loaded policies on main thread."""
        # Clear existing rows
        child = self._policy_list.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._policy_list.remove(child)
            child = next_child

        if not policies:
            placeholder = Gtk.ListBoxRow()
            placeholder.set_activatable(False)
            label = Gtk.Label(label="No policies configured")
            label.add_css_class("dim-label")
            label.set_margin_top(24)
            label.set_margin_bottom(24)
            placeholder.set_child(label)
            self._policy_list.append(placeholder)
        else:
            for policy in policies:
                row = self._create_policy_row(policy)
                self._policy_list.append(row)

        return False

    def _create_policy_row(self, policy: dict) -> Gtk.ListBoxRow:
        """Create a list row for a policy."""
        row = Gtk.ListBoxRow()
        row.set_activatable(True)

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(12)
        box.set_margin_end(12)

        # Policy name and scope
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        info_box.set_hexpand(True)

        name_label = Gtk.Label(label=policy.get("name", "Unnamed"))
        name_label.set_xalign(0.0)
        name_label.add_css_class("heading")
        info_box.append(name_label)

        scope_text = f"{policy.get('scope', 'global')} • {policy.get('period', 'monthly')}"
        scope_label = Gtk.Label(label=scope_text)
        scope_label.set_xalign(0.0)
        scope_label.add_css_class("dim-label")
        info_box.append(scope_label)

        box.append(info_box)

        # Limit amount
        limit_text = f"${policy.get('limit_amount', '0.00')}"
        limit_label = Gtk.Label(label=limit_text)
        limit_label.add_css_class("numeric")
        box.append(limit_label)

        # Enabled status
        enabled = policy.get("enabled", True)
        status_label = Gtk.Label(label="Active" if enabled else "Disabled")
        status_label.add_css_class("success" if enabled else "dim-label")
        box.append(status_label)

        row.set_child(box)
        return row

    def _load_policies(self) -> bool:
        """Load policies (sync fallback)."""
        # Clear existing rows
        child = self._policy_list.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._policy_list.remove(child)
            child = next_child

        placeholder = Gtk.ListBoxRow()
        placeholder.set_activatable(False)
        label = Gtk.Label(label="No policies configured")
        label.add_css_class("dim-label")
        label.set_margin_top(24)
        label.set_margin_bottom(24)
        placeholder.set_child(label)
        self._policy_list.append(placeholder)

        return False

    def _on_add_policy(self, _button: Gtk.Button) -> None:
        """Show the add policy dialog."""
        window = self._get_parent_window()
        if window:
            dialog = PolicyEditorDialog(
                self.ATLAS,
                window,
                on_save=self._on_policy_saved,
            )
            dialog.present()

    def _on_policy_saved(self, policy_data: dict) -> None:
        """Handle policy save callback."""
        logger.info("Policy saved: %s", policy_data.get("name"))
        self._refresh_policies()

    def _get_parent_window(self) -> Optional[Gtk.Window]:
        """Get the parent window for dialogs."""
        widget = self.get_parent()
        while widget:
            if isinstance(widget, Gtk.Window):
                return widget
            widget = widget.get_parent()
        return None

    def refresh(self) -> None:
        """Public method to refresh policy list."""
        self._refresh_policies()
