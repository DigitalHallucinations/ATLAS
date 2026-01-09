"""Alerts panel for viewing and managing budget alerts."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # type: ignore[import-untyped]

from .widgets import clear_container, create_badge, format_currency

logger = logging.getLogger(__name__)


class AlertsPanel(Gtk.Box):
    """Panel for viewing and acknowledging budget alerts."""

    def __init__(self, atlas: Any, parent_controller: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas
        self._parent = parent_controller

        self.set_margin_top(16)
        self.set_margin_bottom(16)
        self.set_margin_start(16)
        self.set_margin_end(16)

        self._alerts: List[Dict[str, Any]] = []
        self._filter_status: str = "all"  # all, active, acknowledged

        # UI elements (initialized in _build_ui)
        self._list_container: Gtk.ListBox
        self._status_combo: Gtk.ComboBoxText
        self._count_label: Gtk.Label

        self._build_ui()
        self._refresh_alerts()

    def _build_ui(self) -> None:
        """Build the alerts panel layout."""
        # Title row
        title_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        title = Gtk.Label(label="Budget Alerts")
        title.set_xalign(0.0)
        title.add_css_class("title-2")
        title.set_hexpand(True)
        title_row.append(title)

        # Active count badge
        self._count_label = Gtk.Label()
        self._count_label.add_css_class("badge")
        self._count_label.add_css_class("badge-warning")
        title_row.append(self._count_label)

        refresh_btn = Gtk.Button.new_from_icon_name("view-refresh-symbolic")
        refresh_btn.add_css_class("flat")
        refresh_btn.set_tooltip_text("Refresh alerts")
        refresh_btn.connect("clicked", lambda _: self._refresh_alerts())
        title_row.append(refresh_btn)

        self.append(title_row)

        # Filter row
        filter_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        filter_row.set_margin_top(8)

        filter_label = Gtk.Label(label="Show:")
        filter_label.add_css_class("dim-label")
        filter_row.append(filter_label)

        self._status_combo = Gtk.ComboBoxText()
        self._status_combo.append("all", "All Alerts")
        self._status_combo.append("active", "Active Only")
        self._status_combo.append("acknowledged", "Acknowledged")
        self._status_combo.set_active(0)
        self._status_combo.connect("changed", self._on_filter_changed)
        filter_row.append(self._status_combo)

        # Acknowledge all button
        ack_all_btn = Gtk.Button(label="Acknowledge All")
        ack_all_btn.add_css_class("flat")
        ack_all_btn.set_halign(Gtk.Align.END)
        ack_all_btn.set_hexpand(True)
        ack_all_btn.connect("clicked", self._on_acknowledge_all)
        filter_row.append(ack_all_btn)

        self.append(filter_row)

        # Alert list
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_vexpand(True)

        self._list_container = Gtk.ListBox()
        self._list_container.set_selection_mode(Gtk.SelectionMode.NONE)
        self._list_container.add_css_class("boxed-list")
        scroller.set_child(self._list_container)

        self.append(scroller)

    def _refresh_alerts(self) -> None:
        """Refresh alerts from the alert service."""
        async def fetch_alerts() -> None:
            try:
                from core.services.budget import get_alert_service, AlertListRequest

                alert_service = await get_alert_service()
                request = AlertListRequest(active_only=False)
                alerts = await alert_service.get_active_alerts(request)
                
                alert_dicts = [
                    {
                        "id": a.id,
                        "severity": a.severity.value if hasattr(a.severity, 'value') else str(a.severity),
                        "trigger_type": a.trigger_type.value if hasattr(a.trigger_type, 'value') else str(a.trigger_type),
                        "policy_name": a.policy_id,  # Could lookup policy name
                        "message": a.message,
                        "triggered_at": a.triggered_at.isoformat() if hasattr(a, 'triggered_at') else "",
                        "acknowledged": getattr(a, 'acknowledged', False),
                        "acknowledged_at": a.acknowledged_at.isoformat() if getattr(a, 'acknowledged_at', None) else None,
                    }
                    for a in alerts
                ]
                GLib.idle_add(self._on_alerts_loaded, alert_dicts)
            except Exception as exc:
                logger.warning("Failed to fetch alerts: %s", exc)
                # Fallback to sample data
                GLib.idle_add(self._on_alerts_loaded, self._generate_sample_alerts())

        asyncio.create_task(fetch_alerts())

    def _on_alerts_loaded(self, alerts: List[Dict[str, Any]]) -> bool:
        """Handle loaded alerts on main thread."""
        self._alerts = alerts
        self._apply_filter()
        self._update_count()
        return False

    def _on_alerts_error(self, error_msg: str) -> bool:
        """Handle alert loading error."""
        logger.warning("Alert loading error: %s", error_msg)
        self._show_empty_state("Unable to load alerts")
        return False

    def _load_alerts(self) -> bool:
        """Load alerts data (sync fallback)."""
        try:
            # Fallback to sample data if async not available
            self._alerts = self._generate_sample_alerts()
            self._apply_filter()
            self._update_count()

        except Exception as exc:
            logger.warning("Failed to load alerts: %s", exc)
            self._show_empty_state("Unable to load alerts")

        return False

    def _generate_sample_alerts(self) -> List[Dict[str, Any]]:
        """Generate sample alerts for demonstration."""
        now = datetime.now(timezone.utc)
        return [
            {
                "id": "alert_1",
                "severity": "warning",
                "trigger_type": "threshold_warning",
                "policy_name": "Global Monthly",
                "message": "Budget usage at 82% ($82.15 of $100.00)",
                "triggered_at": now.isoformat(),
                "acknowledged": False,
                "acknowledged_at": None,
            },
            {
                "id": "alert_2",
                "severity": "critical",
                "trigger_type": "threshold_critical",
                "policy_name": "OpenAI Provider",
                "message": "Budget usage at 96% ($48.20 of $50.00)",
                "triggered_at": now.isoformat(),
                "acknowledged": False,
                "acknowledged_at": None,
            },
            {
                "id": "alert_3",
                "severity": "info",
                "trigger_type": "period_reset",
                "policy_name": "Global Monthly",
                "message": "Budget period reset. Previous spend: $94.50",
                "triggered_at": now.isoformat(),
                "acknowledged": True,
                "acknowledged_at": now.isoformat(),
            },
        ]

    def _on_filter_changed(self, _combo: Gtk.ComboBoxText) -> None:
        """Handle filter change."""
        self._filter_status = self._status_combo.get_active_id() if self._status_combo else "all"
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Apply current filter to alert list."""
        if not self._list_container:
            return

        clear_container(self._list_container)

        filtered = []
        for alert in self._alerts:
            if self._filter_status == "active" and alert.get("acknowledged"):
                continue
            if self._filter_status == "acknowledged" and not alert.get("acknowledged"):
                continue
            filtered.append(alert)

        if not filtered:
            self._show_empty_state("No alerts match the current filter")
            return

        for alert in filtered:
            row = self._create_alert_row(alert)
            self._list_container.append(row)

    def _create_alert_row(self, alert: Dict[str, Any]) -> Gtk.ListBoxRow:
        """Create a list row for an alert."""
        row = Gtk.ListBoxRow()
        row.set_activatable(False)

        outer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        outer_box.set_margin_top(12)
        outer_box.set_margin_bottom(12)
        outer_box.set_margin_start(12)
        outer_box.set_margin_end(12)

        # Header row with severity badge and timestamp
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        severity = alert.get("severity", "info")
        severity_badge = self._create_severity_badge(severity)
        header.append(severity_badge)

        policy_label = Gtk.Label(label=alert.get("policy_name", "Unknown Policy"))
        policy_label.add_css_class("heading")
        policy_label.set_hexpand(True)
        policy_label.set_xalign(0.0)
        header.append(policy_label)

        # Timestamp
        ts_str = alert.get("triggered_at", "")
        try:
            if ts_str.endswith("Z"):
                ts_str = ts_str.replace("Z", "+00:00")
            ts = datetime.fromisoformat(ts_str)
            ts_display = ts.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            ts_display = "Unknown"

        ts_label = Gtk.Label(label=ts_display)
        ts_label.add_css_class("dim-label")
        header.append(ts_label)

        outer_box.append(header)

        # Message
        message_label = Gtk.Label(label=alert.get("message", ""))
        message_label.set_xalign(0.0)
        message_label.set_wrap(True)
        message_label.set_wrap_mode(True)
        outer_box.append(message_label)

        # Actions row
        if not alert.get("acknowledged"):
            actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            actions.set_margin_top(8)

            alert_id = str(alert.get("id", ""))
            ack_btn = Gtk.Button(label="Acknowledge")
            ack_btn.add_css_class("flat")
            ack_btn.connect(
                "clicked",
                lambda _, aid=alert_id: self._on_acknowledge(aid),
            )
            actions.append(ack_btn)

            outer_box.append(actions)
        else:
            ack_info = Gtk.Label(label="✓ Acknowledged")
            ack_info.set_xalign(0.0)
            ack_info.add_css_class("dim-label")
            ack_info.set_margin_top(4)
            outer_box.append(ack_info)

        row.set_child(outer_box)

        # Apply severity styling to row
        if severity == "critical":
            row.add_css_class("alert-critical")
        elif severity == "warning":
            row.add_css_class("alert-warning")

        return row

    def _create_severity_badge(self, severity: str) -> Gtk.Label:
        """Create a severity badge label."""
        severity_display = severity.upper()
        badge = Gtk.Label(label=severity_display)
        badge.add_css_class("badge")

        if severity == "critical":
            badge.add_css_class("badge-error")
        elif severity == "warning":
            badge.add_css_class("badge-warning")
        else:
            badge.add_css_class("badge-info")

        return badge

    def _update_count(self) -> None:
        """Update the active alert count badge."""
        if not self._count_label:
            return

        active_count = sum(1 for a in self._alerts if not a.get("acknowledged"))

        if active_count > 0:
            self._count_label.set_text(str(active_count))
            self._count_label.set_visible(True)
        else:
            self._count_label.set_visible(False)

    def _show_empty_state(self, message: str) -> None:
        """Show empty state message."""
        if not self._list_container:
            return

        clear_container(self._list_container)

        row = Gtk.ListBoxRow()
        row.set_activatable(False)

        label = Gtk.Label(label=message)
        label.add_css_class("dim-label")
        label.set_margin_top(24)
        label.set_margin_bottom(24)
        row.set_child(label)

        self._list_container.append(row)

    def _on_acknowledge(self, alert_id: str) -> None:
        """Handle acknowledge button click."""
        logger.info("Acknowledging alert: %s", alert_id)

        async def do_acknowledge() -> None:
            try:
                from core.services.budget import get_alert_service
                from core.services.common import Actor

                actor = Actor(
                    type="user",
                    id="gtkui-user",
                    tenant_id="local",
                    permissions={"budget:write"},
                )

                alert_service = await get_alert_service()
                result = await alert_service.acknowledge_alert(actor, alert_id)
                if result:
                    logger.info("Alert acknowledged: %s", alert_id)

                # Update local state
                GLib.idle_add(self._mark_acknowledged, alert_id)
            except Exception as exc:
                logger.warning("Failed to acknowledge alert: %s", exc)

        asyncio.create_task(do_acknowledge())

    def _mark_acknowledged(self, alert_id: str) -> bool:
        """Mark alert as acknowledged in local state."""
        for alert in self._alerts:
            if alert.get("id") == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
                break
        self._apply_filter()
        self._update_count()
        return False

    def _on_acknowledge_all(self, _button: Gtk.Button) -> None:
        """Acknowledge all active alerts."""
        async def do_acknowledge_all() -> None:
            try:
                from core.services.budget import get_alert_service
                from core.services.common import Actor

                actor = Actor(
                    type="user",
                    id="gtkui-user",
                    tenant_id="local",
                    permissions={"budget:write"},
                )

                alert_service = await get_alert_service()
                for alert in self._alerts:
                    if not alert.get("acknowledged"):
                        await alert_service.acknowledge_alert(actor, alert["id"])

                GLib.idle_add(self._mark_all_acknowledged)
            except Exception as exc:
                logger.warning("Failed to acknowledge all alerts: %s", exc)

        asyncio.create_task(do_acknowledge_all())

    def _mark_all_acknowledged(self) -> bool:
        """Mark all alerts as acknowledged in local state."""
        now = datetime.now(timezone.utc).isoformat()
        for alert in self._alerts:
            if not alert.get("acknowledged"):
                alert["acknowledged"] = True
                alert["acknowledged_at"] = now

        logger.info("Acknowledged all alerts")
        self._apply_filter()
        self._update_count()
        return False

    def refresh(self) -> None:
        """Public method to refresh alerts."""
        self._refresh_alerts()
