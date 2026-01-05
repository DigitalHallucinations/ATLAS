"""GTK budget management workspace controller."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from .alerts_panel import AlertsPanel
from .dashboard import BudgetDashboard
from .policy_editor import PolicyListPanel
from .reports_view import ReportsView
from .usage_history import UsageHistoryView

logger = logging.getLogger(__name__)


class BudgetManagement:
    """Controller responsible for rendering the budget management workspace."""

    def __init__(self, atlas: Any, parent_window: Any) -> None:
        self.ATLAS = atlas
        self.parent_window = parent_window

        self._widget: Optional[Gtk.Widget] = None
        self._view_stack: Optional[Gtk.Stack] = None
        self._view_switcher: Optional[Gtk.StackSwitcher] = None

        # View instances
        self._dashboard: Optional[BudgetDashboard] = None
        self._policies: Optional[PolicyListPanel] = None
        self._history: Optional[UsageHistoryView] = None
        self._reports: Optional[ReportsView] = None
        self._alerts: Optional[AlertsPanel] = None

        # MessageBus subscriptions
        self._bus_subscriptions: List[Any] = []

    def get_embeddable_widget(self) -> Gtk.Widget:
        """Return the workspace widget, creating it on first use."""
        if self._widget is None:
            self._widget = self._build_workspace()
            self._subscribe_to_bus()
        return self._widget

    def _build_workspace(self) -> Gtk.Widget:
        """Build the budget management workspace."""
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        root.set_hexpand(True)
        root.set_vexpand(True)

        # Header with view switcher
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        header.set_margin_top(12)
        header.set_margin_bottom(8)
        header.set_margin_start(16)
        header.set_margin_end(16)

        title = Gtk.Label(label="Budget Manager")
        title.set_xalign(0.0)
        title.add_css_class("title-1")
        header.append(title)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        header.append(spacer)

        # View switcher
        self._view_stack = Gtk.Stack()
        self._view_stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self._view_stack.set_transition_duration(200)

        self._view_switcher = Gtk.StackSwitcher()
        self._view_switcher.set_stack(self._view_stack)
        header.append(self._view_switcher)

        root.append(header)

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        root.append(separator)

        # Build views
        self._dashboard = BudgetDashboard(self.ATLAS, self)
        self._view_stack.add_titled(self._dashboard, "dashboard", "Dashboard")

        self._policies = PolicyListPanel(self.ATLAS, self)
        self._view_stack.add_titled(self._policies, "policies", "Policies")

        self._history = UsageHistoryView(self.ATLAS, self)
        self._view_stack.add_titled(self._history, "history", "History")

        self._reports = ReportsView(self.ATLAS, self)
        self._view_stack.add_titled(self._reports, "reports", "Reports")

        self._alerts = AlertsPanel(self.ATLAS, self)
        self._view_stack.add_titled(self._alerts, "alerts", "Alerts")

        # Set icons for stack pages
        self._view_stack.get_page(self._dashboard).set_icon_name("view-dashboard-symbolic")
        self._view_stack.get_page(self._policies).set_icon_name("emblem-documents-symbolic")
        self._view_stack.get_page(self._history).set_icon_name("document-open-recent-symbolic")
        self._view_stack.get_page(self._reports).set_icon_name("x-office-spreadsheet-symbolic")
        self._view_stack.get_page(self._alerts).set_icon_name("dialog-warning-symbolic")

        self._view_stack.set_hexpand(True)
        self._view_stack.set_vexpand(True)
        root.append(self._view_stack)

        return root

    def _subscribe_to_bus(self) -> None:
        """Subscribe to budget-related MessageBus channels."""
        try:
            from core.messaging import MessageBus

            bus = MessageBus.get_instance()

            # Subscribe to budget events
            channels = [
                "BUDGET_USAGE",
                "BUDGET_ALERT",
                "BUDGET_POLICY",
            ]

            for channel in channels:
                try:
                    subscription = bus.subscribe(channel, self._on_bus_message)
                    self._bus_subscriptions.append(subscription)
                except Exception as exc:
                    logger.debug("Failed to subscribe to %s: %s", channel, exc)

        except ImportError:
            logger.debug("MessageBus not available for budget subscriptions")
        except Exception as exc:
            logger.debug("Failed to subscribe to budget channels: %s", exc)

    def _on_bus_message(self, message: Any) -> None:
        """Handle incoming MessageBus events."""
        GLib.idle_add(self._process_bus_message, message)

    def _process_bus_message(self, message: Any) -> bool:
        """Process a MessageBus message on the main thread."""
        try:
            channel = getattr(message, "channel", None)
            payload = getattr(message, "payload", {})

            if channel == "BUDGET_USAGE":
                # Refresh dashboard and history on usage events
                if self._dashboard:
                    self._dashboard.refresh()
                if self._history:
                    self._history.refresh()

            elif channel == "BUDGET_ALERT":
                # Refresh alerts panel
                if self._alerts:
                    self._alerts.refresh()
                # Also refresh dashboard for alert count
                if self._dashboard:
                    self._dashboard.refresh()

            elif channel == "BUDGET_POLICY":
                # Refresh policies and dashboard
                if self._policies:
                    self._policies.refresh()
                if self._dashboard:
                    self._dashboard.refresh()

        except Exception as exc:
            logger.debug("Error processing budget bus message: %s", exc)

        return False

    def _on_close_request(self) -> None:
        """Clean up resources when the workspace tab is closed."""
        for subscription in list(self._bus_subscriptions):
            try:
                subscription.cancel()
            except Exception:
                logger.debug("Failed to cancel budget bus subscription", exc_info=True)
            finally:
                self._bus_subscriptions.remove(subscription)

    def show_dashboard(self) -> None:
        """Switch to the dashboard view."""
        if self._view_stack:
            self._view_stack.set_visible_child_name("dashboard")

    def show_policies(self) -> None:
        """Switch to the policies view."""
        if self._view_stack:
            self._view_stack.set_visible_child_name("policies")

    def show_history(self) -> None:
        """Switch to the history view."""
        if self._view_stack:
            self._view_stack.set_visible_child_name("history")

    def show_reports(self) -> None:
        """Switch to the reports view."""
        if self._view_stack:
            self._view_stack.set_visible_child_name("reports")

    def show_alerts(self) -> None:
        """Switch to the alerts view."""
        if self._view_stack:
            self._view_stack.set_visible_child_name("alerts")

    def refresh_all(self) -> None:
        """Refresh all budget views."""
        if self._dashboard:
            self._dashboard.refresh()
        if self._policies:
            self._policies.refresh()
        if self._history:
            self._history.refresh()
        if self._reports:
            self._reports.refresh()
        if self._alerts:
            self._alerts.refresh()
