"""Budget dashboard view displaying spending overview and key metrics."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # type: ignore[import-untyped]

from .widgets import (
    clear_container,
    create_progress_bar,
    create_section_header,
    create_stat_card,
    format_currency,
    format_percentage,
)

logger = logging.getLogger(__name__)


class BudgetDashboard(Gtk.Box):
    """Main dashboard showing budget status, spending summary, and quick actions."""

    def __init__(self, atlas: Any, parent_controller: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas
        self._parent = parent_controller

        self.set_margin_top(16)
        self.set_margin_bottom(16)
        self.set_margin_start(16)
        self.set_margin_end(16)

        # Widget containers (initialized in _build_ui)
        self._overview_container: Gtk.Box
        self._breakdown_container: Gtk.Box
        self._alerts_preview: Gtk.Box
        self._recent_activity: Gtk.Box

        self._build_ui()
        self._refresh_data()

    def _build_ui(self) -> None:
        """Construct the dashboard layout."""
        # Title row
        title_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        title = Gtk.Label(label="Budget Dashboard")
        title.set_xalign(0.0)
        title.add_css_class("title-2")
        title.set_hexpand(True)
        title_row.append(title)

        refresh_btn = Gtk.Button.new_from_icon_name("view-refresh-symbolic")
        refresh_btn.add_css_class("flat")
        refresh_btn.set_tooltip_text("Refresh dashboard")
        refresh_btn.connect("clicked", lambda _: self._refresh_data())
        title_row.append(refresh_btn)

        self.append(title_row)

        # Scrolled content area
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_vexpand(True)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)

        # Overview section - key metrics cards
        self._overview_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self._overview_container.set_homogeneous(True)
        content.append(self._overview_container)

        # Budget progress section
        progress_header = create_section_header("Current Period Progress")
        content.append(progress_header)

        self._progress_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        content.append(self._progress_container)

        # Breakdown section
        breakdown_header = create_section_header("Spending Breakdown")
        content.append(breakdown_header)

        self._breakdown_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        content.append(self._breakdown_container)

        # Active alerts preview
        alerts_header = create_section_header("Recent Alerts")
        content.append(alerts_header)

        self._alerts_preview = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        content.append(self._alerts_preview)

        # Recent activity
        activity_header = create_section_header("Recent Activity")
        content.append(activity_header)

        self._recent_activity = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        content.append(self._recent_activity)

        scroller.set_child(content)
        self.append(scroller)

    def _refresh_data(self) -> None:
        """Fetch and display current budget data."""
        async def fetch_dashboard_data() -> None:
            try:
                from core.services.budget import (
                    get_policy_service,
                    get_tracking_service,
                    get_alert_service,
                    UsageSummaryRequest,
                    AlertListRequest,
                    BudgetScope,
                    BudgetPeriod,
                )
                from core.services.common import Actor

                # Create a system actor for fetching data
                actor = Actor(
                    type="system",
                    id="gtkui-dashboard",
                    tenant_id="local",
                    permissions={"budget:read"},
                )

                # Get services
                policy_service = await get_policy_service()
                tracking_service = await get_tracking_service()
                alert_service = await get_alert_service()

                # Get current spending summary
                summary_request = UsageSummaryRequest(
                    scope=BudgetScope.GLOBAL,
                    period=BudgetPeriod.MONTHLY,
                )
                summary_result = await tracking_service.get_usage_summary(actor, summary_request)
                summary = summary_result.data if summary_result.success else None

                # Get all policies
                policies_result = await policy_service.list_policies(actor, enabled_only=True)
                policies = policies_result.data if policies_result.success else []

                # Get active alerts
                alert_request = AlertListRequest(active_only=True)
                alerts = await alert_service.get_active_alerts(alert_request)

                # Get recent usage from tracking service buffer
                recent_usage = tracking_service.get_recent_records(limit=5) if hasattr(tracking_service, 'get_recent_records') else []

                data = {
                    "summary": summary,
                    "policies": policies,
                    "alerts": alerts,
                    "recent_usage": recent_usage,
                }
                GLib.idle_add(self._on_data_loaded, data)
            except Exception as exc:
                logger.warning("Failed to fetch dashboard data: %s", exc)
                GLib.idle_add(self._on_data_error, str(exc))

        asyncio.create_task(fetch_dashboard_data())

    def _on_data_loaded(self, data: Optional[Dict[str, Any]]) -> bool:
        """Handle loaded dashboard data on main thread."""
        self._dashboard_data = data
        try:
            self._populate_overview()
            self._populate_progress()
            self._populate_breakdown()
            self._populate_alerts()
            self._populate_activity()
        except Exception as exc:
            logger.warning("Failed to populate dashboard: %s", exc)
            self._show_error_state()
        return False

    def _on_data_error(self, error_msg: str) -> bool:
        """Handle data loading error."""
        logger.warning("Dashboard data error: %s", error_msg)
        self._show_error_state()
        return False

    def _load_dashboard_data(self) -> bool:
        """Load dashboard data (sync fallback)."""
        try:
            self._populate_overview()
            self._populate_progress()
            self._populate_breakdown()
            self._populate_alerts()
            self._populate_activity()
        except Exception as exc:
            logger.warning("Failed to load dashboard data: %s", exc)
            self._show_error_state()
        return False

    def _populate_overview(self) -> None:
        """Populate the overview metrics cards."""
        if not self._overview_container:
            return

        clear_container(self._overview_container)

        # Get data from loaded dashboard data or use defaults
        if hasattr(self, '_dashboard_data') and self._dashboard_data:
            summary = self._dashboard_data.get('summary')
            policies = self._dashboard_data.get('policies', [])
            if summary:
                total_spent = summary.total_spent
                budget_limit = summary.effective_limit
                remaining = summary.remaining
                active_policies = len(policies)
            else:
                total_spent = Decimal("0.00")
                budget_limit = Decimal("100.00")
                remaining = budget_limit
                active_policies = 0
        else:
            total_spent = Decimal("0.00")
            budget_limit = Decimal("100.00")
            remaining = budget_limit
            active_policies = 0

        spent_card = create_stat_card(
            "Total Spent",
            format_currency(total_spent),
            "This period",
        )
        self._overview_container.append(spent_card)

        remaining_card = create_stat_card(
            "Remaining",
            format_currency(remaining),
            f"of {format_currency(budget_limit)}",
        )
        self._overview_container.append(remaining_card)

        policies_card = create_stat_card(
            "Active Policies",
            str(active_policies),
            "Managing budgets",
        )
        self._overview_container.append(policies_card)

        usage_pct = float(total_spent / budget_limit * 100) if budget_limit > 0 else 0
        usage_card = create_stat_card(
            "Usage",
            format_percentage(usage_pct),
            "of budget consumed",
        )
        self._overview_container.append(usage_card)

    def _populate_progress(self) -> None:
        """Populate budget progress indicators."""
        if not self._progress_container:
            return

        clear_container(self._progress_container)

        # Get policy data from loaded dashboard data
        policies_data = []
        if hasattr(self, '_dashboard_data') and self._dashboard_data:
            policies = self._dashboard_data.get('policies', [])
            summary = self._dashboard_data.get('summary')
            if policies and summary:
                # Show the summary as the main policy progress
                policies_data.append(
                    ("Current Period", summary.total_spent, summary.effective_limit)
                )
        
        # Fallback if no data
        if not policies_data:
            policies_data = [
                ("No active policies", Decimal("0.00"), Decimal("100.00")),
            ]

        for name, spent, limit in policies_data:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
            row.set_margin_start(8)
            row.set_margin_end(8)

            label = Gtk.Label(label=name)
            label.set_xalign(0.0)
            label.set_width_chars(20)
            row.append(label)

            if limit > 0:
                progress = create_progress_bar(
                    float(spent),
                    float(limit),
                    show_text=True,
                )
                progress.set_hexpand(True)
                row.append(progress)
            else:
                no_limit = Gtk.Label(label="No limit set")
                no_limit.add_css_class("dim-label")
                row.append(no_limit)

            amount_label = Gtk.Label(
                label=f"{format_currency(spent)} / {format_currency(limit)}"
            )
            amount_label.set_xalign(1.0)
            amount_label.set_width_chars(18)
            row.append(amount_label)

            self._progress_container.append(row)

    def _populate_breakdown(self) -> None:
        """Populate spending breakdown by category."""
        if not self._breakdown_container:
            return

        clear_container(self._breakdown_container)

        # Example breakdown data
        breakdown = [
            ("LLM Completions", Decimal("28.50"), "60.2%"),
            ("Image Generation", Decimal("12.30"), "26.0%"),
            ("Embeddings", Decimal("4.22"), "8.9%"),
            ("Audio Processing", Decimal("2.30"), "4.9%"),
        ]

        for category, amount, pct in breakdown:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            row.set_margin_start(8)
            row.set_margin_end(8)

            cat_label = Gtk.Label(label=category)
            cat_label.set_xalign(0.0)
            cat_label.set_hexpand(True)
            row.append(cat_label)

            pct_label = Gtk.Label(label=pct)
            pct_label.add_css_class("dim-label")
            pct_label.set_width_chars(8)
            row.append(pct_label)

            amount_label = Gtk.Label(label=format_currency(amount))
            amount_label.set_xalign(1.0)
            amount_label.set_width_chars(12)
            row.append(amount_label)

            self._breakdown_container.append(row)

    def _populate_alerts(self) -> None:
        """Populate recent alerts preview."""
        if not self._alerts_preview:
            return

        clear_container(self._alerts_preview)

        # Get alerts from loaded dashboard data
        alerts = []
        if hasattr(self, '_dashboard_data') and self._dashboard_data:
            alerts = self._dashboard_data.get('alerts', [])

        if not alerts:
            placeholder = Gtk.Label(label="No active alerts")
            placeholder.add_css_class("dim-label")
            placeholder.set_margin_start(8)
            self._alerts_preview.append(placeholder)
            return

        for alert in alerts[:3]:  # Show max 3 alerts
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            row.set_margin_start(8)
            row.set_margin_end(8)

            severity_label = Gtk.Label(label=f"[{alert.severity.value.upper()}]")
            severity_label.add_css_class(f"alert-{alert.severity.value}")
            row.append(severity_label)

            msg_label = Gtk.Label(label=alert.message[:50])
            msg_label.set_xalign(0.0)
            msg_label.set_hexpand(True)
            msg_label.set_ellipsize(3)  # PANGO_ELLIPSIZE_END
            row.append(msg_label)

            self._alerts_preview.append(row)

    def _populate_activity(self) -> None:
        """Populate recent activity list."""
        if not self._recent_activity:
            return

        clear_container(self._recent_activity)

        # Get recent usage from loaded dashboard data
        recent_usage = []
        if hasattr(self, '_dashboard_data') and self._dashboard_data:
            recent_usage = self._dashboard_data.get('recent_usage', [])

        if not recent_usage:
            placeholder = Gtk.Label(label="No recent activity")
            placeholder.add_css_class("dim-label")
            placeholder.set_margin_start(8)
            self._recent_activity.append(placeholder)
            return

        for record in recent_usage[-5:]:  # Show max 5 records
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            row.set_margin_start(8)
            row.set_margin_end(8)

            provider_label = Gtk.Label(label=record.provider)
            provider_label.set_width_chars(10)
            provider_label.set_xalign(0.0)
            row.append(provider_label)

            model_label = Gtk.Label(label=record.model[:15])
            model_label.set_width_chars(15)
            model_label.set_xalign(0.0)
            model_label.add_css_class("dim-label")
            row.append(model_label)

            cost_label = Gtk.Label(label=format_currency(record.cost_usd))
            cost_label.set_xalign(1.0)
            cost_label.add_css_class("numeric")
            row.append(cost_label)

            self._recent_activity.append(row)

    def _show_error_state(self) -> None:
        """Show error state when data loading fails."""
        if self._overview_container:
            clear_container(self._overview_container)
            error = Gtk.Label(label="Unable to load budget data")
            error.add_css_class("error-label")
            self._overview_container.append(error)

    def refresh(self) -> None:
        """Public method to refresh dashboard data."""
        self._refresh_data()
