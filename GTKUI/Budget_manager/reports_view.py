"""Reports view for displaying usage analytics and generated reports."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # type: ignore[import-untyped]

from .widgets import clear_container, create_section_header, format_currency, format_percentage

logger = logging.getLogger(__name__)


class ReportsView(Gtk.Box):
    """View for displaying usage analytics and generating reports."""

    def __init__(self, atlas: Any, parent_controller: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas
        self._parent = parent_controller

        self.set_margin_top(16)
        self.set_margin_bottom(16)
        self.set_margin_start(16)
        self.set_margin_end(16)

        # Report configuration (initialized in _build_ui)
        self._period_combo: Gtk.ComboBoxText
        self._grouping_combo: Gtk.ComboBoxText

        # Report display areas (initialized in _build_ui)
        self._summary_container: Gtk.Box
        self._chart_container: Gtk.Box
        self._breakdown_container: Gtk.Box
        self._trends_container: Gtk.Box

        # Current report data
        self._current_report: Optional[Dict[str, Any]] = None

        self._build_ui()
        self._generate_report()

    def _build_ui(self) -> None:
        """Build the reports view layout."""
        # Title row
        title_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        title = Gtk.Label(label="Usage Reports")
        title.set_xalign(0.0)
        title.add_css_class("title-2")
        title.set_hexpand(True)
        title_row.append(title)

        export_btn = Gtk.Button.new_from_icon_name("document-save-symbolic")
        export_btn.add_css_class("flat")
        export_btn.set_tooltip_text("Export report")
        export_btn.connect("clicked", self._on_export)
        title_row.append(export_btn)

        self.append(title_row)

        # Report configuration bar
        config_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        config_box.set_margin_top(8)
        config_box.set_margin_bottom(8)

        # Period selector
        period_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        period_label = Gtk.Label(label="Report Period")
        period_label.set_xalign(0.0)
        period_label.add_css_class("caption")
        period_box.append(period_label)

        self._period_combo = Gtk.ComboBoxText()
        self._period_combo.append("today", "Today")
        self._period_combo.append("week", "This Week")
        self._period_combo.append("month", "This Month")
        self._period_combo.append("quarter", "This Quarter")
        self._period_combo.append("year", "This Year")
        self._period_combo.append("all", "All Time")
        self._period_combo.set_active(2)  # Default to month
        self._period_combo.connect("changed", self._on_config_changed)
        period_box.append(self._period_combo)
        config_box.append(period_box)

        # Grouping selector
        grouping_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        grouping_label = Gtk.Label(label="Group By")
        grouping_label.set_xalign(0.0)
        grouping_label.add_css_class("caption")
        grouping_box.append(grouping_label)

        self._grouping_combo = Gtk.ComboBoxText()
        self._grouping_combo.append("day", "Day")
        self._grouping_combo.append("week", "Week")
        self._grouping_combo.append("provider", "Provider")
        self._grouping_combo.append("model", "Model")
        self._grouping_combo.append("operation", "Operation Type")
        self._grouping_combo.set_active(2)  # Default to provider
        self._grouping_combo.connect("changed", self._on_config_changed)
        grouping_box.append(self._grouping_combo)
        config_box.append(grouping_box)

        # Generate button
        generate_btn = Gtk.Button(label="Generate Report")
        generate_btn.add_css_class("suggested-action")
        generate_btn.set_valign(Gtk.Align.END)
        generate_btn.connect("clicked", lambda _: self._generate_report())
        config_box.append(generate_btn)

        self.append(config_box)

        # Scrolled content area
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_vexpand(True)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)

        # Summary section
        summary_header = create_section_header("Summary")
        content.append(summary_header)

        self._summary_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self._summary_container.set_homogeneous(True)
        content.append(self._summary_container)

        # Breakdown section
        breakdown_header = create_section_header("Breakdown")
        content.append(breakdown_header)

        self._breakdown_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        content.append(self._breakdown_container)

        # Trends section
        trends_header = create_section_header("Trends")
        content.append(trends_header)

        self._trends_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        content.append(self._trends_container)

        scroller.set_child(content)
        self.append(scroller)

    def _on_config_changed(self, _widget) -> None:
        """Handle report configuration change."""
        pass  # Report generated on button click

    def _generate_report(self) -> None:
        """Generate report with current configuration."""
        period = self._period_combo.get_active_id() if self._period_combo else "month"
        grouping = self._grouping_combo.get_active_id() if self._grouping_combo else "provider"

        async def fetch_report() -> None:
            try:
                from core.services.budget import get_tracking_service, UsageSummaryRequest, BudgetScope, BudgetPeriod
                from core.services.common import Actor
                from datetime import timedelta

                actor = Actor(
                    type="system",
                    id="gtkui-reports",
                    tenant_id="local",
                    permissions={"budget:read"},
                )

                tracking_service = await get_tracking_service()

                # Map period to BudgetPeriod
                period_map = {
                    "today": BudgetPeriod.DAILY,
                    "week": BudgetPeriod.WEEKLY,
                    "month": BudgetPeriod.MONTHLY,
                    "quarter": BudgetPeriod.QUARTERLY,
                    "year": BudgetPeriod.YEARLY,
                    "all": BudgetPeriod.YEARLY,  # Fallback
                }
                budget_period = period_map.get(period, BudgetPeriod.MONTHLY)

                # Get usage summary
                request = UsageSummaryRequest(
                    scope=BudgetScope.GLOBAL,
                    period=budget_period,
                )
                result = await tracking_service.get_usage_summary(actor, request)

                if result.success and result.data:
                    report_data = {
                        "total_spent": str(result.data.total_spent) if hasattr(result.data, 'total_spent') else "0.00",
                        "period": period,
                        "grouping": grouping,
                    }
                    GLib.idle_add(self._on_report_loaded, report_data)
                else:
                    GLib.idle_add(self._on_report_loaded, {})
            except Exception as exc:
                logger.warning("Failed to generate report: %s", exc)
                GLib.idle_add(self._on_report_error, str(exc))

        asyncio.create_task(fetch_report())

    def _on_report_loaded(self, report_data: Dict[str, Any]) -> bool:
        """Handle loaded report data on main thread."""
        self._current_report = report_data
        self._populate_summary()
        self._populate_breakdown(
            self._grouping_combo.get_active_id() if self._grouping_combo else "provider"
        )
        self._populate_trends()
        return False

    def _on_report_error(self, error_msg: str) -> bool:
        """Handle report generation error."""
        logger.warning("Report generation error: %s", error_msg)
        return False

    def _load_report_data(self) -> bool:
        """Load and display report data (sync fallback)."""
        try:
            # Use sample data for fallback
            self._populate_summary()
            self._populate_breakdown(
                self._grouping_combo.get_active_id() if self._grouping_combo else "provider"
            )
            self._populate_trends()

        except Exception as exc:
            logger.warning("Failed to generate report: %s", exc)

        return False

    def _populate_summary(self) -> None:
        """Populate summary statistics."""
        if not self._summary_container:
            return

        clear_container(self._summary_container)

        # Sample summary data
        summary_data = [
            ("Total Spend", format_currency(Decimal("127.45")), None),
            ("Requests", "1,847", None),
            ("Avg Cost/Request", format_currency(Decimal("0.069")), None),
            ("Most Used", "GPT-4", "42% of requests"),
        ]

        for title, value, subtitle in summary_data:
            card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            card.add_css_class("budget-stat-card")
            card.set_margin_top(8)
            card.set_margin_bottom(8)
            card.set_margin_start(12)
            card.set_margin_end(12)

            title_lbl = Gtk.Label(label=title)
            title_lbl.set_xalign(0.0)
            title_lbl.add_css_class("caption")
            card.append(title_lbl)

            value_lbl = Gtk.Label(label=value)
            value_lbl.set_xalign(0.0)
            value_lbl.add_css_class("title-3")
            card.append(value_lbl)

            if subtitle:
                sub_lbl = Gtk.Label(label=subtitle)
                sub_lbl.set_xalign(0.0)
                sub_lbl.add_css_class("dim-label")
                card.append(sub_lbl)

            self._summary_container.append(card)

    def _populate_breakdown(self, grouping: str) -> None:
        """Populate breakdown by grouping."""
        if not self._breakdown_container:
            return

        clear_container(self._breakdown_container)

        # Sample breakdown data based on grouping
        if grouping == "provider":
            breakdown = [
                ("OpenAI", Decimal("78.50"), 61.6),
                ("Anthropic", Decimal("32.25"), 25.3),
                ("Google", Decimal("12.40"), 9.7),
                ("Other", Decimal("4.30"), 3.4),
            ]
        elif grouping == "operation":
            breakdown = [
                ("LLM Completion", Decimal("95.20"), 74.7),
                ("Image Generation", Decimal("18.50"), 14.5),
                ("Embeddings", Decimal("8.75"), 6.9),
                ("Audio", Decimal("5.00"), 3.9),
            ]
        else:
            breakdown = [
                ("GPT-4", Decimal("52.30"), 41.0),
                ("Claude-3-Opus", Decimal("28.15"), 22.1),
                ("GPT-3.5-Turbo", Decimal("26.20"), 20.6),
                ("Gemini-Pro", Decimal("12.40"), 9.7),
                ("Other", Decimal("8.40"), 6.6),
            ]

        for name, amount, pct in breakdown:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
            row.set_margin_start(8)
            row.set_margin_end(8)

            name_label = Gtk.Label(label=name)
            name_label.set_xalign(0.0)
            name_label.set_width_chars(20)
            row.append(name_label)

            # Progress bar representing percentage
            progress = Gtk.ProgressBar()
            progress.set_fraction(pct / 100)
            progress.set_hexpand(True)
            progress.set_valign(Gtk.Align.CENTER)
            row.append(progress)

            pct_label = Gtk.Label(label=format_percentage(pct))
            pct_label.set_width_chars(8)
            pct_label.add_css_class("dim-label")
            row.append(pct_label)

            amount_label = Gtk.Label(label=format_currency(amount))
            amount_label.set_xalign(1.0)
            amount_label.set_width_chars(12)
            row.append(amount_label)

            self._breakdown_container.append(row)

    def _populate_trends(self) -> None:
        """Populate trends section."""
        if not self._trends_container:
            return

        clear_container(self._trends_container)

        # Sample trend insights
        trends = [
            ("↑ 23%", "Spending increased vs. last period", "warning"),
            ("↓ 15%", "Cost per request decreased", "success"),
            ("→", "Provider mix unchanged", "neutral"),
        ]

        for indicator, description, style in trends:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
            row.set_margin_start(8)
            row.set_margin_end(8)

            indicator_label = Gtk.Label(label=indicator)
            indicator_label.set_width_chars(6)
            indicator_label.add_css_class(f"trend-{style}")
            row.append(indicator_label)

            desc_label = Gtk.Label(label=description)
            desc_label.set_xalign(0.0)
            row.append(desc_label)

            self._trends_container.append(row)

    def _on_export(self, _button: Gtk.Button) -> None:
        """Handle export button click."""
        logger.info("Report export requested")

        async def do_export() -> None:
            try:
                from core.services.budget import get_tracking_service, UsageSummaryRequest, BudgetScope, BudgetPeriod
                from core.services.common import Actor

                actor = Actor(
                    type="system",
                    id="gtkui-reports",
                    tenant_id="local",
                    permissions={"budget:read"},
                )

                tracking_service = await get_tracking_service()

                request = UsageSummaryRequest(
                    scope=BudgetScope.GLOBAL,
                    period=BudgetPeriod.MONTHLY,
                )
                result = await tracking_service.get_usage_summary(actor, request)

                if result.success and result.data:
                    # Generate simple HTML report
                    html_content = f"""
                    <html>
                    <head><title>Budget Report</title></head>
                    <body>
                    <h1>Usage Report</h1>
                    <p>Total Spent: ${result.data.total_spent if hasattr(result.data, 'total_spent') else '0.00'}</p>
                    <p>Generated: {datetime.now(timezone.utc).isoformat()}</p>
                    </body>
                    </html>
                    """
                    GLib.idle_add(self._copy_to_clipboard, html_content)
                    logger.info("Report exported as HTML")
            except Exception as exc:
                logger.warning("Failed to export report: %s", exc)

        asyncio.create_task(do_export())

    def _copy_to_clipboard(self, content: str) -> bool:
        """Copy content to clipboard."""
        clipboard = self.get_clipboard()
        if clipboard:
            clipboard.set(content)
            logger.info("Export data copied to clipboard")
        return False

    def refresh(self) -> None:
        """Public method to refresh the report."""
        self._generate_report()
