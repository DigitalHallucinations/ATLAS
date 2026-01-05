"""Usage history view for browsing historical spending records."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # type: ignore[import-untyped]

from .widgets import clear_container, format_currency

logger = logging.getLogger(__name__)


class UsageHistoryView(Gtk.Box):
    """View for browsing and filtering usage history records."""

    def __init__(self, atlas: Any, parent_controller: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas
        self._parent = parent_controller

        self.set_margin_top(16)
        self.set_margin_bottom(16)
        self.set_margin_start(16)
        self.set_margin_end(16)

        self._records: List[Dict[str, Any]] = []
        self._filtered_records: List[Dict[str, Any]] = []

        # Filter state
        self._date_from: Optional[datetime] = None
        self._date_to: Optional[datetime] = None
        self._provider_filter: Optional[str] = None
        self._type_filter: Optional[str] = None

        # UI elements
        self._list_container: Optional[Gtk.ListBox] = None
        self._provider_combo: Optional[Gtk.ComboBoxText] = None
        self._type_combo: Optional[Gtk.ComboBoxText] = None
        self._date_from_entry: Optional[Gtk.Entry] = None
        self._date_to_entry: Optional[Gtk.Entry] = None
        self._summary_label: Optional[Gtk.Label] = None

        self._build_ui()
        self._refresh_data()

    def _build_ui(self) -> None:
        """Build the usage history view layout."""
        # Title row
        title_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        title = Gtk.Label(label="Usage History")
        title.set_xalign(0.0)
        title.add_css_class("title-2")
        title.set_hexpand(True)
        title_row.append(title)

        refresh_btn = Gtk.Button.new_from_icon_name("view-refresh-symbolic")
        refresh_btn.add_css_class("flat")
        refresh_btn.set_tooltip_text("Refresh history")
        refresh_btn.connect("clicked", lambda _: self._refresh_data())
        title_row.append(refresh_btn)

        export_btn = Gtk.Button.new_from_icon_name("document-save-symbolic")
        export_btn.add_css_class("flat")
        export_btn.set_tooltip_text("Export history")
        export_btn.connect("clicked", self._on_export)
        title_row.append(export_btn)

        self.append(title_row)

        # Filter bar
        filter_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        filter_box.set_margin_top(8)
        filter_box.set_margin_bottom(8)

        # Provider filter
        provider_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        provider_label = Gtk.Label(label="Provider")
        provider_label.set_xalign(0.0)
        provider_label.add_css_class("caption")
        provider_box.append(provider_label)

        self._provider_combo = Gtk.ComboBoxText()
        self._provider_combo.append("all", "All Providers")
        self._provider_combo.set_active(0)
        self._provider_combo.connect("changed", self._on_filter_changed)
        provider_box.append(self._provider_combo)
        filter_box.append(provider_box)

        # Type filter
        type_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        type_label = Gtk.Label(label="Operation Type")
        type_label.set_xalign(0.0)
        type_label.add_css_class("caption")
        type_box.append(type_label)

        self._type_combo = Gtk.ComboBoxText()
        self._type_combo.append("all", "All Types")
        self._type_combo.append("llm_completion", "LLM Completion")
        self._type_combo.append("image_generation", "Image Generation")
        self._type_combo.append("embedding", "Embedding")
        self._type_combo.append("audio", "Audio")
        self._type_combo.set_active(0)
        self._type_combo.connect("changed", self._on_filter_changed)
        type_box.append(self._type_combo)
        filter_box.append(type_box)

        # Date range
        date_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        date_label = Gtk.Label(label="Date Range")
        date_label.set_xalign(0.0)
        date_label.add_css_class("caption")
        date_box.append(date_label)

        date_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        self._date_from_entry = Gtk.Entry()
        self._date_from_entry.set_placeholder_text("From (YYYY-MM-DD)")
        self._date_from_entry.set_width_chars(12)
        self._date_from_entry.connect("activate", self._on_filter_changed)
        date_row.append(self._date_from_entry)

        to_label = Gtk.Label(label="to")
        to_label.add_css_class("dim-label")
        date_row.append(to_label)

        self._date_to_entry = Gtk.Entry()
        self._date_to_entry.set_placeholder_text("To (YYYY-MM-DD)")
        self._date_to_entry.set_width_chars(12)
        self._date_to_entry.connect("activate", self._on_filter_changed)
        date_row.append(self._date_to_entry)

        date_box.append(date_row)
        filter_box.append(date_box)

        # Apply/Clear buttons
        button_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        spacer = Gtk.Label(label=" ")
        spacer.add_css_class("caption")
        button_box.append(spacer)

        btn_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        apply_btn = Gtk.Button(label="Apply")
        apply_btn.connect("clicked", self._on_apply_filters)
        btn_row.append(apply_btn)

        clear_btn = Gtk.Button(label="Clear")
        clear_btn.add_css_class("flat")
        clear_btn.connect("clicked", self._on_clear_filters)
        btn_row.append(clear_btn)

        button_box.append(btn_row)
        filter_box.append(button_box)

        self.append(filter_box)

        # Summary row
        self._summary_label = Gtk.Label()
        self._summary_label.set_xalign(0.0)
        self._summary_label.add_css_class("dim-label")
        self.append(self._summary_label)

        # Scrolled list
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_vexpand(True)

        self._list_container = Gtk.ListBox()
        self._list_container.set_selection_mode(Gtk.SelectionMode.NONE)
        self._list_container.add_css_class("boxed-list")
        scroller.set_child(self._list_container)

        self.append(scroller)

    def _refresh_data(self) -> None:
        """Refresh usage history data."""
        async def fetch_usage_data() -> None:
            try:
                from modules.budget import get_budget_manager_sync
                from modules.budget.models import BudgetScope, BudgetPeriod

                manager = get_budget_manager_sync()
                if manager:
                    # Get current spending summary which includes usage
                    summary = await manager.get_current_spend(
                        scope=BudgetScope.GLOBAL,
                        period=BudgetPeriod.MONTHLY,
                    )

                    # Get usage records from manager's buffer
                    async with manager._usage_lock:
                        records = [
                            {
                                "id": r.id,
                                "timestamp": r.timestamp.isoformat(),
                                "provider": r.provider,
                                "model": r.model,
                                "operation_type": r.operation_type.value,
                                "cost": r.cost_usd,
                                "tokens_in": r.input_tokens,
                                "tokens_out": r.output_tokens,
                            }
                            for r in manager._usage_records
                        ]

                    GLib.idle_add(self._on_records_loaded, records)
                else:
                    # Manager not initialized - use sample data
                    GLib.idle_add(self._on_records_loaded, self._generate_sample_records())
            except Exception as exc:
                logger.warning("Failed to fetch usage data: %s", exc)
                GLib.idle_add(self._on_records_error, str(exc))

        asyncio.create_task(fetch_usage_data())

    def _on_records_loaded(self, records: List[Dict[str, Any]]) -> bool:
        """Handle loaded records on main thread."""
        self._records = records
        self._populate_provider_filter()
        self._apply_filters()
        return False

    def _on_records_error(self, error_msg: str) -> bool:
        """Handle record loading error."""
        logger.warning("Usage history loading error: %s", error_msg)
        self._show_empty_state("Unable to load usage history")
        return False

    def _load_usage_data(self) -> bool:
        """Load usage records from budget manager (sync fallback)."""
        try:
            # Fallback to sample data
            self._records = self._generate_sample_records()
            self._populate_provider_filter()
            self._apply_filters()
        except Exception as exc:
            logger.warning("Failed to load usage history: %s", exc)
            self._show_empty_state("Unable to load usage history")
        return False

    def _generate_sample_records(self) -> List[Dict[str, Any]]:
        """Generate sample records for demonstration."""
        now = datetime.now(timezone.utc)
        samples = []

        for i in range(20):
            record = {
                "id": f"usage_{i}",
                "timestamp": (now - timedelta(hours=i * 6)).isoformat(),
                "provider": ["openai", "anthropic", "google"][i % 3],
                "model": ["gpt-4", "claude-3", "gemini-pro"][i % 3],
                "operation_type": ["llm_completion", "embedding", "image_generation"][i % 3],
                "cost": Decimal(str(round(0.01 + (i * 0.15), 4))),
                "tokens_in": 150 + (i * 50) if i % 3 == 0 else None,
                "tokens_out": 300 + (i * 100) if i % 3 == 0 else None,
            }
            samples.append(record)

        return samples

    def _populate_provider_filter(self) -> None:
        """Populate provider filter with available providers."""
        if not self._provider_combo:
            return

        # Get unique providers
        providers = set()
        for record in self._records:
            provider = record.get("provider")
            if provider:
                providers.add(provider)

        # Clear existing items except "All"
        while True:
            text = self._provider_combo.get_active_text()
            if self._provider_combo.get_active() > 0:
                self._provider_combo.remove(self._provider_combo.get_active())
            else:
                break

        # Add providers
        for provider in sorted(providers):
            self._provider_combo.append(provider, provider.title())

        self._provider_combo.set_active(0)

    def _on_filter_changed(self, *_args) -> None:
        """Handle filter value change."""
        pass  # Filters applied on button click

    def _on_apply_filters(self, _button: Gtk.Button) -> None:
        """Apply current filters."""
        self._apply_filters()

    def _on_clear_filters(self, _button: Gtk.Button) -> None:
        """Clear all filters."""
        if self._provider_combo:
            self._provider_combo.set_active(0)
        if self._type_combo:
            self._type_combo.set_active(0)
        if self._date_from_entry:
            self._date_from_entry.set_text("")
        if self._date_to_entry:
            self._date_to_entry.set_text("")

        self._apply_filters()

    def _apply_filters(self) -> None:
        """Apply filters to the record list."""
        # Get filter values
        provider = None
        if self._provider_combo and self._provider_combo.get_active_id() != "all":
            provider = self._provider_combo.get_active_id()

        op_type = None
        if self._type_combo and self._type_combo.get_active_id() != "all":
            op_type = self._type_combo.get_active_id()

        date_from = None
        if self._date_from_entry:
            try:
                date_str = self._date_from_entry.get_text().strip()
                if date_str:
                    date_from = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        date_to = None
        if self._date_to_entry:
            try:
                date_str = self._date_to_entry.get_text().strip()
                if date_str:
                    date_to = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        # Filter records
        self._filtered_records = []
        for record in self._records:
            if provider and record.get("provider") != provider:
                continue
            if op_type and record.get("operation_type") != op_type:
                continue
            if date_from or date_to:
                try:
                    ts_str = record.get("timestamp", "")
                    if ts_str.endswith("Z"):
                        ts_str = ts_str.replace("Z", "+00:00")
                    record_ts = datetime.fromisoformat(ts_str)
                    if date_from and record_ts < date_from:
                        continue
                    if date_to and record_ts > date_to:
                        continue
                except ValueError:
                    continue

            self._filtered_records.append(record)

        self._populate_list()
        self._update_summary()

    def _populate_list(self) -> None:
        """Populate the list with filtered records."""
        if not self._list_container:
            return

        clear_container(self._list_container)

        if not self._filtered_records:
            self._show_empty_state("No records match the current filters")
            return

        for record in self._filtered_records:
            row = self._create_record_row(record)
            self._list_container.append(row)

    def _create_record_row(self, record: Dict[str, Any]) -> Gtk.ListBoxRow:
        """Create a list row for a usage record."""
        row = Gtk.ListBoxRow()
        row.set_activatable(False)

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(12)
        box.set_margin_end(12)

        # Timestamp
        ts_str = record.get("timestamp", "")
        try:
            if ts_str.endswith("Z"):
                ts_str = ts_str.replace("Z", "+00:00")
            ts = datetime.fromisoformat(ts_str)
            ts_display = ts.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            ts_display = "Unknown"

        ts_label = Gtk.Label(label=ts_display)
        ts_label.set_width_chars(16)
        ts_label.set_xalign(0.0)
        ts_label.add_css_class("monospace")
        box.append(ts_label)

        # Provider/Model
        provider = record.get("provider", "unknown")
        model = record.get("model", "")
        provider_text = f"{provider}/{model}" if model else provider

        provider_label = Gtk.Label(label=provider_text)
        provider_label.set_width_chars(20)
        provider_label.set_xalign(0.0)
        provider_label.set_ellipsize(True)
        box.append(provider_label)

        # Operation type
        op_type = record.get("operation_type", "unknown")
        op_display = op_type.replace("_", " ").title()

        type_label = Gtk.Label(label=op_display)
        type_label.set_width_chars(16)
        type_label.set_xalign(0.0)
        type_label.add_css_class("dim-label")
        box.append(type_label)

        # Tokens (for LLM operations)
        tokens_in = record.get("tokens_in")
        tokens_out = record.get("tokens_out")
        if tokens_in is not None or tokens_out is not None:
            tokens_text = f"{tokens_in or 0}→{tokens_out or 0}"
        else:
            tokens_text = "—"

        tokens_label = Gtk.Label(label=tokens_text)
        tokens_label.set_width_chars(12)
        tokens_label.set_xalign(1.0)
        tokens_label.add_css_class("dim-label")
        box.append(tokens_label)

        # Cost
        cost = record.get("cost", Decimal("0"))
        cost_label = Gtk.Label(label=format_currency(cost))
        cost_label.set_width_chars(10)
        cost_label.set_xalign(1.0)
        cost_label.add_css_class("numeric")
        box.append(cost_label)

        row.set_child(box)
        return row

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

    def _update_summary(self) -> None:
        """Update the summary label."""
        if not self._summary_label:
            return

        count = len(self._filtered_records)
        total = sum(
            r.get("cost", Decimal("0")) for r in self._filtered_records
        )

        self._summary_label.set_text(
            f"Showing {count} records • Total: {format_currency(total)}"
        )

    def _on_export(self, _button: Gtk.Button) -> None:
        """Handle export button click."""
        import csv
        import io

        if not self._filtered_records:
            logger.info("No records to export")
            return

        # Generate CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Timestamp", "Provider", "Model", "Operation", "Tokens In", "Tokens Out", "Cost"])

        for record in self._filtered_records:
            writer.writerow([
                record.get("timestamp", ""),
                record.get("provider", ""),
                record.get("model", ""),
                record.get("operation_type", ""),
                record.get("tokens_in", ""),
                record.get("tokens_out", ""),
                str(record.get("cost", "0")),
            ])

        csv_content = output.getvalue()
        logger.info("Exported %d records to CSV", len(self._filtered_records))

        # Copy to clipboard or save to file
        clipboard = self.get_clipboard()
        if clipboard:
            clipboard.set(csv_content)
            logger.info("Export data copied to clipboard")

    def refresh(self) -> None:
        """Public method to refresh usage history."""
        self._refresh_data()
