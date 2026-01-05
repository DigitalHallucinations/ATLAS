"""Report generation for budget analytics.

Provides usage reporting, aggregation, trend analysis,
and export capabilities.
"""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from modules.logging.logger import setup_logger

from .models import BudgetPeriod, OperationType, SpendSummary, UsageRecord

if TYPE_CHECKING:
    from .manager import BudgetManager

logger = setup_logger(__name__)


class ReportGrouping(Enum):
    """Grouping options for usage reports."""

    NONE = "none"
    PROVIDER = "provider"
    MODEL = "model"
    OPERATION = "operation"
    USER = "user"
    TENANT = "tenant"
    PERSONA = "persona"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class ExportFormat(Enum):
    """Export format options."""

    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class UsageReportRow:
    """Single row in a usage report.

    Attributes:
        group_key: Grouping key (provider name, date, etc.).
        total_cost: Total cost for this group.
        record_count: Number of usage records.
        total_tokens: Total tokens used.
        total_images: Total images generated.
        avg_cost_per_request: Average cost per request.
        breakdown: Sub-breakdown if multiple groupings.
    """

    group_key: str
    total_cost: Decimal
    record_count: int = 0
    total_tokens: int = 0
    total_images: int = 0
    avg_cost_per_request: Decimal = field(default_factory=lambda: Decimal("0"))
    breakdown: Dict[str, "UsageReportRow"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate derived fields."""
        if self.record_count > 0 and self.avg_cost_per_request == Decimal("0"):
            self.avg_cost_per_request = self.total_cost / Decimal(str(self.record_count))

    def as_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "group_key": self.group_key,
            "total_cost": str(self.total_cost),
            "record_count": self.record_count,
            "total_tokens": self.total_tokens,
            "total_images": self.total_images,
            "avg_cost_per_request": str(self.avg_cost_per_request),
            "breakdown": {k: v.as_dict() for k, v in self.breakdown.items()},
        }


@dataclass
class UsageReport:
    """Complete usage report.

    Attributes:
        title: Report title.
        start_date: Report start date.
        end_date: Report end date.
        generated_at: When report was generated.
        total_cost: Total cost in period.
        total_records: Total number of usage records.
        rows: Report data rows.
        groupings: Groupings used.
        currency: Currency code.
        metadata: Additional report metadata.
    """

    title: str
    start_date: datetime
    end_date: datetime
    total_cost: Decimal
    total_records: int
    rows: List[UsageReportRow]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    groupings: List[ReportGrouping] = field(default_factory=list)
    currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def avg_daily_cost(self) -> Decimal:
        """Calculate average daily cost."""
        days = max(1, (self.end_date - self.start_date).days)
        return self.total_cost / Decimal(str(days))

    @property
    def projected_monthly_cost(self) -> Decimal:
        """Project monthly cost based on current rate."""
        return self.avg_daily_cost * Decimal("30")

    def as_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "title": self.title,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "total_cost": str(self.total_cost),
            "total_records": self.total_records,
            "avg_daily_cost": str(self.avg_daily_cost),
            "projected_monthly_cost": str(self.projected_monthly_cost),
            "groupings": [g.value for g in self.groupings],
            "currency": self.currency,
            "rows": [r.as_dict() for r in self.rows],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.as_dict(), indent=indent)

    def to_csv(self) -> str:
        """Export to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        headers = ["Group", "Total Cost", "Records", "Tokens", "Images", "Avg Cost"]
        writer.writerow(headers)

        # Rows
        for row in self.rows:
            writer.writerow([
                row.group_key,
                f"{row.total_cost:.6f}",
                row.record_count,
                row.total_tokens,
                row.total_images,
                f"{row.avg_cost_per_request:.6f}",
            ])

        # Summary
        writer.writerow([])
        writer.writerow(["Total", f"{self.total_cost:.6f}", self.total_records])

        return output.getvalue()

    def to_markdown(self) -> str:
        """Export to Markdown string."""
        lines = [
            f"# {self.title}",
            "",
            f"**Period:** {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Summary",
            "",
            f"- **Total Cost:** ${self.total_cost:.2f} {self.currency}",
            f"- **Total Requests:** {self.total_records:,}",
            f"- **Avg Daily Cost:** ${self.avg_daily_cost:.2f}",
            f"- **Projected Monthly:** ${self.projected_monthly_cost:.2f}",
            "",
            "## Breakdown",
            "",
            "| Group | Cost | Requests | Tokens | Images | Avg/Request |",
            "|-------|------|----------|--------|--------|-------------|",
        ]

        for row in self.rows:
            lines.append(
                f"| {row.group_key} | ${row.total_cost:.4f} | "
                f"{row.record_count:,} | {row.total_tokens:,} | "
                f"{row.total_images} | ${row.avg_cost_per_request:.6f} |"
            )

        return "\n".join(lines)

    def to_html(self) -> str:
        """Export to HTML string.

        Generates a styled HTML report suitable for printing to PDF
        or viewing in a browser.

        Returns:
            HTML string with embedded CSS styling.
        """
        # Generate table rows
        table_rows = []
        for row in self.rows:
            table_rows.append(f"""
                <tr>
                    <td>{row.group_key}</td>
                    <td class="number">${row.total_cost:.4f}</td>
                    <td class="number">{row.record_count:,}</td>
                    <td class="number">{row.total_tokens:,}</td>
                    <td class="number">{row.total_images}</td>
                    <td class="number">${row.avg_cost_per_request:.6f}</td>
                </tr>""")

        rows_html = "".join(table_rows)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .meta {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .summary {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .summary-item {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-label {{
            font-size: 0.85em;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .summary-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f5f6fa;
        }}
        .number {{
            text-align: right;
            font-family: 'Courier New', monospace;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            font-size: 0.85em;
        }}
        @media print {{
            body {{ padding: 0; }}
            .summary {{ background: #f9f9f9; -webkit-print-color-adjust: exact; }}
        }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <div class="meta">
        <p><strong>Period:</strong> {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}</p>
        <p><strong>Generated:</strong> {self.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>

    <h2>Summary</h2>
    <div class="summary">
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-label">Total Cost</div>
                <div class="summary-value">${self.total_cost:.2f} {self.currency}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Total Requests</div>
                <div class="summary-value">{self.total_records:,}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Average Daily</div>
                <div class="summary-value">${self.avg_daily_cost:.2f}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Projected Monthly</div>
                <div class="summary-value">${self.projected_monthly_cost:.2f}</div>
            </div>
        </div>
    </div>

    <h2>Breakdown</h2>
    <table>
        <thead>
            <tr>
                <th>Group</th>
                <th class="number">Cost</th>
                <th class="number">Requests</th>
                <th class="number">Tokens</th>
                <th class="number">Images</th>
                <th class="number">Avg/Request</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <div class="footer">
        <p>Generated by ATLAS Budget Manager</p>
    </div>
</body>
</html>"""
        return html

class ReportGenerator:
    """Generates usage reports from budget data.

    Provides various report types including:
    - Summary reports
    - Detailed usage reports
    - Trend analysis
    - Provider/model comparisons
    - Cost projections

    Usage::

        generator = ReportGenerator(budget_manager)

        # Generate monthly report
        report = await generator.generate_report(
            start_date=datetime(2026, 1, 1),
            end_date=datetime(2026, 1, 31),
            group_by=[ReportGrouping.PROVIDER, ReportGrouping.MODEL],
        )

        # Export to different formats
        json_str = report.to_json()
        csv_str = report.to_csv()
        md_str = report.to_markdown()
    """

    def __init__(self, budget_manager: Optional["BudgetManager"] = None):
        """Initialize the report generator.

        Args:
            budget_manager: Budget manager for data access.
        """
        self.budget_manager = budget_manager
        self.logger = setup_logger(__name__)

    async def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: Optional[List[ReportGrouping]] = None,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> UsageReport:
        """Generate a usage report.

        Args:
            start_date: Report start date.
            end_date: Report end date.
            group_by: Grouping options.
            title: Report title.
            user_id: Filter by user.
            tenant_id: Filter by tenant.
            provider: Filter by provider.
            model: Filter by model.

        Returns:
            UsageReport with aggregated data.
        """
        group_by = group_by or [ReportGrouping.PROVIDER]
        title = title or f"Usage Report: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

        # Get usage records
        records = await self._get_records(
            start_date=start_date,
            end_date=end_date,
            user_id=user_id,
            tenant_id=tenant_id,
            provider=provider,
            model=model,
        )

        # Aggregate by groupings
        rows = self._aggregate_records(records, group_by)

        # Calculate totals
        total_cost = sum((r.total_cost for r in rows), Decimal("0"))
        total_records = sum(r.record_count for r in rows)

        return UsageReport(
            title=title,
            start_date=start_date,
            end_date=end_date,
            total_cost=total_cost,
            total_records=total_records,
            rows=rows,
            groupings=group_by,
            metadata={
                "filters": {
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "provider": provider,
                    "model": model,
                },
            },
        )

    async def _get_records(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[UsageRecord]:
        """Get usage records matching criteria.

        Args:
            start_date: Start of period.
            end_date: End of period.
            user_id: Filter by user.
            tenant_id: Filter by tenant.
            provider: Filter by provider.
            model: Filter by model.

        Returns:
            List of matching UsageRecords.
        """
        if self.budget_manager is None:
            return []

        # Get records from budget manager's buffer
        async with self.budget_manager._usage_lock:
            records = list(self.budget_manager._usage_records)

        # Also query persisted records via persistence layer
        if self.budget_manager._persistence is not None:
            try:
                persisted = await self.budget_manager._persistence.get_usage_records(
                    start_date=start_date,
                    end_date=end_date,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    provider=provider,
                    model=model,
                )
                records.extend(persisted)
            except Exception as exc:
                logger.warning("Failed to query persisted records: %s", exc)

        # Filter by date range
        records = [
            r for r in records
            if start_date <= r.timestamp <= end_date
        ]

        # Apply additional filters
        if user_id:
            records = [r for r in records if r.user_id == user_id]
        if tenant_id:
            records = [r for r in records if r.tenant_id == tenant_id]
        if provider:
            records = [r for r in records if r.provider == provider]
        if model:
            records = [r for r in records if r.model == model]

        return records

    def _aggregate_records(
        self,
        records: List[UsageRecord],
        group_by: List[ReportGrouping],
    ) -> List[UsageReportRow]:
        """Aggregate records by grouping.

        Args:
            records: Usage records to aggregate.
            group_by: Grouping options.

        Returns:
            List of aggregated report rows.
        """
        if not records:
            return []

        primary_grouping = group_by[0] if group_by else ReportGrouping.NONE

        # Group records
        groups: Dict[str, List[UsageRecord]] = defaultdict(list)

        for record in records:
            key = self._get_group_key(record, primary_grouping)
            groups[key].append(record)

        # Aggregate each group
        rows: List[UsageReportRow] = []

        for group_key, group_records in sorted(groups.items()):
            total_cost = sum((r.cost_usd for r in group_records), Decimal("0"))
            total_tokens = sum(
                (r.total_tokens or 0) for r in group_records
            )
            total_images = sum(
                (r.images_generated or 0) for r in group_records
            )

            # Secondary grouping
            breakdown: Dict[str, UsageReportRow] = {}
            if len(group_by) > 1:
                secondary = group_by[1]
                sub_groups: Dict[str, List[UsageRecord]] = defaultdict(list)
                for record in group_records:
                    sub_key = self._get_group_key(record, secondary)
                    sub_groups[sub_key].append(record)

                for sub_key, sub_records in sub_groups.items():
                    sub_cost = sum((r.cost_usd for r in sub_records), Decimal("0"))
                    sub_tokens = sum((r.total_tokens or 0) for r in sub_records)
                    sub_images = sum((r.images_generated or 0) for r in sub_records)
                    breakdown[sub_key] = UsageReportRow(
                        group_key=sub_key,
                        total_cost=sub_cost,
                        record_count=len(sub_records),
                        total_tokens=sub_tokens,
                        total_images=sub_images,
                    )

            rows.append(
                UsageReportRow(
                    group_key=group_key,
                    total_cost=total_cost,
                    record_count=len(group_records),
                    total_tokens=total_tokens,
                    total_images=total_images,
                    breakdown=breakdown,
                )
            )

        # Sort by cost descending
        rows.sort(key=lambda r: r.total_cost, reverse=True)

        return rows

    def _get_group_key(
        self,
        record: UsageRecord,
        grouping: ReportGrouping,
    ) -> str:
        """Get grouping key for a record.

        Args:
            record: Usage record.
            grouping: Grouping type.

        Returns:
            Group key string.
        """
        if grouping == ReportGrouping.PROVIDER:
            return record.provider
        elif grouping == ReportGrouping.MODEL:
            return record.model
        elif grouping == ReportGrouping.OPERATION:
            return record.operation_type.value
        elif grouping == ReportGrouping.USER:
            return record.user_id or "unknown"
        elif grouping == ReportGrouping.TENANT:
            return record.tenant_id or "default"
        elif grouping == ReportGrouping.PERSONA:
            return record.persona or "default"
        elif grouping == ReportGrouping.HOUR:
            return record.timestamp.strftime("%Y-%m-%d %H:00")
        elif grouping == ReportGrouping.DAY:
            return record.timestamp.strftime("%Y-%m-%d")
        elif grouping == ReportGrouping.WEEK:
            # ISO week
            return record.timestamp.strftime("%Y-W%W")
        elif grouping == ReportGrouping.MONTH:
            return record.timestamp.strftime("%Y-%m")
        else:
            return "all"

    # =========================================================================
    # Specialized Reports
    # =========================================================================

    async def generate_summary_report(
        self,
        period: BudgetPeriod = BudgetPeriod.MONTHLY,
    ) -> UsageReport:
        """Generate a summary report for the current period.

        Args:
            period: Budget period.

        Returns:
            UsageReport summary.
        """
        # Calculate period boundaries
        now = datetime.now(timezone.utc)

        if period == BudgetPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == BudgetPeriod.WEEKLY:
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(weeks=1)
        elif period == BudgetPeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
        else:
            start = now - timedelta(days=30)
            end = now

        return await self.generate_report(
            start_date=start,
            end_date=end,
            group_by=[ReportGrouping.PROVIDER, ReportGrouping.MODEL],
            title=f"{period.value.title()} Usage Summary",
        )

    async def generate_trend_report(
        self,
        days: int = 30,
        interval: ReportGrouping = ReportGrouping.DAY,
    ) -> UsageReport:
        """Generate a trend report over time.

        Args:
            days: Number of days to include.
            interval: Time interval for grouping.

        Returns:
            UsageReport with time-series data.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        return await self.generate_report(
            start_date=start,
            end_date=end,
            group_by=[interval, ReportGrouping.PROVIDER],
            title=f"{days}-Day Trend Report",
        )

    async def generate_comparison_report(
        self,
        current_start: datetime,
        current_end: datetime,
        previous_start: datetime,
        previous_end: datetime,
    ) -> Dict[str, Any]:
        """Generate a comparison report between two periods.

        Args:
            current_start: Current period start.
            current_end: Current period end.
            previous_start: Previous period start.
            previous_end: Previous period end.

        Returns:
            Comparison data dictionary.
        """
        current = await self.generate_report(
            start_date=current_start,
            end_date=current_end,
            group_by=[ReportGrouping.PROVIDER],
            title="Current Period",
        )

        previous = await self.generate_report(
            start_date=previous_start,
            end_date=previous_end,
            group_by=[ReportGrouping.PROVIDER],
            title="Previous Period",
        )

        # Calculate changes
        cost_change = current.total_cost - previous.total_cost
        cost_change_pct = (
            float(cost_change / previous.total_cost * 100)
            if previous.total_cost > 0 else 0
        )

        record_change = current.total_records - previous.total_records
        record_change_pct = (
            record_change / previous.total_records * 100
            if previous.total_records > 0 else 0
        )

        return {
            "current_period": current.as_dict(),
            "previous_period": previous.as_dict(),
            "comparison": {
                "cost_change": str(cost_change),
                "cost_change_percent": cost_change_pct,
                "record_change": record_change,
                "record_change_percent": record_change_pct,
                "cost_trend": "up" if cost_change > 0 else "down" if cost_change < 0 else "flat",
            },
        }

    async def generate_projection_report(
        self,
        days_to_project: int = 30,
    ) -> Dict[str, Any]:
        """Generate a cost projection report.

        Args:
            days_to_project: Number of days to project.

        Returns:
            Projection data dictionary.
        """
        # Get last 30 days of data
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)

        report = await self.generate_report(
            start_date=start,
            end_date=end,
            group_by=[ReportGrouping.DAY],
        )

        # Calculate daily average
        actual_days = max(1, (end - start).days)
        daily_avg = report.total_cost / Decimal(str(actual_days))

        # Project forward
        projected_cost = daily_avg * Decimal(str(days_to_project))

        # Get current budget policies for context
        budget_context: Dict[str, Any] = {}
        if self.budget_manager:
            from .models import BudgetScope
            policies = await self.budget_manager.get_policies(
                scope=BudgetScope.GLOBAL
            )
            if policies:
                policy = policies[0]
                budget_context = {
                    "policy_name": policy.name,
                    "limit_amount": str(policy.limit_amount),
                    "period": policy.period.value,
                    "projected_percent_of_budget": float(
                        projected_cost / policy.limit_amount * 100
                    ) if policy.limit_amount > 0 else None,
                }

        return {
            "analysis_period_days": actual_days,
            "daily_average": str(daily_avg),
            "projection_days": days_to_project,
            "projected_cost": str(projected_cost),
            "confidence": "medium" if actual_days >= 14 else "low",
            "budget_context": budget_context,
            "methodology": "linear_extrapolation",
        }

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_report(
        self,
        report: UsageReport,
        format: ExportFormat = ExportFormat.JSON,
    ) -> str:
        """Export a report to the specified format.

        Args:
            report: Report to export.
            format: Export format.

        Returns:
            Formatted string.
        """
        if format == ExportFormat.JSON:
            return report.to_json()
        elif format == ExportFormat.CSV:
            return report.to_csv()
        elif format == ExportFormat.MARKDOWN:
            return report.to_markdown()
        elif format == ExportFormat.HTML:
            return report.to_html()
        else:
            return report.to_json()

    async def export_raw_records(
        self,
        start_date: datetime,
        end_date: datetime,
        format: ExportFormat = ExportFormat.CSV,
    ) -> str:
        """Export raw usage records.

        Args:
            start_date: Start of period.
            end_date: End of period.
            format: Export format.

        Returns:
            Formatted string with raw records.
        """
        records = await self._get_records(start_date, end_date)

        if format == ExportFormat.JSON:
            return json.dumps(
                [r.as_dict() for r in records],
                indent=2,
            )

        elif format == ExportFormat.CSV:
            output = io.StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow([
                "timestamp", "provider", "model", "operation",
                "input_tokens", "output_tokens", "images",
                "cost_usd", "user_id", "conversation_id",
            ])

            # Records
            for r in records:
                writer.writerow([
                    r.timestamp.isoformat(),
                    r.provider,
                    r.model,
                    r.operation_type.value,
                    r.input_tokens or "",
                    r.output_tokens or "",
                    r.images_generated or "",
                    str(r.cost_usd),
                    r.user_id or "",
                    r.conversation_id or "",
                ])

            return output.getvalue()

        else:
            return json.dumps([r.as_dict() for r in records], indent=2)
