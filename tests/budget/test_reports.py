"""Tests for budget reporting functionality."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from modules.budget.models import BudgetPeriod, OperationType
from modules.budget.reports import (
    ExportFormat,
    ReportGenerator,
    ReportGrouping,
    UsageReport,
    UsageReportRow,
)


@pytest.fixture
def mock_budget_manager() -> MagicMock:
    """Create a mock BudgetManager for testing."""
    manager = MagicMock()
    manager._usage_lock = AsyncMock()
    manager._usage_records = []
    # Mock async methods
    manager.get_policies = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def sample_report_rows() -> list[UsageReportRow]:
    """Create sample report rows."""
    return [
        UsageReportRow(
            group_key="openai",
            total_cost=Decimal("10.50"),
            record_count=100,
            total_tokens=50000,
        ),
        UsageReportRow(
            group_key="anthropic",
            total_cost=Decimal("8.25"),
            record_count=75,
            total_tokens=40000,
        ),
    ]


@pytest.fixture
def sample_report(sample_report_rows: list[UsageReportRow]) -> UsageReport:
    """Create a sample UsageReport."""
    now = datetime.now(timezone.utc)
    return UsageReport(
        title="Test Report",
        start_date=now - timedelta(days=30),
        end_date=now,
        total_cost=Decimal("18.75"),
        total_records=175,
        rows=sample_report_rows,
        groupings=[ReportGrouping.PROVIDER],
    )


class TestReportGroupingEnum:
    """Tests for ReportGrouping enumeration."""

    def test_report_grouping_values(self) -> None:
        """Test that ReportGrouping has expected values."""
        assert ReportGrouping.PROVIDER is not None
        assert ReportGrouping.MODEL is not None
        assert ReportGrouping.OPERATION is not None
        assert ReportGrouping.DAY is not None
        assert ReportGrouping.MONTH is not None

    def test_report_grouping_from_string(self) -> None:
        """Test creating ReportGrouping from string."""
        provider = ReportGrouping("provider")
        assert provider == ReportGrouping.PROVIDER

        model = ReportGrouping("model")
        assert model == ReportGrouping.MODEL


class TestExportFormatEnum:
    """Tests for ExportFormat enumeration."""

    def test_export_format_values(self) -> None:
        """Test that ExportFormat has expected values."""
        assert ExportFormat.JSON is not None
        assert ExportFormat.CSV is not None
        assert ExportFormat.MARKDOWN is not None
        assert ExportFormat.HTML is not None

    def test_export_format_from_string(self) -> None:
        """Test creating ExportFormat from string."""
        json_format = ExportFormat("json")
        assert json_format == ExportFormat.JSON

        html_format = ExportFormat("html")
        assert html_format == ExportFormat.HTML

        csv_format = ExportFormat("csv")
        assert csv_format == ExportFormat.CSV


class TestUsageReportRow:
    """Tests for UsageReportRow dataclass."""

    def test_create_report_row(self) -> None:
        """Test creating a basic report row."""
        row = UsageReportRow(
            group_key="openai",
            total_cost=Decimal("10.00"),
            record_count=50,
            total_tokens=25000,
        )

        assert row.group_key == "openai"
        assert row.total_cost == Decimal("10.00")
        assert row.record_count == 50
        assert row.total_tokens == 25000

    def test_avg_cost_calculated(self) -> None:
        """Test that average cost per request is calculated."""
        row = UsageReportRow(
            group_key="test",
            total_cost=Decimal("100.00"),
            record_count=50,
        )

        # Should auto-calculate: 100 / 50 = 2.00
        assert row.avg_cost_per_request == Decimal("2.00")

    def test_row_as_dict(self) -> None:
        """Test converting row to dictionary."""
        row = UsageReportRow(
            group_key="openai",
            total_cost=Decimal("10.00"),
            record_count=50,
        )

        data = row.as_dict()

        assert isinstance(data, dict)
        assert data["group_key"] == "openai"
        assert "total_cost" in data


class TestUsageReport:
    """Tests for UsageReport dataclass."""

    def test_create_usage_report(self, sample_report: UsageReport) -> None:
        """Test creating a usage report."""
        assert sample_report.title == "Test Report"
        assert sample_report.total_cost == Decimal("18.75")
        assert sample_report.total_records == 175
        assert len(sample_report.rows) == 2

    def test_report_to_json(self, sample_report: UsageReport) -> None:
        """Test exporting report to JSON."""
        json_str = sample_report.to_json()

        assert isinstance(json_str, str)
        assert "Test Report" in json_str
        assert "18.75" in json_str or "18.750000" in json_str

    def test_report_to_csv(self, sample_report: UsageReport) -> None:
        """Test exporting report to CSV."""
        csv_str = sample_report.to_csv()

        assert isinstance(csv_str, str)
        # CSV should have headers and data rows
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 2  # Header + data

    def test_report_to_markdown(self, sample_report: UsageReport) -> None:
        """Test exporting report to Markdown."""
        md_str = sample_report.to_markdown()

        assert isinstance(md_str, str)
        assert "# Test Report" in md_str
        assert "|" in md_str  # Table formatting

    def test_report_to_html(self, sample_report: UsageReport) -> None:
        """Test exporting report to HTML."""
        html_str = sample_report.to_html()

        assert isinstance(html_str, str)
        assert "<!DOCTYPE html>" in html_str
        assert "<title>Test Report</title>" in html_str
        assert "<table>" in html_str
        # Check for summary values
        assert "$18.75" in html_str or "18.75" in html_str
        # Check for styled elements
        assert "class=\"summary\"" in html_str
        assert "class=\"number\"" in html_str


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def test_create_generator(self, mock_budget_manager: MagicMock) -> None:
        """Test creating a ReportGenerator."""
        generator = ReportGenerator(mock_budget_manager)

        assert generator is not None
        assert generator.budget_manager is mock_budget_manager

    def test_create_generator_without_manager(self) -> None:
        """Test creating generator without budget manager."""
        generator = ReportGenerator()

        assert generator is not None
        assert generator.budget_manager is None

    @pytest.mark.asyncio
    async def test_generate_report_basic(
        self, mock_budget_manager: MagicMock
    ) -> None:
        """Test generating a basic report."""
        generator = ReportGenerator(mock_budget_manager)

        now = datetime.now(timezone.utc)
        report = await generator.generate_report(
            start_date=now - timedelta(days=30),
            end_date=now,
        )

        assert isinstance(report, UsageReport)
        assert report.start_date < report.end_date

    @pytest.mark.asyncio
    async def test_generate_report_grouped_by_provider(
        self, mock_budget_manager: MagicMock
    ) -> None:
        """Test report grouped by provider."""
        generator = ReportGenerator(mock_budget_manager)

        now = datetime.now(timezone.utc)
        report = await generator.generate_report(
            start_date=now - timedelta(days=7),
            end_date=now,
            group_by=[ReportGrouping.PROVIDER],
        )

        assert report is not None
        assert ReportGrouping.PROVIDER in report.groupings

    @pytest.mark.asyncio
    async def test_generate_report_grouped_by_model(
        self, mock_budget_manager: MagicMock
    ) -> None:
        """Test report grouped by model."""
        generator = ReportGenerator(mock_budget_manager)

        now = datetime.now(timezone.utc)
        report = await generator.generate_report(
            start_date=now - timedelta(days=7),
            end_date=now,
            group_by=[ReportGrouping.MODEL],
        )

        assert report is not None
        assert ReportGrouping.MODEL in report.groupings

    @pytest.mark.asyncio
    async def test_generate_report_with_filters(
        self, mock_budget_manager: MagicMock
    ) -> None:
        """Test report with filters applied."""
        generator = ReportGenerator(mock_budget_manager)

        now = datetime.now(timezone.utc)
        report = await generator.generate_report(
            start_date=now - timedelta(days=7),
            end_date=now,
            provider="openai",
            model="gpt-4o",
        )

        assert report is not None
        # Filters should be in metadata
        assert report.metadata["filters"]["provider"] == "openai"
        assert report.metadata["filters"]["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate_report_with_title(
        self, mock_budget_manager: MagicMock
    ) -> None:
        """Test report with custom title."""
        generator = ReportGenerator(mock_budget_manager)

        now = datetime.now(timezone.utc)
        report = await generator.generate_report(
            start_date=now - timedelta(days=7),
            end_date=now,
            title="Custom Report Title",
        )

        assert report.title == "Custom Report Title"

    @pytest.mark.asyncio
    async def test_generate_summary_report(
        self, mock_budget_manager: MagicMock
    ) -> None:
        """Test generating a summary report."""
        generator = ReportGenerator(mock_budget_manager)

        summary = await generator.generate_summary_report(period=BudgetPeriod.MONTHLY)

        assert summary is not None

    @pytest.mark.asyncio
    async def test_generate_trend_report(
        self, mock_budget_manager: MagicMock
    ) -> None:
        """Test generating a trend report."""
        generator = ReportGenerator(mock_budget_manager)

        trend = await generator.generate_trend_report(
            days=14,
            interval=ReportGrouping.DAY,
        )

        assert trend is not None

    @pytest.mark.asyncio
    async def test_generate_comparison_report(
        self, mock_budget_manager: MagicMock
    ) -> None:
        """Test generating a comparison report."""
        generator = ReportGenerator(mock_budget_manager)

        now = datetime.now(timezone.utc)
        comparison = await generator.generate_comparison_report(
            current_start=now - timedelta(days=30),
            current_end=now,
            previous_start=now - timedelta(days=60),
            previous_end=now - timedelta(days=30),
        )

        assert comparison is not None

    @pytest.mark.asyncio
    async def test_generate_projection_report(
        self, mock_budget_manager: MagicMock
    ) -> None:
        """Test generating a projection report."""
        generator = ReportGenerator(mock_budget_manager)

        projection = await generator.generate_projection_report(
            days_to_project=30,
        )

        assert projection is not None


class TestReportExportMethods:
    """Tests for report export functionality."""

    def test_export_json_format(self, sample_report: UsageReport) -> None:
        """Test JSON export produces valid JSON."""
        import json

        json_str = sample_report.to_json()
        data = json.loads(json_str)

        assert isinstance(data, dict)
        assert "title" in data
        assert "total_cost" in data

    def test_export_csv_headers(self, sample_report: UsageReport) -> None:
        """Test CSV export has proper headers."""
        csv_str = sample_report.to_csv()
        lines = csv_str.strip().split("\n")

        # First line should be headers
        headers = lines[0].lower()
        assert "group" in headers or "key" in headers

    def test_export_markdown_structure(self, sample_report: UsageReport) -> None:
        """Test Markdown export has proper structure."""
        md_str = sample_report.to_markdown()

        # Should have title heading
        assert md_str.startswith("#")
        # Should have table
        assert "| Group |" in md_str or "|" in md_str

    def test_export_html_structure(self, sample_report: UsageReport) -> None:
        """Test HTML export has proper structure."""
        html_str = sample_report.to_html()

        # Should have HTML document structure
        assert "<!DOCTYPE html>" in html_str
        assert "<html" in html_str
        assert "</html>" in html_str
        # Should have CSS styling
        assert "<style>" in html_str
        # Should have table
        assert "<table>" in html_str
        assert "<thead>" in html_str
        assert "<tbody>" in html_str
        # Should have summary section
        assert "summary-grid" in html_str


class TestMultipleGroupings:
    """Tests for reports with multiple groupings."""

    @pytest.mark.asyncio
    async def test_multiple_groupings(
        self, mock_budget_manager: MagicMock
    ) -> None:
        """Test report with multiple grouping levels."""
        generator = ReportGenerator(mock_budget_manager)

        now = datetime.now(timezone.utc)
        report = await generator.generate_report(
            start_date=now - timedelta(days=7),
            end_date=now,
            group_by=[ReportGrouping.PROVIDER, ReportGrouping.MODEL],
        )

        assert report is not None
        assert len(report.groupings) == 2
        assert ReportGrouping.PROVIDER in report.groupings
        assert ReportGrouping.MODEL in report.groupings
