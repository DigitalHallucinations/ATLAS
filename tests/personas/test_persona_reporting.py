"""
Tests for PersonaReportGenerator.

Tests automated report generation and scheduling.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.services.personas.types import (
    PersonaPerformanceMetrics,
    TokenUsage,
    ImprovementArea,
)
from core.services.personas.reporting import (
    PersonaReportGenerator,
    ReportConfig,
    ReportFormat,
    ReportFrequency,
    ReportResult,
)


@pytest.fixture
def mock_analytics_service() -> MagicMock:
    """Create a mock PersonaAnalyticsService."""
    service = MagicMock()
    service.get_metrics = AsyncMock()
    service.compare_personas = AsyncMock()
    service.identify_improvement_areas = AsyncMock()
    return service


@pytest.fixture
def sample_metrics() -> PersonaPerformanceMetrics:
    """Create sample metrics for testing."""
    return PersonaPerformanceMetrics(
        persona_id="test-persona",
        period_start=datetime.now(timezone.utc) - timedelta(days=7),
        period_end=datetime.now(timezone.utc),
        total_interactions=100,
        avg_response_time_ms=1500.0,
        token_usage=TokenUsage(
            prompt_tokens=5000,
            completion_tokens=3000,
            total_tokens=8000,
        ),
        task_success_rate=0.85,
        escalation_rate=0.1,
        retry_rate=0.05,
        tools_used={"calculator": 20, "search": 30},
        skills_invoked={"math": 15, "research": 25},
        capability_gaps=["image_generation"],
    )


@pytest.fixture
def report_config(tmp_path: Path) -> ReportConfig:
    """Create a sample report configuration."""
    return ReportConfig(
        report_id="test-report",
        name="Test Analytics Report",
        persona_ids=["persona1", "persona2"],
        frequency=ReportFrequency.WEEKLY,
        formats=[ReportFormat.JSON, ReportFormat.MARKDOWN],
        output_directory=tmp_path / "reports",
        include_comparisons=True,
        include_anomalies=True,
        include_improvements=True,
    )


class TestPersonaReportGenerator:
    """Tests for PersonaReportGenerator."""
    
    def test_init(self, mock_analytics_service: MagicMock) -> None:
        """Test generator initialization."""
        generator = PersonaReportGenerator(mock_analytics_service)
        assert generator._service == mock_analytics_service
        assert len(generator._report_configs) == 0
    
    def test_register_report(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
    ) -> None:
        """Test registering a report configuration."""
        generator = PersonaReportGenerator(mock_analytics_service)
        
        generator.register_report(report_config)
        
        assert report_config.report_id in generator._report_configs
        assert generator.get_report_config(report_config.report_id) == report_config
    
    def test_unregister_report(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
    ) -> None:
        """Test unregistering a report."""
        generator = PersonaReportGenerator(mock_analytics_service)
        generator.register_report(report_config)
        
        result = generator.unregister_report(report_config.report_id)
        
        assert result is True
        assert generator.get_report_config(report_config.report_id) is None
    
    def test_unregister_nonexistent_report(
        self,
        mock_analytics_service: MagicMock,
    ) -> None:
        """Test unregistering a non-existent report."""
        generator = PersonaReportGenerator(mock_analytics_service)
        
        result = generator.unregister_report("nonexistent")
        
        assert result is False
    
    def test_get_report_configs(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
    ) -> None:
        """Test getting all report configurations."""
        generator = PersonaReportGenerator(mock_analytics_service)
        generator.register_report(report_config)
        
        configs = generator.get_report_configs()
        
        assert len(configs) == 1
        assert configs[0] == report_config
    
    @pytest.mark.asyncio
    async def test_generate_report_success(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test successful report generation."""
        mock_analytics_service.get_metrics.return_value = sample_metrics
        mock_analytics_service.compare_personas.return_value = {
            "persona1": sample_metrics,
        }
        mock_analytics_service.identify_improvement_areas.return_value = []
        
        generator = PersonaReportGenerator(mock_analytics_service)
        
        result = await generator.generate_report(report_config)
        
        assert result.success is True
        assert result.report_id == report_config.report_id
        assert len(result.output_files) == 2  # JSON and Markdown
        assert result.persona_count == 2
        
        # Verify files were created
        for file_path in result.output_files:
            assert file_path.exists()
    
    @pytest.mark.asyncio
    async def test_generate_report_json_format(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test JSON report format."""
        mock_analytics_service.get_metrics.return_value = sample_metrics
        mock_analytics_service.compare_personas.return_value = {}
        mock_analytics_service.identify_improvement_areas.return_value = []
        
        report_config.formats = [ReportFormat.JSON]
        generator = PersonaReportGenerator(mock_analytics_service)
        
        result = await generator.generate_report(report_config)
        
        assert result.success
        json_file = result.output_files[0]
        
        content = json.loads(json_file.read_text())
        assert "report_name" in content
        assert "personas" in content
        assert content["report_name"] == report_config.name
    
    @pytest.mark.asyncio
    async def test_generate_report_csv_format(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test CSV report format."""
        mock_analytics_service.get_metrics.return_value = sample_metrics
        mock_analytics_service.compare_personas.return_value = {}
        mock_analytics_service.identify_improvement_areas.return_value = []
        
        report_config.formats = [ReportFormat.CSV]
        generator = PersonaReportGenerator(mock_analytics_service)
        
        result = await generator.generate_report(report_config)
        
        assert result.success
        csv_file = result.output_files[0]
        
        content = csv_file.read_text()
        assert "Persona ID" in content
        assert "Total Interactions" in content
    
    @pytest.mark.asyncio
    async def test_generate_report_markdown_format(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test Markdown report format."""
        mock_analytics_service.get_metrics.return_value = sample_metrics
        mock_analytics_service.compare_personas.return_value = {}
        mock_analytics_service.identify_improvement_areas.return_value = []
        
        report_config.formats = [ReportFormat.MARKDOWN]
        generator = PersonaReportGenerator(mock_analytics_service)
        
        result = await generator.generate_report(report_config)
        
        assert result.success
        md_file = result.output_files[0]
        
        content = md_file.read_text()
        assert f"# {report_config.name}" in content
        assert "## Performance Summary" in content
    
    @pytest.mark.asyncio
    async def test_generate_report_html_format(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test HTML report format."""
        mock_analytics_service.get_metrics.return_value = sample_metrics
        mock_analytics_service.compare_personas.return_value = {}
        mock_analytics_service.identify_improvement_areas.return_value = []
        
        report_config.formats = [ReportFormat.HTML]
        generator = PersonaReportGenerator(mock_analytics_service)
        
        result = await generator.generate_report(report_config)
        
        assert result.success
        html_file = result.output_files[0]
        
        content = html_file.read_text()
        assert "<!DOCTYPE html>" in content
        assert report_config.name in content
    
    @pytest.mark.asyncio
    async def test_generate_report_error_handling(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
    ) -> None:
        """Test error handling during report generation."""
        mock_analytics_service.get_metrics.side_effect = Exception("Service error")
        
        generator = PersonaReportGenerator(mock_analytics_service)
        
        result = await generator.generate_report(report_config)
        
        # Should still succeed but with error data for the persona
        assert result.success is True  # Overall generation succeeds
    
    @pytest.mark.asyncio
    async def test_generate_on_demand_report(
        self,
        mock_analytics_service: MagicMock,
        tmp_path: Path,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test on-demand report generation."""
        mock_analytics_service.get_metrics.return_value = sample_metrics
        mock_analytics_service.compare_personas.return_value = {}
        mock_analytics_service.identify_improvement_areas.return_value = []
        
        generator = PersonaReportGenerator(mock_analytics_service)
        
        result = await generator.generate_on_demand_report(
            persona_ids=["test-persona"],
            formats=[ReportFormat.JSON],
            output_directory=tmp_path / "on_demand",
            name="On Demand Test",
        )
        
        assert result.success
        assert "on_demand" in result.report_id
    
    def test_calculate_period_start_daily(
        self,
        mock_analytics_service: MagicMock,
    ) -> None:
        """Test period calculation for daily frequency."""
        generator = PersonaReportGenerator(mock_analytics_service)
        end = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        
        start = generator._calculate_period_start(ReportFrequency.DAILY, end)
        
        assert (end - start).days == 1
    
    def test_calculate_period_start_weekly(
        self,
        mock_analytics_service: MagicMock,
    ) -> None:
        """Test period calculation for weekly frequency."""
        generator = PersonaReportGenerator(mock_analytics_service)
        end = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        
        start = generator._calculate_period_start(ReportFrequency.WEEKLY, end)
        
        assert (end - start).days == 7
    
    def test_calculate_period_start_monthly(
        self,
        mock_analytics_service: MagicMock,
    ) -> None:
        """Test period calculation for monthly frequency."""
        generator = PersonaReportGenerator(mock_analytics_service)
        end = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        
        start = generator._calculate_period_start(ReportFrequency.MONTHLY, end)
        
        assert (end - start).days == 30
    
    def test_get_due_reports_none_registered(
        self,
        mock_analytics_service: MagicMock,
    ) -> None:
        """Test getting due reports when none are registered."""
        generator = PersonaReportGenerator(mock_analytics_service)
        
        due = generator.get_due_reports()
        
        assert len(due) == 0
    
    def test_get_due_reports_never_run(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
    ) -> None:
        """Test that reports with no last_run are due."""
        generator = PersonaReportGenerator(mock_analytics_service)
        generator.register_report(report_config)
        
        due = generator.get_due_reports()
        
        assert len(due) == 1
        assert due[0] == report_config
    
    def test_get_due_reports_daily_overdue(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
    ) -> None:
        """Test that daily reports become due after 24 hours."""
        report_config.frequency = ReportFrequency.DAILY
        report_config.last_run = datetime.now(timezone.utc) - timedelta(hours=25)
        
        generator = PersonaReportGenerator(mock_analytics_service)
        generator.register_report(report_config)
        
        due = generator.get_due_reports()
        
        assert len(due) == 1
    
    def test_get_due_reports_daily_not_due(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
    ) -> None:
        """Test that daily reports are not due if run recently."""
        report_config.frequency = ReportFrequency.DAILY
        report_config.last_run = datetime.now(timezone.utc) - timedelta(hours=12)
        
        generator = PersonaReportGenerator(mock_analytics_service)
        generator.register_report(report_config)
        
        due = generator.get_due_reports()
        
        assert len(due) == 0
    
    def test_get_due_reports_disabled(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
    ) -> None:
        """Test that disabled reports are not due."""
        report_config.enabled = False
        
        generator = PersonaReportGenerator(mock_analytics_service)
        generator.register_report(report_config)
        
        due = generator.get_due_reports()
        
        assert len(due) == 0
    
    def test_get_due_reports_on_demand_excluded(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
    ) -> None:
        """Test that on-demand reports are never automatically due."""
        report_config.frequency = ReportFrequency.ON_DEMAND
        
        generator = PersonaReportGenerator(mock_analytics_service)
        generator.register_report(report_config)
        
        due = generator.get_due_reports()
        
        assert len(due) == 0
    
    @pytest.mark.asyncio
    async def test_run_due_reports(
        self,
        mock_analytics_service: MagicMock,
        report_config: ReportConfig,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test running all due reports."""
        mock_analytics_service.get_metrics.return_value = sample_metrics
        mock_analytics_service.compare_personas.return_value = {}
        mock_analytics_service.identify_improvement_areas.return_value = []
        
        generator = PersonaReportGenerator(mock_analytics_service)
        generator.register_report(report_config)
        
        results = await generator.run_due_reports()
        
        assert len(results) == 1
        assert results[0].success
    
    def test_cleanup_old_reports(
        self,
        mock_analytics_service: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test cleaning up old reports."""
        output_dir = tmp_path / "reports"
        output_dir.mkdir()
        
        # Create an old file
        old_file = output_dir / "old_report.json"
        old_file.write_text("{}")
        
        # Set modification time to 100 days ago
        import os
        old_time = datetime.now().timestamp() - (100 * 24 * 60 * 60)
        os.utime(old_file, (old_time, old_time))
        
        # Create a new file
        new_file = output_dir / "new_report.json"
        new_file.write_text("{}")
        
        config = ReportConfig(
            report_id="test",
            name="Test",
            persona_ids=[],
            frequency=ReportFrequency.DAILY,
            formats=[ReportFormat.JSON],
            output_directory=output_dir,
            retention_days=90,
        )
        
        generator = PersonaReportGenerator(mock_analytics_service)
        generator.register_report(config)
        
        removed = generator.cleanup_old_reports()
        
        assert removed == 1
        assert not old_file.exists()
        assert new_file.exists()


class TestReportConfig:
    """Tests for ReportConfig dataclass."""
    
    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values are set correctly."""
        config = ReportConfig(
            report_id="test",
            name="Test Report",
            persona_ids=["p1"],
            frequency=ReportFrequency.WEEKLY,
            formats=[ReportFormat.JSON],
            output_directory=tmp_path,
        )
        
        assert config.include_comparisons is True
        assert config.include_anomalies is True
        assert config.include_improvements is True
        assert config.include_variants is False
        assert config.recipients == []
        assert config.retention_days == 90
        assert config.enabled is True
        assert config.last_run is None


class TestReportFormat:
    """Tests for ReportFormat enum."""
    
    def test_format_values(self) -> None:
        """Test format enum values."""
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.CSV.value == "csv"
        assert ReportFormat.MARKDOWN.value == "markdown"
        assert ReportFormat.HTML.value == "html"


class TestReportFrequency:
    """Tests for ReportFrequency enum."""
    
    def test_frequency_values(self) -> None:
        """Test frequency enum values."""
        assert ReportFrequency.DAILY.value == "daily"
        assert ReportFrequency.WEEKLY.value == "weekly"
        assert ReportFrequency.MONTHLY.value == "monthly"
        assert ReportFrequency.ON_DEMAND.value == "on_demand"
