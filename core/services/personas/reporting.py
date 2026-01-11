"""
Persona analytics automated reporting.

Provides scheduled report generation and export for persona
performance analytics.

Author: ATLAS Team
Date: Jan 11, 2026
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from core.services.personas import (
        PersonaAnalyticsService,
        PersonaPerformanceMetrics,
    )
    from core.services.common import Actor
    from core.config import ConfigManager

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report output formats."""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"


class ReportFrequency(Enum):
    """Report generation frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_DEMAND = "on_demand"


@dataclass
class ReportConfig:
    """Configuration for automated report generation."""
    report_id: str
    name: str
    persona_ids: List[str]  # Empty means all personas
    frequency: ReportFrequency
    formats: List[ReportFormat]
    output_directory: Path
    include_comparisons: bool = True
    include_anomalies: bool = True
    include_improvements: bool = True
    include_variants: bool = False
    recipients: List[str] = field(default_factory=list)  # Email recipients
    retention_days: int = 90
    enabled: bool = True
    last_run: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ReportResult:
    """Result of a report generation."""
    report_id: str
    generated_at: datetime
    formats_generated: List[ReportFormat]
    output_files: List[Path]
    persona_count: int
    period_start: datetime
    period_end: datetime
    success: bool
    error: Optional[str] = None


class PersonaReportGenerator:
    """
    Generates analytics reports for personas.
    
    Supports multiple output formats and can be scheduled
    for automated periodic generation.
    """
    
    def __init__(
        self,
        analytics_service: "PersonaAnalyticsService",
        config_manager: Optional["ConfigManager"] = None,
    ) -> None:
        """
        Initialize the report generator.
        
        Args:
            analytics_service: The analytics service for data
            config_manager: Optional config manager for settings
        """
        self._service = analytics_service
        self._config = config_manager
        self._report_configs: Dict[str, ReportConfig] = {}
        self._generators: Dict[ReportFormat, Callable[..., str]] = {
            ReportFormat.JSON: self._generate_json,
            ReportFormat.CSV: self._generate_csv,
            ReportFormat.MARKDOWN: self._generate_markdown,
            ReportFormat.HTML: self._generate_html,
        }
    
    # =========================================================================
    # Report Configuration
    # =========================================================================
    
    def register_report(self, config: ReportConfig) -> None:
        """Register a report configuration."""
        self._report_configs[config.report_id] = config
        logger.info("Registered report: %s (%s)", config.name, config.report_id)
    
    def unregister_report(self, report_id: str) -> bool:
        """Unregister a report configuration."""
        if report_id in self._report_configs:
            del self._report_configs[report_id]
            logger.info("Unregistered report: %s", report_id)
            return True
        return False
    
    def get_report_configs(self) -> List[ReportConfig]:
        """Get all registered report configurations."""
        return list(self._report_configs.values())
    
    def get_report_config(self, report_id: str) -> Optional[ReportConfig]:
        """Get a specific report configuration."""
        return self._report_configs.get(report_id)
    
    # =========================================================================
    # Report Generation
    # =========================================================================
    
    async def generate_report(
        self,
        config: ReportConfig,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> ReportResult:
        """
        Generate a report based on configuration.
        
        Args:
            config: The report configuration
            start: Override start of reporting period
            end: Override end of reporting period
            
        Returns:
            ReportResult with generation status and output files
        """
        now = datetime.now(timezone.utc)
        
        # Determine period based on frequency if not specified
        if end is None:
            end = now
        if start is None:
            start = self._calculate_period_start(config.frequency, end)
        
        try:
            # Collect data for all personas
            data = await self._collect_report_data(config, start, end)
            
            # Generate reports in each format
            output_files: List[Path] = []
            for fmt in config.formats:
                generator = self._generators.get(fmt)
                if generator:
                    content = generator(data, config, start, end)
                    file_path = self._write_report(
                        content, fmt, config, start, end
                    )
                    output_files.append(file_path)
            
            # Update last run time
            config.last_run = now
            
            return ReportResult(
                report_id=config.report_id,
                generated_at=now,
                formats_generated=config.formats,
                output_files=output_files,
                persona_count=len(data.get("personas", {})),
                period_start=start,
                period_end=end,
                success=True,
            )
            
        except Exception as e:
            logger.error("Report generation failed: %s", e)
            return ReportResult(
                report_id=config.report_id,
                generated_at=now,
                formats_generated=[],
                output_files=[],
                persona_count=0,
                period_start=start,
                period_end=end,
                success=False,
                error=str(e),
            )
    
    async def generate_on_demand_report(
        self,
        persona_ids: List[str],
        formats: List[ReportFormat],
        output_directory: Path,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        name: str = "On-Demand Report",
    ) -> ReportResult:
        """
        Generate an on-demand report without pre-configuration.
        
        Args:
            persona_ids: Personas to include (empty for all)
            formats: Output formats to generate
            output_directory: Where to save reports
            start: Start of period
            end: End of period
            name: Report name
            
        Returns:
            ReportResult with generation status
        """
        config = ReportConfig(
            report_id=f"on_demand_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            name=name,
            persona_ids=persona_ids,
            frequency=ReportFrequency.ON_DEMAND,
            formats=formats,
            output_directory=output_directory,
        )
        return await self.generate_report(config, start=start, end=end)
    
    async def _collect_report_data(
        self,
        config: ReportConfig,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Any]:
        """Collect all data needed for the report."""
        persona_ids = config.persona_ids
        
        # If no specific personas, we'd need to get all
        # For now, use the provided list
        if not persona_ids:
            # Would query persona service for all persona IDs
            persona_ids = []
        
        data: Dict[str, Any] = {
            "report_name": config.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "personas": {},
            "comparison": None,
            "anomalies": [],
            "improvements": {},
        }
        
        # Collect metrics for each persona
        for persona_id in persona_ids:
            try:
                metrics = await self._service.get_metrics(persona_id, start, end)
                if metrics:
                    data["personas"][persona_id] = self._metrics_to_dict(metrics)
            except Exception as e:
                logger.warning("Failed to get metrics for %s: %s", persona_id, e)
                data["personas"][persona_id] = {"error": str(e)}
        
        # Collect comparison data
        if config.include_comparisons and len(persona_ids) > 1:
            try:
                comparison = await self._service.compare_personas(
                    persona_ids, start, end
                )
                data["comparison"] = {
                    pid: self._metrics_to_dict(m)
                    for pid, m in comparison.items()
                }
            except Exception as e:
                logger.warning("Failed to get comparison: %s", e)
        
        # Collect anomalies
        if config.include_anomalies:
            for persona_id in persona_ids:
                try:
                    # Anomalies come from the metrics
                    pass  # Would extract from interaction log
                except Exception:
                    pass
        
        # Collect improvements
        if config.include_improvements:
            for persona_id in persona_ids:
                try:
                    areas = await self._service.identify_improvement_areas(persona_id)
                    data["improvements"][persona_id] = [
                        {
                            "area": a.area,
                            "priority": a.priority,
                            "suggestions": list(a.suggestions or []),
                        }
                        for a in areas
                    ]
                except Exception as e:
                    logger.warning("Failed to get improvements for %s: %s", persona_id, e)
        
        return data
    
    def _metrics_to_dict(self, metrics: "PersonaPerformanceMetrics") -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "persona_id": metrics.persona_id,
            "total_interactions": metrics.total_interactions,
            "avg_response_time_ms": metrics.avg_response_time_ms,
            "task_success_rate": metrics.task_success_rate,
            "escalation_rate": metrics.escalation_rate,
            "retry_rate": metrics.retry_rate,
            "token_usage": {
                "prompt_tokens": metrics.token_usage.prompt_tokens if metrics.token_usage else 0,
                "completion_tokens": metrics.token_usage.completion_tokens if metrics.token_usage else 0,
                "total_tokens": metrics.token_usage.total_tokens if metrics.token_usage else 0,
            },
            "tools_used": dict(metrics.tools_used or {}),
            "skills_invoked": dict(metrics.skills_invoked or {}),
            "capability_gaps": list(metrics.capability_gaps or []),
        }
    
    def _calculate_period_start(
        self,
        frequency: ReportFrequency,
        end: datetime,
    ) -> datetime:
        """Calculate period start based on frequency."""
        if frequency == ReportFrequency.DAILY:
            return end - timedelta(days=1)
        elif frequency == ReportFrequency.WEEKLY:
            return end - timedelta(weeks=1)
        elif frequency == ReportFrequency.MONTHLY:
            return end - timedelta(days=30)
        else:  # ON_DEMAND defaults to 7 days
            return end - timedelta(days=7)
    
    def _write_report(
        self,
        content: str,
        fmt: ReportFormat,
        config: ReportConfig,
        start: datetime,
        end: datetime,
    ) -> Path:
        """Write report content to file."""
        config.output_directory.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        extension = fmt.value
        filename = f"{config.report_id}_{timestamp}.{extension}"
        
        file_path = config.output_directory / filename
        file_path.write_text(content, encoding="utf-8")
        
        logger.info("Wrote report: %s", file_path)
        return file_path
    
    # =========================================================================
    # Format Generators
    # =========================================================================
    
    def _generate_json(
        self,
        data: Dict[str, Any],
        config: ReportConfig,
        start: datetime,
        end: datetime,
    ) -> str:
        """Generate JSON format report."""
        return json.dumps(data, indent=2, default=str)
    
    def _generate_csv(
        self,
        data: Dict[str, Any],
        config: ReportConfig,
        start: datetime,
        end: datetime,
    ) -> str:
        """Generate CSV format report."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Persona ID",
            "Total Interactions",
            "Avg Response Time (ms)",
            "Success Rate",
            "Escalation Rate",
            "Retry Rate",
            "Prompt Tokens",
            "Completion Tokens",
            "Total Tokens",
        ])
        
        # Data rows
        for persona_id, metrics in data.get("personas", {}).items():
            if isinstance(metrics, dict) and "error" not in metrics:
                token_usage = metrics.get("token_usage", {})
                writer.writerow([
                    persona_id,
                    metrics.get("total_interactions", 0),
                    f"{metrics.get('avg_response_time_ms', 0):.2f}",
                    f"{metrics.get('task_success_rate', 0):.2%}",
                    f"{metrics.get('escalation_rate', 0):.2%}",
                    f"{metrics.get('retry_rate', 0):.2%}",
                    token_usage.get("prompt_tokens", 0),
                    token_usage.get("completion_tokens", 0),
                    token_usage.get("total_tokens", 0),
                ])
        
        return output.getvalue()
    
    def _generate_markdown(
        self,
        data: Dict[str, Any],
        config: ReportConfig,
        start: datetime,
        end: datetime,
    ) -> str:
        """Generate Markdown format report."""
        lines: List[str] = []
        
        lines.append(f"# {data.get('report_name', 'Persona Analytics Report')}")
        lines.append("")
        lines.append(f"**Generated:** {data.get('generated_at', 'N/A')}")
        period = data.get("period", {})
        lines.append(f"**Period:** {period.get('start', 'N/A')} to {period.get('end', 'N/A')}")
        lines.append("")
        
        # Summary table
        lines.append("## Performance Summary")
        lines.append("")
        lines.append("| Persona | Interactions | Success Rate | Avg Latency | Escalations |")
        lines.append("|---------|-------------|--------------|-------------|-------------|")
        
        for persona_id, metrics in data.get("personas", {}).items():
            if isinstance(metrics, dict) and "error" not in metrics:
                lines.append(
                    f"| {persona_id} "
                    f"| {metrics.get('total_interactions', 0)} "
                    f"| {metrics.get('task_success_rate', 0):.1%} "
                    f"| {metrics.get('avg_response_time_ms', 0):.0f}ms "
                    f"| {metrics.get('escalation_rate', 0):.1%} |"
                )
        
        lines.append("")
        
        # Improvement areas
        if data.get("improvements"):
            lines.append("## Improvement Recommendations")
            lines.append("")
            for persona_id, areas in data["improvements"].items():
                if areas:
                    lines.append(f"### {persona_id}")
                    lines.append("")
                    for area in areas:
                        lines.append(f"- **{area.get('area', 'N/A')}** (Priority: {area.get('priority', 'N/A')})")
                        for suggestion in area.get("suggestions", []):
                            lines.append(f"  - {suggestion}")
                    lines.append("")
        
        return "\n".join(lines)
    
    def _generate_html(
        self,
        data: Dict[str, Any],
        config: ReportConfig,
        start: datetime,
        end: datetime,
    ) -> str:
        """Generate HTML format report."""
        period = data.get("period", {})
        
        html_parts: List[str] = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='UTF-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"  <title>{data.get('report_name', 'Persona Analytics Report')}</title>",
            "  <style>",
            "    body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
            "    table { width: 100%; border-collapse: collapse; margin: 20px 0; }",
            "    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
            "    th { background-color: #f4f4f4; }",
            "    tr:nth-child(even) { background-color: #f9f9f9; }",
            "    .success { color: #28a745; }",
            "    .warning { color: #ffc107; }",
            "    .danger { color: #dc3545; }",
            "    .metric-card { display: inline-block; padding: 20px; margin: 10px; background: #f5f5f5; border-radius: 8px; }",
            "  </style>",
            "</head>",
            "<body>",
            f"  <h1>{data.get('report_name', 'Persona Analytics Report')}</h1>",
            f"  <p><strong>Generated:</strong> {data.get('generated_at', 'N/A')}</p>",
            f"  <p><strong>Period:</strong> {period.get('start', 'N/A')} to {period.get('end', 'N/A')}</p>",
            "",
            "  <h2>Performance Summary</h2>",
            "  <table>",
            "    <thead>",
            "      <tr>",
            "        <th>Persona</th>",
            "        <th>Interactions</th>",
            "        <th>Success Rate</th>",
            "        <th>Avg Latency</th>",
            "        <th>Escalation Rate</th>",
            "        <th>Total Tokens</th>",
            "      </tr>",
            "    </thead>",
            "    <tbody>",
        ]
        
        for persona_id, metrics in data.get("personas", {}).items():
            if isinstance(metrics, dict) and "error" not in metrics:
                success_rate = metrics.get("task_success_rate", 0)
                success_class = "success" if success_rate >= 0.8 else ("warning" if success_rate >= 0.6 else "danger")
                token_usage = metrics.get("token_usage", {})
                
                html_parts.extend([
                    "      <tr>",
                    f"        <td>{persona_id}</td>",
                    f"        <td>{metrics.get('total_interactions', 0)}</td>",
                    f"        <td class='{success_class}'>{success_rate:.1%}</td>",
                    f"        <td>{metrics.get('avg_response_time_ms', 0):.0f}ms</td>",
                    f"        <td>{metrics.get('escalation_rate', 0):.1%}</td>",
                    f"        <td>{token_usage.get('total_tokens', 0):,}</td>",
                    "      </tr>",
                ])
        
        html_parts.extend([
            "    </tbody>",
            "  </table>",
            "</body>",
            "</html>",
        ])
        
        return "\n".join(html_parts)
    
    # =========================================================================
    # Scheduled Execution
    # =========================================================================
    
    def get_due_reports(self) -> List[ReportConfig]:
        """Get reports that are due for generation."""
        now = datetime.now(timezone.utc)
        due: List[ReportConfig] = []
        
        for config in self._report_configs.values():
            if not config.enabled:
                continue
            
            if config.frequency == ReportFrequency.ON_DEMAND:
                continue
            
            if config.last_run is None:
                due.append(config)
                continue
            
            elapsed = now - config.last_run
            
            if config.frequency == ReportFrequency.DAILY and elapsed >= timedelta(days=1):
                due.append(config)
            elif config.frequency == ReportFrequency.WEEKLY and elapsed >= timedelta(weeks=1):
                due.append(config)
            elif config.frequency == ReportFrequency.MONTHLY and elapsed >= timedelta(days=30):
                due.append(config)
        
        return due
    
    async def run_due_reports(self) -> List[ReportResult]:
        """Run all due reports and return results."""
        due_reports = self.get_due_reports()
        results: List[ReportResult] = []
        
        for config in due_reports:
            result = await self.generate_report(config)
            results.append(result)
        
        return results
    
    def cleanup_old_reports(self) -> int:
        """Remove reports older than retention period. Returns count removed."""
        removed = 0
        now = datetime.now(timezone.utc)
        
        for config in self._report_configs.values():
            if not config.output_directory.exists():
                continue
            
            cutoff = now - timedelta(days=config.retention_days)
            
            for file_path in config.output_directory.iterdir():
                if file_path.is_file():
                    try:
                        mtime = datetime.fromtimestamp(
                            file_path.stat().st_mtime, timezone.utc
                        )
                        if mtime < cutoff:
                            file_path.unlink()
                            removed += 1
                            logger.info("Removed old report: %s", file_path)
                    except Exception as e:
                        logger.warning("Failed to remove %s: %s", file_path, e)
        
        return removed
