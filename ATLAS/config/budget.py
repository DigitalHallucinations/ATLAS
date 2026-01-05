"""Budget configuration section for ConfigManager.

Provides configuration settings for budget management including:
- Default policies and limits
- Alert thresholds
- Tracking options
- Persistence settings
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    pass

logger = setup_logger(__name__)


# Default configuration values
DEFAULT_BUDGET_CONFIG = {
    # Feature flags
    "enabled": True,
    "tracking_enabled": True,
    "alerts_enabled": True,
    "cost_estimation_enabled": True,
    
    # Default limits (in USD)
    "default_global_monthly_limit": 100.00,
    "default_user_monthly_limit": 25.00,
    "default_user_daily_limit": 5.00,
    
    # Alert thresholds (percentages)
    "alert_thresholds": [50, 75, 90, 100],
    "soft_limit_percent": 80,
    
    # Tracking settings
    "usage_buffer_size": 100,
    "usage_flush_interval_seconds": 60,
    "spend_cache_ttl_seconds": 30,
    
    # Persistence settings
    "persistence_backend": "memory",  # "memory", "sql", "mongodb"
    "retention_days": 90,
    
    # Behavior settings
    "block_on_limit": False,  # If true, blocks requests when limit reached
    "warn_on_approaching_limit": True,
    "show_cost_in_ui": True,
    
    # Anomaly detection
    "anomaly_detection_enabled": True,
    "anomaly_multiplier": 3.0,  # Flag if cost > 3x daily average
}


class BudgetConfigSection:
    """Configuration section for budget management.
    
    Provides typed accessors for budget-related configuration
    and ensures sensible defaults.
    
    Usage::
    
        config = ConfigManager()
        budget_cfg = config.budget
        
        if budget_cfg.is_enabled:
            limit = budget_cfg.default_global_monthly_limit
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        yaml_config: Dict[str, Any],
        env_config: Dict[str, Any],
        logger: Any,
        write_yaml_callback: Optional[Callable[[str, Any], None]] = None,
    ):
        """Initialize the budget configuration section.
        
        Args:
            config: Main configuration dictionary.
            yaml_config: YAML-sourced configuration.
            env_config: Environment-sourced configuration.
            logger: Logger instance.
            write_yaml_callback: Callback to persist YAML changes.
        """
        self._config = config
        self._yaml_config = yaml_config
        self._env_config = env_config
        self._logger = logger
        self._write_yaml = write_yaml_callback
        
        # Apply defaults
        self._apply_defaults()
    
    def _apply_defaults(self) -> None:
        """Apply default values for missing configuration."""
        budget_section = self._yaml_config.setdefault("budget", {})
        
        for key, default_value in DEFAULT_BUDGET_CONFIG.items():
            if key not in budget_section:
                budget_section[key] = default_value
        
        # Also update main config
        self._config.setdefault("budget", {})
        self._config["budget"].update(budget_section)
    
    def _get(self, key: str, default: Any = None) -> Any:
        """Get a budget configuration value.
        
        Args:
            key: Configuration key.
            default: Default value if not found.
            
        Returns:
            Configuration value.
        """
        # Check environment first
        env_key = f"ATLAS_BUDGET_{key.upper()}"
        env_value = self._env_config.get(env_key)
        if env_value is not None:
            return env_value
        
        # Check YAML config
        budget_section = self._yaml_config.get("budget", {})
        if key in budget_section:
            return budget_section[key]
        
        # Check main config
        main_budget = self._config.get("budget", {})
        if key in main_budget:
            return main_budget[key]
        
        return default if default is not None else DEFAULT_BUDGET_CONFIG.get(key)
    
    def _set(self, key: str, value: Any) -> None:
        """Set a budget configuration value.
        
        Args:
            key: Configuration key.
            value: Value to set.
        """
        budget_section = self._yaml_config.setdefault("budget", {})
        budget_section[key] = value
        
        main_budget = self._config.setdefault("budget", {})
        main_budget[key] = value
        
        if self._write_yaml:
            self._write_yaml("budget", budget_section)
    
    # =========================================================================
    # Feature Flags
    # =========================================================================
    
    @property
    def is_enabled(self) -> bool:
        """Whether budget management is enabled."""
        return bool(self._get("enabled", True))
    
    @is_enabled.setter
    def is_enabled(self, value: bool) -> None:
        self._set("enabled", bool(value))
    
    @property
    def tracking_enabled(self) -> bool:
        """Whether usage tracking is enabled."""
        return bool(self._get("tracking_enabled", True))
    
    @tracking_enabled.setter
    def tracking_enabled(self, value: bool) -> None:
        self._set("tracking_enabled", bool(value))
    
    @property
    def alerts_enabled(self) -> bool:
        """Whether budget alerts are enabled."""
        return bool(self._get("alerts_enabled", True))
    
    @alerts_enabled.setter
    def alerts_enabled(self, value: bool) -> None:
        self._set("alerts_enabled", bool(value))
    
    @property
    def cost_estimation_enabled(self) -> bool:
        """Whether to estimate costs before requests."""
        return bool(self._get("cost_estimation_enabled", True))
    
    # =========================================================================
    # Default Limits
    # =========================================================================
    
    @property
    def default_global_monthly_limit(self) -> Decimal:
        """Default global monthly budget limit in USD."""
        value = self._get("default_global_monthly_limit", 100.00)
        return Decimal(str(value))
    
    @default_global_monthly_limit.setter
    def default_global_monthly_limit(self, value: Decimal) -> None:
        self._set("default_global_monthly_limit", float(value))
    
    @property
    def default_user_monthly_limit(self) -> Decimal:
        """Default per-user monthly budget limit in USD."""
        value = self._get("default_user_monthly_limit", 25.00)
        return Decimal(str(value))
    
    @default_user_monthly_limit.setter
    def default_user_monthly_limit(self, value: Decimal) -> None:
        self._set("default_user_monthly_limit", float(value))
    
    @property
    def default_user_daily_limit(self) -> Decimal:
        """Default per-user daily budget limit in USD."""
        value = self._get("default_user_daily_limit", 5.00)
        return Decimal(str(value))
    
    @default_user_daily_limit.setter
    def default_user_daily_limit(self, value: Decimal) -> None:
        self._set("default_user_daily_limit", float(value))
    
    # =========================================================================
    # Alert Settings
    # =========================================================================
    
    @property
    def alert_thresholds(self) -> List[int]:
        """Alert threshold percentages."""
        thresholds = self._get("alert_thresholds", [50, 75, 90, 100])
        if isinstance(thresholds, list):
            return [int(t) for t in thresholds]
        return [50, 75, 90, 100]
    
    @alert_thresholds.setter
    def alert_thresholds(self, value: List[int]) -> None:
        self._set("alert_thresholds", list(value))
    
    @property
    def soft_limit_percent(self) -> float:
        """Percentage at which soft limit warning triggers."""
        return float(self._get("soft_limit_percent", 80))
    
    @soft_limit_percent.setter
    def soft_limit_percent(self, value: float) -> None:
        self._set("soft_limit_percent", float(value))
    
    # =========================================================================
    # Tracking Settings
    # =========================================================================
    
    @property
    def usage_buffer_size(self) -> int:
        """Number of usage records to buffer before flushing."""
        return int(self._get("usage_buffer_size", 100))
    
    @property
    def usage_flush_interval_seconds(self) -> int:
        """Interval between usage buffer flushes."""
        return int(self._get("usage_flush_interval_seconds", 60))
    
    @property
    def spend_cache_ttl_seconds(self) -> int:
        """TTL for cached spend calculations."""
        return int(self._get("spend_cache_ttl_seconds", 30))
    
    # =========================================================================
    # Persistence Settings
    # =========================================================================
    
    @property
    def persistence_backend(self) -> str:
        """Storage backend: 'memory', 'sql', or 'mongodb'."""
        return str(self._get("persistence_backend", "memory"))
    
    @persistence_backend.setter
    def persistence_backend(self, value: str) -> None:
        if value not in ("memory", "sql", "mongodb"):
            raise ValueError(f"Invalid persistence backend: {value}")
        self._set("persistence_backend", value)
    
    @property
    def retention_days(self) -> int:
        """Number of days to retain usage records."""
        return int(self._get("retention_days", 90))
    
    @retention_days.setter
    def retention_days(self, value: int) -> None:
        self._set("retention_days", max(1, int(value)))
    
    # =========================================================================
    # Behavior Settings
    # =========================================================================
    
    @property
    def block_on_limit(self) -> bool:
        """Whether to block requests when limit is reached."""
        return bool(self._get("block_on_limit", False))
    
    @block_on_limit.setter
    def block_on_limit(self, value: bool) -> None:
        self._set("block_on_limit", bool(value))
    
    @property
    def warn_on_approaching_limit(self) -> bool:
        """Whether to warn when approaching limit."""
        return bool(self._get("warn_on_approaching_limit", True))
    
    @property
    def show_cost_in_ui(self) -> bool:
        """Whether to show cost information in the UI."""
        return bool(self._get("show_cost_in_ui", True))
    
    @show_cost_in_ui.setter
    def show_cost_in_ui(self, value: bool) -> None:
        self._set("show_cost_in_ui", bool(value))
    
    # =========================================================================
    # Anomaly Detection
    # =========================================================================
    
    @property
    def anomaly_detection_enabled(self) -> bool:
        """Whether anomaly detection is enabled."""
        return bool(self._get("anomaly_detection_enabled", True))
    
    @property
    def anomaly_multiplier(self) -> float:
        """Multiplier for flagging anomalous costs."""
        return float(self._get("anomaly_multiplier", 3.0))
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def as_dict(self) -> Dict[str, Any]:
        """Return all budget settings as a dictionary."""
        return {
            "enabled": self.is_enabled,
            "tracking_enabled": self.tracking_enabled,
            "alerts_enabled": self.alerts_enabled,
            "cost_estimation_enabled": self.cost_estimation_enabled,
            "default_global_monthly_limit": float(self.default_global_monthly_limit),
            "default_user_monthly_limit": float(self.default_user_monthly_limit),
            "default_user_daily_limit": float(self.default_user_daily_limit),
            "alert_thresholds": self.alert_thresholds,
            "soft_limit_percent": self.soft_limit_percent,
            "usage_buffer_size": self.usage_buffer_size,
            "usage_flush_interval_seconds": self.usage_flush_interval_seconds,
            "spend_cache_ttl_seconds": self.spend_cache_ttl_seconds,
            "persistence_backend": self.persistence_backend,
            "retention_days": self.retention_days,
            "block_on_limit": self.block_on_limit,
            "warn_on_approaching_limit": self.warn_on_approaching_limit,
            "show_cost_in_ui": self.show_cost_in_ui,
            "anomaly_detection_enabled": self.anomaly_detection_enabled,
            "anomaly_multiplier": self.anomaly_multiplier,
        }
