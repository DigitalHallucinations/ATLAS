# Budget Manager

The Budget Manager provides comprehensive cost tracking, budget policy enforcement, and usage reporting for all AI provider interactions in ATLAS.

## Overview

The Budget Manager tracks API usage and costs across all configured AI providers, enabling:

- **Real-time cost tracking** - Monitor spending as it happens
- **Budget policies** - Set spending limits per user, provider, or globally  
- **Alert system** - Get notified when approaching or exceeding limits
- **Usage reports** - Generate detailed cost breakdowns and trends
- **Cost projections** - Forecast future spending based on historical usage

## Quick Start

### Enable Budget Tracking

Budget tracking is enabled by default. To configure it, edit `config.yaml`:

```yaml
budget:
  enabled: true
  flush_interval: 300  # Seconds between database flushes
  cache_ttl: 60        # Cache expiry in seconds
  default_limit: 100.00  # Default monthly limit in USD
  soft_limit_percent: 80  # Warning threshold percentage
```

### Create a Budget Policy

```python
from modules.budget import get_budget_manager
from modules.budget.models import BudgetPolicy, BudgetPeriod, BudgetScope

manager = await get_budget_manager()

policy = BudgetPolicy(
    name="Monthly Global Budget",
    scope=BudgetScope.GLOBAL,
    period=BudgetPeriod.MONTHLY,
    limit_amount=100.00,
    soft_limit_percent=80,
)

await manager.set_policy(policy)
```

### Check Budget Before API Calls

```python
from modules.budget import get_usage_tracker

tracker = await get_usage_tracker()

# Check if budget allows the request
result = await tracker.check_budget(
    model="gpt-4o",
    estimated_tokens=1000,
    user_id="user123",
)

if result.allowed:
    # Proceed with API call
    ...
```

### Record Usage After API Calls

```python
# After a successful API call
await tracker.record_usage(
    model="gpt-4o",
    provider="openai",
    input_tokens=500,
    output_tokens=250,
    user_id="user123",
)
```

## Architecture

### Components

| Component | Purpose |
| --------- | ------- |
| `BudgetManager` | Central coordinator for budget operations |
| `UsageTracker` | Records and aggregates usage data |
| `PricingRegistry` | Maintains current pricing for all models |
| `AlertEngine` | Evaluates thresholds and triggers notifications |
| `ReportGenerator` | Creates usage reports and exports |
| `BudgetStore` | Persistence layer for budget data |

### Data Flow

```Text
API Call → UsageTracker.record_usage() → Cost Calculation → AlertEngine.evaluate()
                    ↓                           ↓
              Usage Buffer              Budget Check
                    ↓                           ↓
              BudgetStore ←──────────── Alert Generation
```

## Budget Policies

### Scopes

| Scope | Description |
| ----- | ----------- |
| `GLOBAL` | Applies to all usage |
| `USER` | Per-user spending limit |
| `PROVIDER` | Per-provider limit (e.g., only $50/month for OpenAI) |
| `MODEL` | Per-model limit (e.g., limit GPT-4 usage) |
| `PROJECT` | Per-project budget allocation |

### Periods

| Period | Description |
| ------ | ----------- |
| `DAILY` | Resets at midnight UTC |
| `WEEKLY` | Resets on Monday at midnight UTC |
| `MONTHLY` | Resets on the 1st at midnight UTC |
| `CUSTOM` | User-defined period with explicit dates |

### Limit Actions

| Action | Behavior |
| ------ | -------- |
| `WARN` | Log warning but allow request |
| `BLOCK` | Reject request when over limit |
| `THROTTLE` | Rate-limit requests near limit |
| `NOTIFY` | Send notification to administrators |

## Pricing Registry

The pricing registry maintains up-to-date pricing for all supported models.

### Supported Providers

- **OpenAI** - GPT-4o, GPT-4o-mini, GPT-4, o1, o1-mini, DALL-E 3, text-embedding-3, Whisper
- **Anthropic** - Claude Sonnet 4, Claude 3.5 Sonnet/Haiku, Claude 3 Opus/Sonnet/Haiku
- **Google** - Gemini 2.5/2.0/1.5 Pro/Flash
- **Mistral** - Mistral Large, Ministral, Pixtral, Codestral
- **Groq** - LLaMA models, Mixtral
- **xAI** - Grok 2/3/3-mini

### Cost Calculation

```python
from modules.budget import get_pricing_registry

registry = await get_pricing_registry()

# Calculate LLM cost
cost = registry.calculate_llm_cost(
    model="gpt-4o",
    input_tokens=1000,
    output_tokens=500,
    cached_tokens=200,  # Prompt caching discount
)

# Calculate image generation cost
cost = registry.calculate_image_cost(
    model="dall-e-3",
    size="1024x1024",
    quality="hd",
    count=2,
)

# Calculate embedding cost
cost = registry.calculate_embedding_cost(
    model="text-embedding-3-small",
    tokens=10000,
)
```

## Alerts

### Alert Types

| Type | Description |
| ---- | ----------- |
| `THRESHOLD_WARNING` | Approaching soft limit (default 80%) |
| `THRESHOLD_CRITICAL` | At or near hard limit (default 95%) |
| `LIMIT_EXCEEDED` | Over budget limit |
| `ANOMALY` | Unusual spending pattern detected |
| `POLICY_VIOLATION` | Request blocked by policy |

### Alert Channels

Alerts can be sent through multiple channels:

- **UI Toast** - In-app notification
- **Desktop Notification** - System notification
- **Log Entry** - Written to application log
- **Email** - (requires email configuration)
- **Webhook** - HTTP POST to configured URL

### Configuring Alert Rules

```python
from modules.budget import get_alert_engine
from modules.budget.alerts import AlertRule, AlertSeverity

engine = await get_alert_engine()

rule = AlertRule(
    name="High Spend Warning",
    threshold_percent=75,
    severity=AlertSeverity.WARNING,
    channels=["ui", "log"],
    cooldown_minutes=60,  # Don't repeat for 1 hour
)

engine.add_rule(rule)
```

## Reports

### Available Reports

| Report | Description |
| ------ | ----------- |
| Summary | Current period totals by provider/model |
| Trend | Usage over time with daily/weekly breakdown |
| Comparison | Period-over-period change analysis |
| Projection | Forecasted spending based on trends |

### Generating Reports

```python
from modules.budget import get_report_generator
from modules.budget.reports import ExportFormat

generator = await get_report_generator()

# Summary report
summary = await generator.generate_summary_report()

# Trend report over 30 days
trend = await generator.generate_trend_report(days=30)

# Export to various formats
json_data = summary.to_json()
csv_data = summary.to_csv()
markdown = summary.to_markdown()
```

## GTK UI

The Budget Manager includes a GTK-based UI accessible from the sidebar.

### Dashboard

The dashboard displays:

- Current period spending vs. budget
- Top spending by provider and model
- Recent usage history
- Active alerts

### Policy Editor

Create and manage budget policies with:

- Scope and period selection
- Limit amount configuration
- Soft limit percentage
- Action on limit exceeded

### Usage History

Browse detailed usage records with:

- Date/time filtering
- Provider/model filtering
- Export capabilities

### Reports View

Generate and view reports:

- Select report type
- Choose date range
- Export to CSV/JSON/Markdown

## API Reference

### BudgetManager

```python
manager = await get_budget_manager()

# Policy management
await manager.set_policy(policy)
policy = await manager.get_policy(policy_id)
policies = await manager.get_policies(scope=BudgetScope.USER)
await manager.delete_policy(policy_id)

# Usage recording
await manager.record_usage(record)

# Budget checking
result = await manager.check_budget(model, tokens, user_id)

# Spending queries
summary = await manager.get_spend_summary(period, scope)
```

### UsageTracker

```python
tracker = await get_usage_tracker()

# Record usage
record = await tracker.record_usage(
    model="gpt-4o",
    provider="openai",
    input_tokens=500,
    output_tokens=250,
    user_id="user123",
)

# Check budget
result = await tracker.check_budget(
    model="gpt-4o",
    estimated_tokens=1000,
    user_id="user123",
)

# Decorator for automatic tracking
@tracker.track_llm_call(model="gpt-4o", user_id="user123")
async def my_llm_function():
    ...
```

### PricingRegistry

```python
registry = await get_pricing_registry()

# Get pricing info
pricing = registry.get_model_pricing("gpt-4o")

# Calculate costs
llm_cost = registry.calculate_llm_cost(model, input_tokens, output_tokens)
image_cost = registry.calculate_image_cost(model, size, quality, count)
embed_cost = registry.calculate_embedding_cost(model, tokens)

# Provider queries
models = registry.get_provider_models("OpenAI")
providers = registry.get_all_providers()
```

### AlertEngine

```python
engine = await get_alert_engine()

# Rule management
rules = engine.get_rules()
engine.add_rule(rule)
engine.remove_rule(rule_id)

# Alert evaluation
alerts = await engine.evaluate_thresholds(spend_summary)

# Alert management
active = engine.get_active_alerts(policy_id=None, unacknowledged_only=False)
engine.acknowledge_alert(alert_id, acknowledged_by)
engine.resolve_alert(alert_id)
```

### ReportGenerator

```python
generator = await get_report_generator()

# Generate reports
report = await generator.generate_report(start_date, end_date, group_by=[])
summary = await generator.generate_summary_report(period)
trend = await generator.generate_trend_report(days, interval)
comparison = await generator.generate_comparison_report(
    current_start, current_end, previous_start, previous_end
)
projection = await generator.generate_projection_report(days_to_project)

# Export
json_str = report.to_json()
csv_str = report.to_csv()
md_str = report.to_markdown()
```

## Configuration Reference

### config.yaml

```yaml
budget:
  # Enable/disable budget tracking
  enabled: true
  
  # How often to flush usage to database (seconds)
  flush_interval: 300
  
  # Cache TTL for spend summaries (seconds)  
  cache_ttl: 60
  
  # Default budget limit for new users (USD)
  default_limit: 100.00
  
  # Percentage for soft limit warnings
  soft_limit_percent: 80
  
  # Default action when limit exceeded
  default_action: warn  # warn, block, throttle, notify

budget_policies:
  # Pre-configured policies (loaded at startup)
  - name: "Global Monthly Budget"
    scope: global
    period: monthly
    limit_amount: 500.00
    soft_limit_percent: 80
    action: warn

alert_channels:
  # Configure notification channels
  ui:
    enabled: true
  log:
    enabled: true
    level: warning
  email:
    enabled: false
    recipients: []
  webhook:
    enabled: false
    url: ""
```

## Best Practices

### Set Appropriate Limits

- Start with conservative limits and adjust based on usage patterns
- Use soft limits (warnings) before hard limits (blocks)
- Set user-level limits for shared deployments

### Monitor Regularly

- Review daily/weekly spending trends
- Set up alerts for unusual activity
- Generate comparison reports to track changes

### Optimize Costs

- Use cheaper models when quality requirements allow
- Enable prompt caching where supported
- Batch requests when possible

## Troubleshooting

### Budget Not Tracking

1. Verify `budget.enabled: true` in config.yaml
2. Check that provider integration hooks are installed
3. Review logs for initialization errors

### Alerts Not Firing

1. Verify alert rules are configured
2. Check that alert channels are enabled
3. Review cooldown settings (may be suppressing alerts)

### Incorrect Costs

1. Verify model names match pricing registry entries
2. Check for custom pricing overrides
3. Review token counting accuracy

## See Also

- [Architecture Overview](architecture-overview.md)
- [Configuration](configuration.md)
- [Provider Manager](docs/developer/provider-manager.md)
