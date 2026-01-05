# Budget Module Agent Guidelines

## Ownership

- **Backend Agent**: Primary owner of all budget module logic including `manager.py`, `tracking.py`, `pricing.py`, `alerts.py`, `reports.py`, and `api.py`.
- **Data/DB Agent**: Owns `persistence.py` and any database migrations related to budget storage.
- **Infra/Config Agent**: Owns budget-related configuration in `ATLAS/config/budget.py` and `config.yaml` budget sections.

## File-scope boundaries

| File | Owner | Description |
| ---- | ----- | ----------- |
| `models.py` | Backend Agent | Data models, enums, validation |
| `manager.py` | Backend Agent | BudgetManager singleton |
| `pricing.py` | Backend Agent | PricingRegistry and cost calculation |
| `tracking.py` | Backend Agent | Usage tracking event handlers |
| `alerts.py` | Backend Agent | Alert engine and notifications |
| `reports.py` | Backend Agent | Report generation and aggregation |
| `persistence.py` | Data/DB Agent | Storage adapters for budget data |
| `api.py` | Backend Agent | Public API surface |

## Integration points

This module integrates with:

- `ATLAS/provider_manager.py` - LLM usage tracking
- `modules/Providers/Media/manager.py` - Image generation cost tracking
- `ATLAS/config/` - Budget configuration
- `modules/storage/manager.py` - Persistence registration
- `ATLAS/messaging/` - Alert notifications via MessageBus

## Safety & validation rules

- Never store or log API keys or credentials
- Validate all monetary values use Decimal for precision
- Ensure thread-safe access to usage records
- Run `pytest tests/budget/` before submitting PRs
- Follow existing singleton patterns from `StorageManager` and `MediaProviderManager`
