# Budget Manager UI Agent Guidelines

## Scope

This directory contains GTK UI components for budget management visualization and control.

## Files

- `__init__.py` - Module exports
- `budget_management.py` - Main controller and workspace coordinator
- `dashboard.py` - Primary budget dashboard with spending overview
- `policy_editor.py` - Budget policy creation and editing dialogs
- `usage_history.py` - Historical usage browsing and filtering
- `reports_view.py` - Usage reports and analytics display
- `alerts_panel.py` - Alert management and acknowledgment

## Patterns

- Follow existing GTKUI manager patterns (Task_manager, Job_manager)
- Use `get_embeddable_widget()` pattern for sidebar integration
- Subscribe to MessageBus channels for real-time updates
- Maintain separation from backend logic in `modules/budget/`

## Required Checks

- Run `python3 main.py` to verify GTK shell starts correctly
- Confirm keyboard navigation works for accessibility
- Test at 1366×768 resolution minimum
