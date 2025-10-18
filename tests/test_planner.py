import pytest

from modules.orchestration.planner import PlanStepStatus, Planner


def test_planner_produces_topological_order():
    metadata = {
        "name": "Example",
        "required_tools": ["alpha", "beta", "gamma"],
        "plan": {
            "steps": [
                {"id": "stage_alpha", "tool": "alpha"},
                {"id": "stage_beta", "tool": "beta", "after": ["stage_alpha"]},
                {"id": "stage_gamma", "tool": "gamma", "after": ["stage_beta"]},
            ]
        },
    }

    planner = Planner()
    plan = planner.build_plan(
        metadata,
        available_tools={"alpha": object(), "beta": object(), "gamma": object()},
    )

    assert plan.topological_order() == ["stage_alpha", "stage_beta", "stage_gamma"]
    assert plan.snapshot()["nodes"]


def test_plan_failure_propagates_cancellations():
    metadata = {
        "name": "FailurePropagation",
        "required_tools": ["alpha", "beta", "gamma"],
        "plan": {
            "steps": [
                {"id": "alpha_step", "tool": "alpha"},
                {"id": "beta_step", "tool": "beta", "after": ["alpha_step"]},
                {"id": "gamma_step", "tool": "gamma", "after": ["beta_step"]},
            ]
        },
    }

    planner = Planner()
    plan = planner.build_plan(
        metadata,
        available_tools={"alpha": object(), "beta": object(), "gamma": object()},
    )

    cancelled = plan.mark_failed("alpha_step", "boom")

    assert plan.status("alpha_step") is PlanStepStatus.FAILED
    assert plan.status("beta_step") is PlanStepStatus.CANCELLED
    assert plan.status("gamma_step") is PlanStepStatus.CANCELLED
    assert cancelled == [
        ("beta_step", "Blocked by 'alpha_step' failure: boom"),
        ("gamma_step", "Blocked by 'alpha_step' failure: boom"),
    ]
    assert plan.cancellation_reason("beta_step") == "Blocked by 'alpha_step' failure: boom"


def test_plan_rejects_unknown_dependency():
    metadata = {
        "name": "Invalid",
        "required_tools": ["alpha"],
        "plan": {
            "steps": [
                {"id": "alpha_step", "tool": "alpha", "after": ["missing"]},
            ]
        },
    }

    planner = Planner()

    with pytest.raises(ValueError):
        planner.build_plan(metadata, available_tools={"alpha": object()})
