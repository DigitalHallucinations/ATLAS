import importlib


def test_planner_decompose_generates_snapshot():
    module = importlib.import_module("modules.Tools.Base_Tools.planner_decompose")

    metadata = {
        "name": "DemoSkill",
        "required_tools": ["alpha"],
        "plan": {
            "steps": [
                {"id": "alpha_step", "tool": "alpha"},
            ]
        },
    }

    persona_overrides = {
        "metadata": {
            "required_tools": ["alpha", "beta"],
            "plan": {
                "steps": [
                    {"id": "alpha_step", "tool": "alpha"},
                    {
                        "id": "beta_step",
                        "tool": "beta",
                        "after": ["alpha_step"],
                        "inputs": {"source": "plan"},
                    },
                ]
            },
        }
    }

    result = module.planner_decompose(
        skill_metadata=metadata,
        available_tools=[{"name": "alpha"}, {"name": "beta"}],
        provided_inputs={"beta": {"origin": "provided"}},
        persona_overrides=persona_overrides,
    )

    assert result["statuses"] == {"alpha_step": "pending", "beta_step": "pending"}
    assert result["edges"] == [{"from": "alpha_step", "to": "beta_step"}]

    steps = result["steps"]
    assert [step["id"] for step in steps] == ["alpha_step", "beta_step"]

    beta_step = steps[1]
    assert beta_step["dependencies"] == ["alpha_step"]
    assert beta_step["inputs"] == {"origin": "provided", "source": "plan"}


def test_planner_decompose_available_in_default_map(monkeypatch):
    aiohttp_mod = importlib.import_module("aiohttp")

    class _ClientTimeout:
        def __init__(self, *_, **__):
            pass

    monkeypatch.setattr(
        aiohttp_mod,
        "ClientTimeout",
        getattr(aiohttp_mod, "ClientTimeout", _ClientTimeout),
        raising=False,
    )

    maps_module = importlib.import_module("modules.Tools.tool_maps.maps")
    maps_module = importlib.reload(maps_module)

    assert "planner.decompose" in maps_module.function_map
    assert maps_module.function_map["planner.decompose"] is not None
