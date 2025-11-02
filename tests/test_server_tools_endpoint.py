from modules.Server.routes import atlas_server


def _lowercase_capabilities(tool):
    return {cap.lower() for cap in tool.get("capabilities", [])}


def test_get_tools_returns_merged_metadata():
    payload = atlas_server.handle_request("/tools")
    assert payload["count"] == len(payload["tools"])

    shared_google = next(
        tool
        for tool in payload["tools"]
        if tool["name"] == "google_search" and tool["persona"] is None
    )
    assert shared_google["version"] == "1.0.0"
    assert "web_search" in _lowercase_capabilities(shared_google)
    assert shared_google["auth"]["required"] is True
    assert shared_google["auth_required"] is True
    assert "settings" in shared_google
    assert "credentials" in shared_google


def test_get_tools_filters_by_capability():
    payload = atlas_server.handle_request("/tools", query={"capability": "web_search"})
    tools = payload["tools"]
    assert tools, "Expected at least one tool with the web_search capability"
    for tool in tools:
        assert "web_search" in _lowercase_capabilities(tool)


def test_get_tools_filters_by_safety_and_persona():
    payload = atlas_server.handle_request(
        "/tools",
        query={"persona": "CodeGenius", "safety_level": "high"},
    )
    tools = payload["tools"]
    assert tools, "Expected at least one high safety tool for CodeGenius"
    for tool in tools:
        persona = (tool["persona"] or "").lower()
        if persona:
            assert persona == "codegenius"
        assert (tool.get("safety_level") or "").lower() == "high"
