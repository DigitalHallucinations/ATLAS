from modules.Personas import utils


def test_normalize_persona_allowlist_handles_iterables() -> None:
    allowlist = utils.normalize_persona_allowlist({"primary": "Atlas", "alt": " atlas "})
    assert allowlist == {"Atlas", "atlas"}


def test_normalize_requires_flags_deduplicates_and_coerces() -> None:
    raw = {"Create": ["FlagA", "flaga", ""], " ": ["ignored"]}
    normalized = utils.normalize_requires_flags(raw)
    assert normalized == {"create": ("FlagA", "flaga")}


def test_collect_missing_flag_requirements_detects_missing_flags() -> None:
    requires = {"create": ("flags.enable", "flags.write"), "update": ("flags.enable",)}
    persona = {"flags": {"enable": "true"}}
    missing = utils.collect_missing_flag_requirements(requires, persona)
    assert missing == {"create": ("flags.write",)}


def test_format_denied_operations_summary_mentions_tool_and_flags() -> None:
    summary = utils.format_denied_operations_summary(
        "browser_tool",
        {"create": ("flags.write", "flags.audit")},
    )
    assert "browser_tool" in summary
    assert "flags.write" in summary
    assert "flags.audit" in summary
