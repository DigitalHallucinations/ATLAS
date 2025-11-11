from pathlib import Path

import pytest

from modules.store_common.manifest_utils import (
    coerce_string,
    coerce_string_tuple,
    iter_persona_manifest_paths,
    merge_with_base,
)


def test_merge_with_base_overrides_and_preserves_base_copy():
    known = {
        "BaseEntry": {
            "name": "BaseEntry",
            "summary": "original",
            "metadata": {"value": 1},
        }
    }

    entry = {"extends": "BaseEntry", "summary": "updated", "metadata": {"value": 2}}
    merged = merge_with_base(entry, known)
    assert merged is not None
    assert merged["name"] == "BaseEntry"
    assert merged["summary"] == "updated"
    assert merged["metadata"] == {"value": 2}

    merged["metadata"]["value"] = 99
    assert known["BaseEntry"]["metadata"]["value"] == 1


def test_merge_with_base_uses_name_when_extends_missing():
    known = {"Shared": {"name": "Shared", "summary": "base"}}
    entry = {"name": "Shared", "description": "custom"}

    merged = merge_with_base(entry, known)
    assert merged is not None
    assert merged["summary"] == "base"
    assert merged["description"] == "custom"


def test_merge_with_base_returns_none_for_unknown_reference():
    merged = merge_with_base({"extends": "Missing"}, {})
    assert merged is None


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, ""),
        ("  spaced  ", "spaced"),
        (123, "123"),
    ],
)
def test_coerce_string(value, expected):
    assert coerce_string(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, tuple()),
        ("single", ("single",)),
        (["A", "", "B"], ("A", "B")),
        (("x", 5), ("x", "5")),
    ],
)
def test_coerce_string_tuple(value, expected):
    assert coerce_string_tuple(value) == expected


def test_iter_persona_manifest_paths(tmp_path: Path) -> None:
    personas_root = tmp_path / "modules" / "Personas"
    (personas_root / "Beta" / "Jobs").mkdir(parents=True, exist_ok=True)
    (personas_root / "Alpha" / "Jobs").mkdir(parents=True, exist_ok=True)

    paths = list(iter_persona_manifest_paths(tmp_path, "Jobs", "jobs.json"))
    assert [name for name, _ in paths] == ["Alpha", "Beta"]
    assert all(path == personas_root / name / "Jobs" / "jobs.json" for name, path in paths)
