"""Command line helpers for exporting and importing persona bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from modules.Personas import (
    PersonaBundleError,
    export_persona_bundle_bytes,
    import_persona_bundle_bytes,
)
from modules.Tools import (
    ToolBundleError,
    export_tool_bundle_bytes,
    import_tool_bundle_bytes,
)
from modules.Skills import (
    SkillBundleError,
    export_skill_bundle_bytes,
    import_skill_bundle_bytes,
)
from modules.Jobs import (
    JobBundleError,
    export_job_bundle_bytes,
    import_job_bundle_bytes,
)
from modules.Tasks import (
    TaskBundleError,
    export_task_bundle_bytes,
    import_task_bundle_bytes,
)
from modules.store_common.package_bundles import (
    AssetPackageError,
    export_asset_package_bytes,
    import_asset_package_bytes,
)


class _ConfigAdapter:
    """Lightweight adapter exposing ``get_app_root`` for persona helpers."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def get_app_root(self) -> str:
        return str(self._root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persona bundle utilities")
    parser.add_argument(
        "--app-root",
        dest="app_root",
        type=Path,
        help="Override the application root when loading personas.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export a persona to a signed bundle")
    export_parser.add_argument("persona", help="Persona name to export")
    export_parser.add_argument("output", type=Path, help="Destination bundle path")
    _add_signing_key_arguments(export_parser)

    import_parser = subparsers.add_parser("import", help="Import a signed persona bundle")
    import_parser.add_argument("bundle", type=Path, help="Bundle file to import")
    import_parser.add_argument(
        "--rationale",
        default="Imported via CLI persona tools",
        help="Audit rationale recorded with the import.",
    )
    _add_signing_key_arguments(import_parser)

    tool_export = subparsers.add_parser("export-tool", help="Export a tool to a signed bundle")
    tool_export.add_argument("tool", help="Tool name to export")
    tool_export.add_argument("output", type=Path, help="Destination bundle path")
    tool_export.add_argument("--persona", help="Persona owner for persona-specific tools")
    _add_signing_key_arguments(tool_export)

    tool_import = subparsers.add_parser("import-tool", help="Import a signed tool bundle")
    tool_import.add_argument("bundle", type=Path, help="Bundle file to import")
    tool_import.add_argument("--rationale", default="Imported via CLI persona tools", help="Audit rationale recorded with the import.")
    _add_signing_key_arguments(tool_import)

    skill_export = subparsers.add_parser("export-skill", help="Export a skill to a signed bundle")
    skill_export.add_argument("skill", help="Skill name to export")
    skill_export.add_argument("output", type=Path, help="Destination bundle path")
    skill_export.add_argument("--persona", help="Persona owner for persona-specific skills")
    _add_signing_key_arguments(skill_export)

    skill_import = subparsers.add_parser("import-skill", help="Import a signed skill bundle")
    skill_import.add_argument("bundle", type=Path, help="Bundle file to import")
    skill_import.add_argument("--rationale", default="Imported via CLI persona tools", help="Audit rationale recorded with the import.")
    _add_signing_key_arguments(skill_import)

    job_export = subparsers.add_parser("export-job", help="Export a job to a signed bundle")
    job_export.add_argument("job", help="Job name to export")
    job_export.add_argument("output", type=Path, help="Destination bundle path")
    job_export.add_argument("--persona", help="Persona owner for persona-specific jobs")
    _add_signing_key_arguments(job_export)

    job_import = subparsers.add_parser("import-job", help="Import a signed job bundle")
    job_import.add_argument("bundle", type=Path, help="Bundle file to import")
    job_import.add_argument("--rationale", default="Imported via CLI persona tools", help="Audit rationale recorded with the import.")
    _add_signing_key_arguments(job_import)

    task_export = subparsers.add_parser("export-task", help="Export a task to a signed bundle")
    task_export.add_argument("task", help="Task name to export")
    task_export.add_argument("output", type=Path, help="Destination bundle path")
    task_export.add_argument("--persona", help="Persona owner for persona-specific tasks")
    _add_signing_key_arguments(task_export)

    task_import = subparsers.add_parser("import-task", help="Import a signed task bundle")
    task_import.add_argument("bundle", type=Path, help="Bundle file to import")
    task_import.add_argument("--rationale", default="Imported via CLI persona tools", help="Audit rationale recorded with the import.")
    _add_signing_key_arguments(task_import)

    package_export = subparsers.add_parser(
        "export-package",
        help="Export a package containing personas, tools, skills, tasks, and jobs",
    )
    package_export.add_argument("--persona", action="append", dest="personas", help="Persona to include in the package (repeatable)")
    package_export.add_argument("--tool", action="append", dest="tools", help="Tool to include (use persona:name for persona-scoped tools)")
    package_export.add_argument("--skill", action="append", dest="skills", help="Skill to include (use persona:name for persona-scoped skills)")
    package_export.add_argument("--task", action="append", dest="tasks", help="Task to include (use persona:name for persona-scoped tasks)")
    package_export.add_argument("--job", action="append", dest="jobs", help="Job to include (use persona:name for persona-scoped jobs)")
    package_export.add_argument("--output", required=True, type=Path, help="Destination package path")
    _add_signing_key_arguments(package_export)

    package_import = subparsers.add_parser(
        "import-package",
        help="Import a package of personas, tools, skills, tasks, and jobs",
    )
    package_import.add_argument("bundle", type=Path, help="Package file to import")
    package_import.add_argument("--rationale", default="Imported via CLI persona tools", help="Audit rationale recorded with the import.")
    _add_signing_key_arguments(package_import)

    return parser


def _add_signing_key_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--signing-key", help="Signing key material provided directly")
    group.add_argument(
        "--signing-key-file",
        type=Path,
        help="Path to a file containing the signing key",
    )


def _load_signing_key(args: argparse.Namespace) -> str:
    if getattr(args, "signing_key", None):
        return str(args.signing_key)
    key_path: Optional[Path] = getattr(args, "signing_key_file", None)
    if key_path is None:
        raise PersonaBundleError("Signing key or signing key file must be provided.")
    try:
        return key_path.read_text(encoding="utf-8").strip()
    except OSError as exc:  # pragma: no cover - surfaced to CLI user
        raise PersonaBundleError(f"Failed to read signing key file: {key_path}") from exc


def _config_from_args(args: argparse.Namespace) -> Optional[_ConfigAdapter]:
    root: Optional[Path] = getattr(args, "app_root", None)
    return _ConfigAdapter(root) if root else None


def _cmd_export(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_bytes, persona = export_persona_bundle_bytes(
        args.persona,
        signing_key=signing_key,
        config_manager=config,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bundle_bytes)

    print(f"Exported persona '{persona.get('name', args.persona)}' to {output_path}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_path: Path = args.bundle
    try:
        bundle_bytes = bundle_path.read_bytes()
    except OSError as exc:
        raise PersonaBundleError(f"Failed to read bundle file: {bundle_path}") from exc

    result = import_persona_bundle_bytes(
        bundle_bytes,
        signing_key=signing_key,
        config_manager=config,
        rationale=args.rationale,
    )

    persona = result.get("persona", {})
    persona_name = persona.get("name") or "unknown"
    print(f"Imported persona '{persona_name}' from {bundle_path}")

    warnings = result.get("warnings") or []
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


def _cmd_export_tool(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_bytes, tool = export_tool_bundle_bytes(
        args.tool,
        signing_key=signing_key,
        persona=args.persona,
        config_manager=config,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bundle_bytes)

    print(f"Exported tool '{tool.get('name', args.tool)}' to {output_path}")
    return 0


def _cmd_import_tool(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_path: Path = args.bundle
    try:
        bundle_bytes = bundle_path.read_bytes()
    except OSError as exc:
        raise ToolBundleError(f"Failed to read bundle file: {bundle_path}") from exc

    result = import_tool_bundle_bytes(
        bundle_bytes,
        signing_key=signing_key,
        config_manager=config,
        rationale=args.rationale,
    )

    tool = result.get("tool", {})
    tool_name = tool.get("name") or "tool"
    print(f"Imported tool '{tool_name}' from {bundle_path}")
    return 0


def _cmd_export_skill(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_bytes, skill = export_skill_bundle_bytes(
        args.skill,
        signing_key=signing_key,
        persona=args.persona,
        config_manager=config,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bundle_bytes)

    print(f"Exported skill '{skill.get('name', args.skill)}' to {output_path}")
    return 0


def _cmd_import_skill(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_path: Path = args.bundle
    try:
        bundle_bytes = bundle_path.read_bytes()
    except OSError as exc:
        raise SkillBundleError(f"Failed to read bundle file: {bundle_path}") from exc

    result = import_skill_bundle_bytes(
        bundle_bytes,
        signing_key=signing_key,
        config_manager=config,
        rationale=args.rationale,
    )

    skill = result.get("skill", {})
    skill_name = skill.get("name") or "skill"
    print(f"Imported skill '{skill_name}' from {bundle_path}")
    return 0


def _cmd_export_job(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_bytes, job = export_job_bundle_bytes(
        args.job,
        signing_key=signing_key,
        persona=args.persona,
        config_manager=config,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bundle_bytes)

    print(f"Exported job '{job.get('name', args.job)}' to {output_path}")
    return 0


def _cmd_import_job(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_path: Path = args.bundle
    try:
        bundle_bytes = bundle_path.read_bytes()
    except OSError as exc:
        raise JobBundleError(f"Failed to read bundle file: {bundle_path}") from exc

    result = import_job_bundle_bytes(
        bundle_bytes,
        signing_key=signing_key,
        config_manager=config,
        rationale=args.rationale,
    )

    job = result.get("job", {})
    job_name = job.get("name") or "job"
    print(f"Imported job '{job_name}' from {bundle_path}")
    return 0


def _cmd_export_task(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_bytes, task = export_task_bundle_bytes(
        args.task,
        signing_key=signing_key,
        persona=args.persona,
        config_manager=config,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bundle_bytes)

    print(f"Exported task '{task.get('name', args.task)}' to {output_path}")
    return 0


def _cmd_import_task(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_path: Path = args.bundle
    try:
        bundle_bytes = bundle_path.read_bytes()
    except OSError as exc:
        raise TaskBundleError(f"Failed to read bundle file: {bundle_path}") from exc

    result = import_task_bundle_bytes(
        bundle_bytes,
        signing_key=signing_key,
        config_manager=config,
        rationale=args.rationale,
    )

    task = result.get("task", {})
    task_name = task.get("name") or "task"
    print(f"Imported task '{task_name}' from {bundle_path}")
    return 0


def _cmd_export_package(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)

    bundle_bytes, metadata = export_asset_package_bytes(
        personas=args.personas,
        tools=args.tools,
        skills=args.skills,
        tasks=args.tasks,
        jobs=args.jobs,
        signing_key=signing_key,
        config_manager=config,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bundle_bytes)

    counts = metadata.get("counts", {})
    summary = ", ".join(
        f"{key}={value}" for key, value in counts.items() if value
    ) or "no assets"
    print(f"Exported asset package ({summary}) to {output_path}")
    return 0


def _cmd_import_package(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_path: Path = args.bundle
    try:
        bundle_bytes = bundle_path.read_bytes()
    except OSError as exc:
        raise AssetPackageError(f"Failed to read package file: {bundle_path}") from exc

    result = import_asset_package_bytes(
        bundle_bytes,
        signing_key=signing_key,
        config_manager=config,
        rationale=args.rationale,
    )

    assets = result.get("assets", [])
    print(f"Imported asset package from {bundle_path} containing {len(assets)} assets")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "export":
            return _cmd_export(args)
        if args.command == "import":
            return _cmd_import(args)
        if args.command == "export-tool":
            return _cmd_export_tool(args)
        if args.command == "import-tool":
            return _cmd_import_tool(args)
        if args.command == "export-skill":
            return _cmd_export_skill(args)
        if args.command == "import-skill":
            return _cmd_import_skill(args)
        if args.command == "export-job":
            return _cmd_export_job(args)
        if args.command == "import-job":
            return _cmd_import_job(args)
        if args.command == "export-task":
            return _cmd_export_task(args)
        if args.command == "import-task":
            return _cmd_import_task(args)
        if args.command == "export-package":
            return _cmd_export_package(args)
        if args.command == "import-package":
            return _cmd_import_package(args)
    except (
        PersonaBundleError,
        ToolBundleError,
        SkillBundleError,
        JobBundleError,
        TaskBundleError,
        AssetPackageError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
