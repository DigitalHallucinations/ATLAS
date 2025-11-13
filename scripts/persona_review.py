#!/usr/bin/env python3
"""Cron-friendly helper that queues persona reviews when overdue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from modules.Personas.persona_review import (
    REVIEW_INTERVAL_DAYS,
    PersonaReviewScheduler,
    compute_review_status,
    discover_persona_names,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Queue persona review tasks")
    parser.add_argument(
        "--persona",
        dest="personas",
        action="append",
        help="Specific persona to evaluate (may be provided multiple times)",
    )
    parser.add_argument(
        "--interval-days",
        type=int,
        default=REVIEW_INTERVAL_DAYS,
        help="Days before a persona review expires (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate personas without queuing tasks",
    )
    return parser


def _collect_persona_names(args: argparse.Namespace) -> List[str]:
    if args.personas:
        return [name for name in args.personas if name]
    return discover_persona_names()


def _dry_run(persona_names: Iterable[str], *, interval_days: int) -> int:
    now = datetime.now(timezone.utc)
    overdue: List[str] = []

    for persona_name in persona_names:
        status = compute_review_status(
            persona_name,
            now=now,
            interval_days=interval_days,
        )
        if status.overdue:
            overdue.append(
                f"{persona_name}: due {status.next_due or status.expires_at or 'now'}"
            )

    if overdue:
        print("Overdue persona reviews detected:")
        for entry in overdue:
            print(f" - {entry}")
    else:
        print("No overdue personas detected.")

    return 0


def _queue_reviews(persona_names: Iterable[str], *, interval_days: int) -> int:
    scheduler = PersonaReviewScheduler(interval_days=interval_days)
    queued = scheduler.scan_and_queue(list(persona_names))

    if queued:
        print("Queued review tasks for: " + ", ".join(sorted(queued)))
    else:
        print("No new review tasks queued.")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    persona_names = _collect_persona_names(args)

    if not persona_names:
        print("No personas discovered. Nothing to do.")
        return 0

    if args.dry_run:
        return _dry_run(persona_names, interval_days=args.interval_days)

    return _queue_reviews(persona_names, interval_days=args.interval_days)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

