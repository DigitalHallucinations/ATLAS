from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import types

import pytest

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub

from modules.logging.audit import (
    PersonaAuditLogger,
    PersonaReviewLogger,
    PersonaReviewQueue,
)
from modules.persona_review import PersonaReviewScheduler, compute_review_status


@pytest.fixture
def review_components(tmp_path: Path):
    audit_log = tmp_path / "audit.jsonl"
    review_log = tmp_path / "reviews.jsonl"
    queue_file = tmp_path / "queue.json"

    audit_logger = PersonaAuditLogger(log_path=audit_log)
    review_logger = PersonaReviewLogger(log_path=review_log)
    review_queue = PersonaReviewQueue(queue_path=queue_file)

    audit_logger.clear()
    review_logger.clear()

    yield audit_logger, review_logger, review_queue

    audit_logger.clear()
    review_logger.clear()


def test_scheduler_queues_overdue_personas(review_components) -> None:
    audit_logger, review_logger, review_queue = review_components

    base = datetime(2024, 5, 1, tzinfo=timezone.utc)

    audit_logger.record_change(
        "Alpha",
        ["tool-a"],
        ["tool-b"],
        timestamp=base - timedelta(days=60),
    )

    audit_logger.record_change(
        "Beta",
        ["tool-a"],
        ["tool-b"],
        timestamp=base - timedelta(days=80),
    )
    review_logger.record_attestation(
        "Beta",
        reviewer="auditor",
        timestamp=base - timedelta(days=50),
        expires_at=base - timedelta(days=5),
    )

    audit_logger.record_change(
        "Gamma",
        ["tool-a"],
        ["tool-b"],
        timestamp=base - timedelta(days=10),
    )
    review_logger.record_attestation(
        "Gamma",
        reviewer="auditor",
        timestamp=base - timedelta(days=5),
        expires_at=base + timedelta(days=60),
    )

    scheduler = PersonaReviewScheduler(
        audit_logger=audit_logger,
        review_logger=review_logger,
        review_queue=review_queue,
        interval_days=45,
    )

    queued = scheduler.scan_and_queue(["Alpha", "Beta", "Gamma"], now=base)

    assert sorted(queued) == ["Alpha", "Beta"]
    assert review_queue.is_pending("Alpha") is True
    assert review_queue.is_pending("Beta") is True
    assert review_queue.is_pending("Gamma") is False

    status_alpha = compute_review_status(
        "Alpha",
        audit_logger=audit_logger,
        review_logger=review_logger,
        review_queue=review_queue,
        now=base,
        interval_days=45,
    )
    assert status_alpha.overdue is True
    assert status_alpha.pending_task is True
    assert status_alpha.next_due is not None

    # Running the scheduler again should not duplicate pending tasks.
    queued_again = scheduler.scan_and_queue(["Alpha", "Beta"], now=base)
    assert queued_again == []


def test_attestation_clears_pending_queue(review_components) -> None:
    audit_logger, review_logger, review_queue = review_components

    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    audit_logger.record_change(
        "Atlas",
        ["tool-a"],
        ["tool-b"],
        timestamp=base - timedelta(days=90),
    )

    review_queue.enqueue(
        "Atlas",
        due_at=base - timedelta(days=1),
        reason="Expired",
        now=base,
    )

    attestation = review_logger.record_attestation(
        "Atlas",
        reviewer="auditor",
        timestamp=base,
        expires_at=base + timedelta(days=90),
    )

    # Queue should remain pending until explicitly cleared.
    assert review_queue.is_pending("Atlas") is True

    review_queue.mark_completed("Atlas", timestamp=base)
    assert review_queue.is_pending("Atlas") is False
    assert attestation.persona_name == "Atlas"
    assert attestation.reviewer == "auditor"
