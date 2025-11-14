from __future__ import annotations

from modules.background_tasks.conversation_summary import extract_followups


def _build_history(entries):
    return [
        {
            "index": index,
            "role": role,
            "content": content,
        }
        for index, (role, content) in enumerate(entries)
    ]


def test_unanswered_question_detected():
    snapshot = {
        "summary": "User requested the latest report",
        "highlights": [],
        "history": _build_history(
            [
                ("assistant", "Here's yesterday's summary."),
                ("user", "Can you send the updated report today?"),
            ]
        ),
    }

    templates = [
        {
            "id": "outstanding-question",
            "kind": "question",
            "title": "Follow up on unanswered question",
            "description": "A participant asked a question that has not been answered yet.",
            "matching": {"unanswered_question": True, "include_roles": ["user"]},
            "task": {"manifest": "ClarifyOutstandingQuestion"},
        }
    ]

    followups = extract_followups(snapshot, templates)

    assert len(followups) == 1
    followup = followups[0]
    assert followup["template_id"] == "outstanding-question"
    assert "unanswered_question" in followup["reasons"]
    assert followup["source"]["role"] == "user"
    assert "report" in followup["evidence"].lower()


def test_answered_question_not_flagged():
    snapshot = {
        "summary": "Assistant responded to question",
        "highlights": [],
        "history": _build_history(
            [
                ("user", "Can you update the roadmap?"),
                ("assistant", "Yes, I'll refresh it now."),
            ]
        ),
    }

    templates = [
        {
            "id": "outstanding-question",
            "kind": "question",
            "title": "Follow up on unanswered question",
            "description": "A participant asked a question that has not been answered yet.",
            "matching": {"unanswered_question": True, "include_roles": ["user"]},
        }
    ]

    followups = extract_followups(snapshot, templates)

    assert followups == []


def test_keyword_highlight_detection():
    snapshot = {
        "summary": "Pending vendor follow up",
        "highlights": ["Need to follow up with the vendor about the invoice."],
        "history": _build_history([("assistant", "Remember the open invoice."), ("user", "Noted.")]),
    }

    templates = [
        {
            "id": "followup-keyword",
            "kind": "reminder",
            "title": "Schedule vendor follow up",
            "description": "Highlights mention a required follow-up action.",
            "matching": {"keywords": ["follow up"], "scope": ["highlights"]},
        }
    ]

    followups = extract_followups(snapshot, templates)

    assert len(followups) == 1
    followup = followups[0]
    assert followup["template_id"] == "followup-keyword"
    assert "keyword" in followup["reasons"]
    assert followup["source"]["type"] == "highlight"
