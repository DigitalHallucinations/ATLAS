from __future__ import annotations

import uuid

import pytest

from modules.conversation_store._shared import (
    _coerce_vector_payload,
    _hash_vector,
)


def test_coerce_vector_payload_generates_checksum_when_missing():
    message_id = uuid.uuid4()
    conversation_id = uuid.uuid4()

    payload = _coerce_vector_payload(
        message_id,
        conversation_id,
        {"values": [0.1, 0.2, 0.3]},
    )

    expected_checksum = _hash_vector([0.1, 0.2, 0.3])
    assert payload.checksum == expected_checksum


def test_coerce_vector_payload_sets_default_metadata():
    message_id = uuid.uuid4()
    conversation_id = uuid.uuid4()

    payload = _coerce_vector_payload(
        message_id,
        conversation_id,
        {
            "values": [1, 2, 3],
            "provider": "provider-x",
            "model": "model-y",
            "model_version": "1",
        },
    )

    metadata = payload.metadata
    assert metadata["conversation_id"] == str(conversation_id)
    assert metadata["message_id"] == str(message_id)
    assert metadata["namespace"] == str(conversation_id)
    assert metadata["provider"] == "provider-x"
    assert metadata["model"] == "model-y"
    assert metadata["model_version"] == "1"
    assert metadata["dimensions"] == 3
    assert metadata["vector_key"] == payload.vector_key


@pytest.mark.parametrize(
    "raw_payload, expected_exception",
    [
        ({"values": []}, ValueError),
        ({"values": "abc"}, TypeError),
        ({}, ValueError),
    ],
)
def test_coerce_vector_payload_errors(raw_payload, expected_exception):
    message_id = uuid.uuid4()
    conversation_id = uuid.uuid4()

    with pytest.raises(expected_exception):
        _coerce_vector_payload(message_id, conversation_id, raw_payload)
