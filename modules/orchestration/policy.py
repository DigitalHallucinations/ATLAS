"""Messaging policy helpers.

This module defines policy metadata for message topics and a resolver that
selects the most specific policy based on topic prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class MessagePolicy:
    """Policy metadata used by the message bus.

    Attributes
    ----------
    tier:
        Logical classification for the topic (for example, "standard",
        "priority", or "critical").
    retry_attempts:
        Default retry attempts when subscribers do not override retry
        configuration.
    retry_delay:
        Delay (in seconds) to wait between retries when handlers fail.
    dlq_topic_template:
        Template used to derive the dead-letter queue topic. The current topic
        is injected via ``{topic}`` formatting.
    replay_start:
        Optional bookmark used by durable backends to replay historical
        messages.
    retention_seconds:
        Desired retention window for the backend to hold messages.
    idempotency_key_field / idempotency_ttl_seconds:
        Optional idempotency hints that can be used by backends or subscribers
        to deduplicate deliveries.
    """

    tier: str = "standard"
    retry_attempts: int = 3
    retry_delay: float = 0.1
    dlq_topic_template: Optional[str] = "dlq.{topic}"
    replay_start: Optional[str] = None
    retention_seconds: Optional[int] = None
    idempotency_key_field: Optional[str] = None
    idempotency_ttl_seconds: Optional[int] = None


class PolicyResolver:
    """Resolve message policies using a longest-prefix match."""

    def __init__(
        self,
        policies: Mapping[str, MessagePolicy],
        *,
        default_policy: Optional[MessagePolicy] = None,
    ) -> None:
        self._policies: Dict[str, MessagePolicy] = dict(policies)
        self._default = default_policy

    def resolve(self, topic: str) -> Optional[MessagePolicy]:
        """Return the most specific policy for *topic*.

        The resolver chooses the policy whose prefix matches *topic* with the
        greatest length. If no prefix matches, the optional ``default_policy``
        is returned.
        """

        selected_prefix: Optional[str] = None
        for prefix in self._policies:
            if topic.startswith(prefix):
                if selected_prefix is None or len(prefix) > len(selected_prefix):
                    selected_prefix = prefix

        if selected_prefix is not None:
            return self._policies[selected_prefix]
        return self._default
