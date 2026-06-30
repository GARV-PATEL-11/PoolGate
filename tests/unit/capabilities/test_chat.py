"""
Unit tests for schemas/common/runtime.py's RuntimeChatMessage.validate_role().

Regression tests for the audit's Phase 7/H2 finding: validate_role() used
to raise a bare ValueError despite its neighboring exception's docstring
claiming it raises InvalidMessageRoleError. It now actually does.
"""

from __future__ import annotations

import pytest

from poolgate.exceptions.request import InvalidMessageRoleError
from poolgate.schemas.common.runtime import RuntimeChatMessage


def test_valid_roles_are_accepted():
    for role in ("system", "user", "assistant", "tool"):
        msg = RuntimeChatMessage(role=role, content="hello")
        assert msg.role == role


def test_invalid_role_raises_invalid_message_role_error():
    with pytest.raises(InvalidMessageRoleError) as exc_info:
        RuntimeChatMessage(role="narrator", content="hello")
    assert exc_info.value.role == "narrator"
    assert "system" in exc_info.value.allowed_roles


def test_invalid_role_error_message_lists_allowed_roles():
    with pytest.raises(InvalidMessageRoleError) as exc_info:
        RuntimeChatMessage(role="bogus", content="x")
    message = str(exc_info.value)
    for role in ("system", "user", "assistant", "tool"):
        assert role in message
