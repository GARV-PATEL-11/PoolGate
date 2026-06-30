"""Unit tests for poolgate/core/config.py."""

from __future__ import annotations

import pytest

from poolgate.core.config import GroqConfig
from poolgate.exceptions.configuration import ConfigurationError, EnvironmentParseError


def _set_numbered_keys(monkeypatch, *keys: str) -> None:
    """Populate TOTAL_GROQ_KEYS + GROQ_API_KEY_NN from positional args."""
    monkeypatch.setenv("TOTAL_GROQ_KEYS", str(len(keys)))
    for i, key in enumerate(keys, start=1):
        monkeypatch.setenv(f"GROQ_API_KEY_{i:02d}", key)


def test_from_env_requires_total_groq_keys(monkeypatch):
    with pytest.raises(ConfigurationError, match="TOTAL_GROQ_KEYS"):
        GroqConfig.from_env()


def test_from_env_rejects_non_integer_total_groq_keys(monkeypatch):
    monkeypatch.setenv("TOTAL_GROQ_KEYS", "not_a_number")
    with pytest.raises(EnvironmentParseError) as exc_info:
        GroqConfig.from_env()
    assert exc_info.value.var_name == "TOTAL_GROQ_KEYS"


def test_from_env_rejects_zero_key_count(monkeypatch):
    monkeypatch.setenv("TOTAL_GROQ_KEYS", "0")
    with pytest.raises(ConfigurationError, match=r"(?i)no valid.*keys"):
        GroqConfig.from_env()


def test_from_env_parses_numbered_keys(monkeypatch):
    _set_numbered_keys(monkeypatch, "gsk_test_1", "gsk_test_2", "gsk_test_3")
    config = GroqConfig.from_env()
    assert config.api_keys == ["gsk_test_1", "gsk_test_2", "gsk_test_3"]


def test_comma_separated_format_is_not_read_by_from_env(monkeypatch):
    """
    Regression guard: the old GROQ_API_KEYS comma-separated variable must NOT
    be honoured by from_env().  Only TOTAL_GROQ_KEYS + GROQ_API_KEY_NN is
    valid.  If someone reverts the loader to the old format, this fails loudly.
    """
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_a,gsk_b")
    with pytest.raises(ConfigurationError):
        GroqConfig.from_env()


def test_from_env_rejects_non_integer_rpm(monkeypatch):
    _set_numbered_keys(monkeypatch, "gsk_a")
    monkeypatch.setenv("GROQ_MAX_RPM", "not_a_number")
    with pytest.raises(EnvironmentParseError) as exc_info:
        GroqConfig.from_env()
    assert exc_info.value.var_name == "GROQ_MAX_RPM"


def test_from_env_loads_failure_threshold(monkeypatch):
    _set_numbered_keys(monkeypatch, "gsk_a")
    monkeypatch.setenv("GROQ_FAILURE_THRESHOLD", "7")
    config = GroqConfig.from_env()
    assert config.failure_threshold == 7
