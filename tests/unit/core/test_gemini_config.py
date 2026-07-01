"""Unit tests for poolgate/core/gemini_config.py."""

from __future__ import annotations

import pytest

from poolgate.core.gemini_config import GeminiConfig
from poolgate.exceptions.configuration import ConfigurationError


def test_default_paths_use_gemini_namespace():
    config = GeminiConfig(api_keys=["AIza_test"])
    assert config.paths.namespace == "gemini"
    assert config.paths.usage_json.name == "gemini_usage.json"
    assert config.paths.tokens_json.name == "gemini_tokens.json"
    assert config.paths.account_json.name == "gemini_account.json"


def test_from_env_requires_total_gemini_keys(monkeypatch):
    with pytest.raises(ConfigurationError, match="TOTAL_GEMINI_KEYS"):
        GeminiConfig.from_env()


def test_from_env_paths_also_namespaced(monkeypatch):
    monkeypatch.setenv("TOTAL_GEMINI_KEYS", "1")
    monkeypatch.setenv("GEMINI_API_KEY_01", "AIza_test")
    config = GeminiConfig.from_env()
    assert config.paths.namespace == "gemini"
