"""Unit tests for llm_models/ — registry, factory, env overrides, validation."""

from __future__ import annotations

import pytest

from clients.registry import MODEL_CAPABILITIES
from exceptions.configuration import InvalidRateLimitConfigError
from exceptions.request import UnknownModelError
from llm_models import get_model_config, MODEL_REGISTRY
from llm_models.base import ModelRateLimitConfig
from llm_models.llama_3_3_70b_versatile import Llama3370BVersatileConfig


class TestModelRegistryCompleteness:

    def test_model_registry_and_capabilities_have_same_keys(self):
        registry_keys = set(MODEL_REGISTRY.keys())
        capability_keys = set(MODEL_CAPABILITIES.keys())
        missing_from_capabilities = registry_keys - capability_keys
        missing_from_registry = capability_keys - registry_keys
        assert (
            not missing_from_capabilities
        ), f"Models in MODEL_REGISTRY but not MODEL_CAPABILITIES: {missing_from_capabilities}"
        assert (
            not missing_from_registry
        ), f"Models in MODEL_CAPABILITIES but not MODEL_REGISTRY: {missing_from_registry}"

    def test_model_registry_has_expected_count(self):
        assert len(MODEL_REGISTRY) >= 17

    def test_all_registry_entries_are_subclasses(self):
        for model_id, cls in MODEL_REGISTRY.items():
            assert issubclass(
                cls, ModelRateLimitConfig
            ), f"{model_id} maps to {cls} which is not a ModelRateLimitConfig subclass"


class TestGetModelConfig:

    def test_returns_config_for_known_model(self):
        cfg = get_model_config("llama-3.3-70b-versatile")
        assert cfg.model_id == "llama-3.3-70b-versatile"

    def test_all_registered_models_resolve(self):
        for model_id in MODEL_REGISTRY:
            cfg = get_model_config(model_id)
            assert cfg.model_id == model_id

    def test_unknown_model_raises_typed_exception(self):
        with pytest.raises(UnknownModelError) as exc_info:
            get_model_config("does-not-exist")
        assert exc_info.value.model_id == "does-not-exist"
        assert isinstance(exc_info.value.available_models, list)
        assert len(exc_info.value.available_models) > 0

    def test_result_is_model_rate_limit_config_instance(self):
        cfg = get_model_config("llama-3.3-70b-versatile")
        assert isinstance(cfg, ModelRateLimitConfig)


class TestModelRateLimitConfigValidation:

    def test_negative_rpm_raises_on_construction(self):
        with pytest.raises(InvalidRateLimitConfigError) as exc_info:
            Llama3370BVersatileConfig(requests_per_minute=-1)
        assert exc_info.value.field == "requests_per_minute"

    def test_zero_rpm_raises_on_construction(self):
        with pytest.raises(InvalidRateLimitConfigError):
            Llama3370BVersatileConfig(requests_per_minute=0)

    def test_valid_construction_succeeds(self):
        cfg = Llama3370BVersatileConfig(requests_per_minute=30)
        assert cfg.requests_per_minute == 30


class TestEnvOverrides:

    def test_from_env_applies_rpm_override(self, monkeypatch):
        monkeypatch.setenv("GROQ_MODEL_LLAMA_33_70B_VERSATILE_REQUESTS_PER_MINUTE", "60")
        cfg = Llama3370BVersatileConfig.from_env()
        assert cfg.requests_per_minute == 60

    def test_from_env_uses_defaults_when_no_override(self, monkeypatch):
        monkeypatch.delenv("GROQ_MODEL_LLAMA_33_70B_VERSATILE_REQUESTS_PER_MINUTE", raising=False)
        cfg = Llama3370BVersatileConfig.from_env()
        assert cfg.requests_per_minute == 30


class TestDerivedProperties:

    def test_is_audio_model_for_whisper(self):
        cfg = get_model_config("whisper-large-v3")
        assert cfg.is_audio_model is True

    def test_is_text_model_for_llama(self):
        cfg = get_model_config("llama-3.3-70b-versatile")
        assert cfg.is_text_model is True

    def test_is_not_audio_model_for_text_model(self):
        cfg = get_model_config("llama-3.3-70b-versatile")
        assert cfg.is_audio_model is False

    def test_active_limits_excludes_none_fields(self):
        cfg = get_model_config("llama-3.3-70b-versatile")
        limits = cfg.active_limits()
        assert all(v is not None for v in limits.values())

    def test_to_dict_round_trips_model_id(self):
        cfg = get_model_config("llama-3.3-70b-versatile")
        d = cfg.to_dict()
        assert d["model_id"] == "llama-3.3-70b-versatile"
        assert d["plan"] == "free"
