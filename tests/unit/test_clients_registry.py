"""Unit tests for clients/registry.py — assert_capability and models_for."""

from __future__ import annotations

import pytest

from clients.registry import MODEL_CAPABILITIES, assert_capability, models_for
from exceptions.request import CapabilityError


class TestAssertCapability:
    def test_known_model_with_supported_capability_passes(self):
        assert_capability("llama-3.3-70b-versatile", "chat")  # no exception

    def test_chat_model_supports_tools(self):
        assert_capability("llama-3.3-70b-versatile", "tools")

    def test_chat_model_supports_structured(self):
        assert_capability("llama-3.3-70b-versatile", "structured")

    def test_whisper_supports_transcription(self):
        assert_capability("whisper-large-v3", "transcription")

    def test_moderation_model_passes_moderation(self):
        assert_capability("meta-llama/llama-prompt-guard-2-86m", "moderation")

    def test_synthesis_model_passes_synthesis(self):
        assert_capability("canopylabs/orpheus-v1-english", "synthesis")

    def test_chat_model_does_not_support_synthesis(self):
        with pytest.raises(CapabilityError) as exc_info:
            assert_capability("llama-3.3-70b-versatile", "synthesis")
        assert exc_info.value.model_id == "llama-3.3-70b-versatile"
        assert exc_info.value.capability == "synthesis"

    def test_whisper_does_not_support_chat(self):
        with pytest.raises(CapabilityError) as exc_info:
            assert_capability("whisper-large-v3", "chat")
        assert exc_info.value.model_id == "whisper-large-v3"
        assert exc_info.value.capability == "chat"

    def test_unregistered_model_raises_capability_error(self):
        with pytest.raises(CapabilityError) as exc_info:
            assert_capability("unknown-model-xyz", "chat")
        assert exc_info.value.model_id == "unknown-model-xyz"
        assert exc_info.value.supported_capabilities == []

    def test_moderation_model_does_not_support_tools(self):
        with pytest.raises(CapabilityError):
            assert_capability("meta-llama/llama-prompt-guard-2-86m", "tools")

    def test_synthesis_model_does_not_support_transcription(self):
        with pytest.raises(CapabilityError):
            assert_capability("canopylabs/orpheus-v1-english", "transcription")


class TestModelsFor:
    def test_moderation_returns_three_models(self):
        models = models_for("moderation")
        assert "meta-llama/llama-prompt-guard-2-22m" in models
        assert "meta-llama/llama-prompt-guard-2-86m" in models
        assert "openai/gpt-oss-safeguard-20b" in models

    def test_transcription_returns_whisper_models(self):
        models = models_for("transcription")
        assert "whisper-large-v3" in models
        assert "whisper-large-v3-turbo" in models

    def test_synthesis_returns_orpheus_models(self):
        models = models_for("synthesis")
        assert "canopylabs/orpheus-arabic-saudi" in models
        assert "canopylabs/orpheus-v1-english" in models

    def test_chat_returns_non_empty_list(self):
        models = models_for("chat")
        assert len(models) > 0
        assert "llama-3.3-70b-versatile" in models

    def test_tools_returns_non_empty_list(self):
        models = models_for("tools")
        assert len(models) > 0

    def test_structured_returns_non_empty_list(self):
        models = models_for("structured")
        assert len(models) > 0

    def test_result_is_sorted(self):
        models = models_for("chat")
        assert models == sorted(models)

    def test_all_six_capabilities_have_at_least_one_model(self):
        for cap in ("chat", "structured", "tools", "moderation", "transcription", "synthesis"):
            assert len(models_for(cap)) > 0, f"No models found for capability '{cap}'"


class TestModelCapabilitiesDict:
    def test_all_entries_are_sets(self):
        for model_id, caps in MODEL_CAPABILITIES.items():
            assert isinstance(caps, set), f"{model_id} caps is not a set"

    def test_capability_strings_are_valid(self):
        valid = {"chat", "structured", "tools", "moderation", "transcription", "synthesis"}
        for model_id, caps in MODEL_CAPABILITIES.items():
            for cap in caps:
                assert cap in valid, f"{model_id} has unknown capability '{cap}'"
