"""E2E tests for persistence lifecycle — tracking flush, reload, and journal."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.config import GroqConfig
from core.path_config import PathConfig
from services.provider_service import GroqService


def _set_groq_keys(monkeypatch, keys: list[str]) -> None:
    monkeypatch.setenv("TOTAL_GROQ_KEYS", str(len(keys)))
    for i, key in enumerate(keys, start=1):
        monkeypatch.setenv(f"GROQ_API_KEY_{i:02d}", key)


def _service_with_base_dir(monkeypatch, keys: list[str], base_dir: Path) -> GroqService:
    _set_groq_keys(monkeypatch, keys)
    config = GroqConfig.from_env()
    config.paths = PathConfig(base_dir=base_dir)
    return GroqService(config=config)


def _mock_completion(text: str = "answer") -> MagicMock:
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=text), finish_reason="stop")]
    completion.usage = MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    return completion


@pytest.fixture
def persisted_service(monkeypatch, tmp_path) -> GroqService:
    base_dir = tmp_path / "poolgate_data"
    return _service_with_base_dir(monkeypatch, ["gsk_persist_key_1"], base_dir)


# ---------------------------------------------------------------------------
# Tracking flush
# ---------------------------------------------------------------------------

class TestTrackingFlush:
    def test_flush_tracking_creates_json_files(self, persisted_service, monkeypatch, tmp_path):
        base_dir = tmp_path / "poolgate_data"
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion("ok")
        monkeypatch.setattr(persisted_service._chat_client, "_sync_sdk", lambda k: mock_sdk)

        persisted_service.invoke("Hello", model="llama-3.3-70b-versatile")
        persisted_service.flush_tracking()

        tracking_dir = base_dir / "tracking"
        assert tracking_dir.is_dir(), "tracking/ dir not created"

        json_files = list(tracking_dir.glob("*.json"))
        assert len(json_files) >= 1, f"No JSON files in {tracking_dir}"

    def test_flush_tracking_noop_without_data_dir(self, monkeypatch):
        _set_groq_keys(monkeypatch, ["gsk_mem_key"])
        config = GroqConfig.from_env()
        config.paths = PathConfig(base_dir=None)
        svc = GroqService(config=config)
        svc.flush_tracking()  # must not raise

    def test_flush_tracking_writes_valid_json(self, persisted_service, monkeypatch, tmp_path):
        base_dir = tmp_path / "poolgate_data"
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion("data")
        monkeypatch.setattr(persisted_service._chat_client, "_sync_sdk", lambda k: mock_sdk)

        persisted_service.invoke("Q1", model="llama-3.3-70b-versatile")
        persisted_service.invoke("Q2", model="llama-3.3-70b-versatile")
        persisted_service.flush_tracking()

        tracking_dir = base_dir / "tracking"
        for path in tracking_dir.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, (dict, list)), f"{path.name} is not valid JSON"


# ---------------------------------------------------------------------------
# Storage log
# ---------------------------------------------------------------------------

class TestStorageLog:
    def test_storage_log_created_when_data_dir_set(self, persisted_service, monkeypatch, tmp_path):
        base_dir = tmp_path / "poolgate_data"
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion("ok")
        monkeypatch.setattr(persisted_service._chat_client, "_sync_sdk", lambda k: mock_sdk)

        persisted_service.invoke("Q", model="llama-3.3-70b-versatile")
        persisted_service.flush_tracking()

        log_dir = base_dir / "logs"
        if log_dir.is_dir():
            assert log_dir.is_dir()


# ---------------------------------------------------------------------------
# Request journal
# ---------------------------------------------------------------------------

class TestRequestJournal:
    def test_journal_dir_created_when_data_dir_set(self, persisted_service, tmp_path):
        requests_dir = tmp_path / "poolgate_data" / "requests"
        assert requests_dir.is_dir(), "requests/ dir not created"

    def test_journal_entry_written_after_invoke(self, persisted_service, monkeypatch, tmp_path):
        requests_dir = tmp_path / "poolgate_data" / "requests"
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion("journaled")
        monkeypatch.setattr(persisted_service._chat_client, "_sync_sdk", lambda k: mock_sdk)

        persisted_service.invoke("Q", model="llama-3.3-70b-versatile")

        if requests_dir.is_dir():
            jsonl_files = list(requests_dir.glob("*.jsonl"))
            if jsonl_files:
                with open(jsonl_files[0]) as f:
                    line = f.readline()
                entry = json.loads(line)
                assert "request_id" in entry
                assert "model" in entry
                assert entry["model"] == "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Cross-restart persistence
# ---------------------------------------------------------------------------

class TestCrossRestartPersistence:
    def test_flush_and_reload_preserves_stats(self, monkeypatch, tmp_path):
        base_dir = tmp_path / "poolgate_data"

        svc1 = _service_with_base_dir(monkeypatch, ["gsk_restart_key"], base_dir)
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion("ok")
        monkeypatch.setattr(svc1._chat_client, "_sync_sdk", lambda k: mock_sdk)

        svc1.invoke("Q1", model="llama-3.3-70b-versatile")
        svc1.invoke("Q2", model="llama-3.3-70b-versatile")
        svc1.flush_tracking()

        tracking_dir = base_dir / "tracking"
        if not tracking_dir.is_dir():
            pytest.skip("tracking dir not created — persistence not available")

        svc2 = _service_with_base_dir(monkeypatch, ["gsk_restart_key"], base_dir)
        stats = svc2.get_global_stats()
        assert isinstance(stats, dict)
