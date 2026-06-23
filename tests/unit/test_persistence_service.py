"""Unit tests for services/persistence_service.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from services.persistence_service import PersistenceService
from tracking.persistence import JSONPersistence, SQLitePersistence


class TestPersistenceServiceConstructors:

    def test_json_constructor_creates_service(self, tmp_path):
        path = str(tmp_path / "usage.json")
        svc = PersistenceService.json(path)
        assert isinstance(svc, PersistenceService)
        assert isinstance(svc._backend, JSONPersistence)

    def test_sqlite_constructor_creates_service(self, tmp_path):
        path = str(tmp_path / "usage.db")
        svc = PersistenceService.sqlite(path)
        assert isinstance(svc, PersistenceService)
        assert isinstance(svc._backend, SQLitePersistence)

    def test_default_constructor_uses_json_backend(self, tmp_path, monkeypatch):
        # Default constructor creates JSONPersistence("usage.json") in CWD;
        # patch to avoid writing to CWD during tests.
        monkeypatch.chdir(tmp_path)
        svc = PersistenceService()
        assert isinstance(svc._backend, JSONPersistence)


class TestLoadTracker:

    def test_load_tracker_calls_backend_load_all(self, tmp_path):
        path = str(tmp_path / "test.json")
        svc = PersistenceService.json(path)

        mock_tracker = MagicMock()
        mock_tracker.export_days.return_value = {}
        mock_tracker.load_days.return_value = None

        days = svc.load_tracker(mock_tracker)
        mock_tracker.load_days.assert_called_once()
        assert isinstance(days, dict)

    def test_load_tracker_passes_loaded_data_to_tracker(self, tmp_path):
        path = str(tmp_path / "test.json")
        svc = PersistenceService.json(path)
        # Pre-populate the backend with a day
        svc.save_day("2026-06-01", {"requests": 5})

        mock_tracker = MagicMock()
        svc.load_tracker(mock_tracker)
        call_args = mock_tracker.load_days.call_args[0][0]
        assert "2026-06-01" in call_args


class TestFlushTracker:

    def test_flush_tracker_calls_export_days_and_saves(self, tmp_path):
        path = str(tmp_path / "test.json")
        svc = PersistenceService.json(path)

        mock_tracker = MagicMock()
        mock_tracker.export_days.return_value = {"2026-06-01": {"requests": 3}}
        svc.flush_tracker(mock_tracker)

        mock_tracker.export_days.assert_called_once()
        # Verify the data was actually persisted
        result = svc.load_all()
        assert "2026-06-01" in result
