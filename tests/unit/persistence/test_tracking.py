"""Unit tests for tracking/persistence.py — JSON and SQLite backends."""

from __future__ import annotations

import pytest

from poolgate.exceptions.persistence import PersistenceError
from poolgate.tracking.persistence import JSONPersistence, SQLitePersistence


class TestJSONPersistence:

    def test_load_all_on_new_file_returns_empty_dict(self, tmp_path):
        path = str(tmp_path / "usage.json")
        backend = JSONPersistence(path)
        # The constructor creates an empty file, load_all should return {}
        result = backend.load_all()
        assert result == {}

    def test_save_day_and_load_all_round_trips(self, tmp_path):
        path = str(tmp_path / "usage.json")
        backend = JSONPersistence(path)
        backend.save_day("2026-06-01", {"requests": 10, "tokens_in": 500})
        result = backend.load_all()
        assert "2026-06-01" in result
        assert result["2026-06-01"]["requests"] == 10

    def test_save_all_bulk_write(self, tmp_path):
        path = str(tmp_path / "usage.json")
        backend = JSONPersistence(path)
        days = {
            "2026-06-01": {"requests": 5},
            "2026-06-02": {"requests": 8},
        }
        backend.save_all(days)
        result = backend.load_all()
        assert len(result) == 2
        assert result["2026-06-01"]["requests"] == 5
        assert result["2026-06-02"]["requests"] == 8

    def test_save_day_overwrites_existing_entry(self, tmp_path):
        path = str(tmp_path / "usage.json")
        backend = JSONPersistence(path)
        backend.save_day("2026-06-01", {"requests": 5})
        backend.save_day("2026-06-01", {"requests": 99})
        result = backend.load_all()
        assert result["2026-06-01"]["requests"] == 99

    def test_corrupt_json_raises_persistence_error(self, tmp_path):
        path = str(tmp_path / "usage.json")
        # Write valid file first (constructor creates it), then corrupt it
        backend = JSONPersistence(path)
        with open(path, "w") as f:
            f.write("NOT_VALID_JSON{{{{")
        with pytest.raises(PersistenceError) as exc_info:
            backend.load_all()
        assert exc_info.value.backend == "json"

    def test_save_all_then_save_day_merges_correctly(self, tmp_path):
        path = str(tmp_path / "usage.json")
        backend = JSONPersistence(path)
        backend.save_all({"2026-06-01": {"r": 1}})
        backend.save_day("2026-06-02", {"r": 2})
        result = backend.load_all()
        assert "2026-06-01" in result
        assert "2026-06-02" in result

    def test_write_oserror_raises_persistence_error(self, tmp_path, monkeypatch):
        import tempfile as _tempfile

        path = str(tmp_path / "usage.json")
        backend = JSONPersistence(path)

        def _bad_mkstemp(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(_tempfile, "mkstemp", _bad_mkstemp)
        with pytest.raises(PersistenceError) as exc_info:
            backend.save_day("2026-06-01", {"requests": 5})
        assert exc_info.value.backend == "json"

    def test_load_returns_empty_when_file_deleted_after_init(self, tmp_path):
        import os

        path = str(tmp_path / "usage.json")
        backend = JSONPersistence(path)
        os.remove(path)
        result = backend.load_all()
        assert result == {}

    def test_write_fdopen_oserror_raises_persistence_error(self, tmp_path, monkeypatch):
        import os as _os

        path = str(tmp_path / "usage.json")
        backend = JSONPersistence(path)

        def _bad_fdopen(*args, **kwargs):
            raise OSError("write failed mid-file")

        monkeypatch.setattr(_os, "fdopen", _bad_fdopen)
        with pytest.raises(PersistenceError) as exc_info:
            backend.save_day("2026-06-01", {"requests": 5})
        assert exc_info.value.backend == "json"


class TestSQLitePersistence:

    def test_load_all_on_new_db_returns_empty_dict(self, tmp_path):
        path = str(tmp_path / "usage.db")
        backend = SQLitePersistence(path)
        assert backend.load_all() == {}

    def test_save_day_and_load_all_round_trips(self, tmp_path):
        path = str(tmp_path / "usage.db")
        backend = SQLitePersistence(path)
        backend.save_day("2026-06-01", {"requests": 7})
        result = backend.load_all()
        assert "2026-06-01" in result
        assert result["2026-06-01"]["requests"] == 7

    def test_save_all_bulk_upsert(self, tmp_path):
        path = str(tmp_path / "usage.db")
        backend = SQLitePersistence(path)
        days = {
            "2026-06-01": {"requests": 3},
            "2026-06-02": {"requests": 6},
        }
        backend.save_all(days)
        result = backend.load_all()
        assert len(result) == 2
        assert result["2026-06-02"]["requests"] == 6

    def test_upsert_overwrites_existing_row(self, tmp_path):
        path = str(tmp_path / "usage.db")
        backend = SQLitePersistence(path)
        backend.save_day("2026-06-01", {"requests": 1})
        backend.save_day("2026-06-01", {"requests": 42})
        result = backend.load_all()
        assert result["2026-06-01"]["requests"] == 42

    def test_db_file_is_created(self, tmp_path):
        import os

        path = str(tmp_path / "usage.db")
        SQLitePersistence(path)
        assert os.path.exists(path)

    def test_init_sqlite_error_raises_persistence_error(self, tmp_path, monkeypatch):
        import sqlite3 as _sqlite3

        path = str(tmp_path / "usage.db")

        def _bad_connect(*args, **kwargs):
            raise _sqlite3.Error("init failure")

        monkeypatch.setattr(_sqlite3, "connect", _bad_connect)
        with pytest.raises(PersistenceError) as exc_info:
            SQLitePersistence(path)
        assert exc_info.value.backend == "sqlite"

    def test_load_all_sqlite_error_raises_persistence_error(self, tmp_path, monkeypatch):
        import sqlite3 as _sqlite3

        path = str(tmp_path / "usage.db")
        backend = SQLitePersistence(path)  # init succeeds with real sqlite3

        def _bad_connect(*args, **kwargs):
            raise _sqlite3.Error("read failure")

        monkeypatch.setattr(_sqlite3, "connect", _bad_connect)
        with pytest.raises(PersistenceError) as exc_info:
            backend.load_all()
        assert exc_info.value.backend == "sqlite"

    def test_save_day_sqlite_error_raises_persistence_error(self, tmp_path, monkeypatch):
        import sqlite3 as _sqlite3

        path = str(tmp_path / "usage.db")
        backend = SQLitePersistence(path)  # init succeeds

        def _bad_connect(*args, **kwargs):
            raise _sqlite3.Error("write failure")

        monkeypatch.setattr(_sqlite3, "connect", _bad_connect)
        with pytest.raises(PersistenceError) as exc_info:
            backend.save_day("2026-06-01", {"requests": 5})
        assert exc_info.value.backend == "sqlite"

    def test_save_all_sqlite_error_raises_persistence_error(self, tmp_path, monkeypatch):
        import sqlite3 as _sqlite3

        path = str(tmp_path / "usage.db")
        backend = SQLitePersistence(path)  # init succeeds

        def _bad_connect(*args, **kwargs):
            raise _sqlite3.Error("bulk write failure")

        monkeypatch.setattr(_sqlite3, "connect", _bad_connect)
        with pytest.raises(PersistenceError) as exc_info:
            backend.save_all({"2026-06-01": {"requests": 5}})
        assert exc_info.value.backend == "sqlite"
