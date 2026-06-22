"""Unit tests for path configuration modules.

Covers:
  A) Root-level path_config.py (Paths class with Logs/Requests/Tracking)
  B) core/path_config.py (PathConfig frozen dataclass)
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from core.path_config import PathConfig


# ===========================================================================
# A) Root-level path_config.py — Paths class
# ===========================================================================


class TestPathsRoot:

	def test_root_is_path_instance(self):
		assert isinstance(Paths.ROOT, Path)

	def test_root_is_not_inside_examples(self):
		assert "examples" not in str(Paths.ROOT)

	def test_data_is_path_instance(self):
		assert isinstance(Paths.DATA, Path)

	def test_data_is_child_of_root(self):
		# DATA must be inside ROOT (or be ROOT itself)
		assert str(Paths.DATA).startswith(str(Paths.ROOT))


class TestPathsLogs:

	def test_general_log_is_path(self):
		assert isinstance(Paths.Logs.GENERAL, Path)

	def test_error_log_is_path(self):
		assert isinstance(Paths.Logs.ERROR, Path)

	def test_info_log_is_path(self):
		assert isinstance(Paths.Logs.INFO, Path)

	def test_request_log_is_path(self):
		assert isinstance(Paths.Logs.REQUEST, Path)

	def test_log_dir_is_path(self):
		assert isinstance(Paths.Logs.DIR, Path)

	def test_logs_under_data(self):
		assert str(Paths.Logs.GENERAL).startswith(str(Paths.DATA))

	def test_log_filenames(self):
		assert Paths.Logs.GENERAL.name == "general.log"
		assert Paths.Logs.ERROR.name == "error.log"
		assert Paths.Logs.REQUEST.name == "request.log"


class TestPathsRequests:

	def test_dir_is_path(self):
		assert isinstance(Paths.Requests.DIR, Path)

	def test_daily_returns_correct_path(self):
		p = Paths.Requests.daily("2026-06-22")
		assert str(p).endswith("requests/2026-06-22.jsonl")

	def test_daily_under_data(self):
		p = Paths.Requests.daily("2026-01-01")
		assert str(p).startswith(str(Paths.DATA))

	def test_today_ends_with_today_iso(self):
		p = Paths.Requests.today()
		today = date.today().isoformat()
		assert str(p).endswith(f"{today}.jsonl")


class TestPathsTracking:

	def test_account_is_path(self):
		assert isinstance(Paths.Tracking.ACCOUNT, Path)

	def test_tokens_is_path(self):
		assert isinstance(Paths.Tracking.TOKENS, Path)

	def test_usage_is_path(self):
		assert isinstance(Paths.Tracking.USAGE, Path)

	def test_tracking_under_data(self):
		assert str(Paths.Tracking.ACCOUNT).startswith(str(Paths.DATA))

	def test_tracking_filenames(self):
		assert Paths.Tracking.ACCOUNT.name == "account.json"
		assert Paths.Tracking.TOKENS.name == "tokens.json"
		assert Paths.Tracking.USAGE.name == "usage.json"


class TestPathsEnsureDirs:

	def test_ensure_dirs_does_not_raise(self):
		# We can't easily redirect the real Paths dirs without patching,
		# but ensure_dirs() uses exist_ok=True so it should never raise
		# even if dirs already exist.
		Paths.ensure_dirs()


class TestPathsSummary:

	def test_summary_is_non_empty_string(self):
		s = Paths.summary()
		assert isinstance(s, str)
		assert len(s) > 0

	def test_summary_contains_root(self):
		assert str(Paths.ROOT) in Paths.summary()


class TestFlatAliases:

	def test_log_error_equals_paths_logs_error(self):
		assert LOG_ERROR == Paths.Logs.ERROR

	def test_log_general_equals_paths_logs_general(self):
		assert LOG_GENERAL == Paths.Logs.GENERAL

	def test_tracking_account_equals_paths_tracking_account(self):
		assert TRACKING_ACCOUNT == Paths.Tracking.ACCOUNT

	def test_tracking_usage_equals_paths_tracking_usage(self):
		assert TRACKING_USAGE == Paths.Tracking.USAGE


# ===========================================================================
# B) core/path_config.py — PathConfig frozen dataclass
# ===========================================================================


class TestPathConfigDefaults:

	def test_no_args_all_none(self):
		p = PathConfig()
		assert p.data_dir is None
		assert p.log_dir is None
		assert p.tracking_dir is None
		assert p.requests_dir is None
		assert p.general_log is None
		assert p.error_log is None
		assert p.request_log is None

	def test_persistence_disabled_when_no_data_dir(self):
		p = PathConfig()
		assert p.persistence_enabled is False

	def test_logging_disabled_when_no_log_dir(self):
		p = PathConfig()
		assert p.logging_enabled is False


class TestPathConfigWithDataDir:

	def test_tracking_dir_computed(self):
		p = PathConfig(data_dir="/tmp/test")
		assert p.tracking_dir == "/tmp/test/tracking"

	def test_log_dir_computed(self):
		p = PathConfig(data_dir="/tmp/test")
		assert p.log_dir == "/tmp/test/logs"

	def test_requests_dir_computed(self):
		p = PathConfig(data_dir="/tmp/test")
		assert p.requests_dir == "/tmp/test/requests"

	def test_persistence_enabled_when_data_dir_set(self):
		p = PathConfig(data_dir="/tmp/test")
		assert p.persistence_enabled is True

	def test_logging_enabled_when_data_dir_set(self):
		p = PathConfig(data_dir="/tmp/test")
		assert p.logging_enabled is True

	def test_log_files_under_log_dir(self):
		p = PathConfig(data_dir="/tmp/test")
		assert p.general_log is not None
		assert p.general_log.startswith(p.log_dir)
		assert p.error_log is not None
		assert p.error_log.startswith(p.log_dir)

	def test_tracking_json_files_under_tracking_dir(self):
		p = PathConfig(data_dir="/tmp/test")
		assert p.usage_json is not None
		assert p.usage_json.startswith(p.tracking_dir)
		assert p.tokens_json is not None
		assert p.account_json is not None


class TestPathConfigLogDirOverride:

	def test_override_replaces_computed_log_dir(self):
		p = PathConfig(data_dir="/tmp/test", log_dir_override="/tmp/logs")
		assert p.log_dir == "/tmp/logs"

	def test_override_without_data_dir(self):
		p = PathConfig(log_dir_override="/tmp/logs")
		assert p.log_dir == "/tmp/logs"
		assert p.logging_enabled is True

	def test_log_files_under_overridden_log_dir(self):
		p = PathConfig(data_dir="/tmp/test", log_dir_override="/tmp/logs")
		assert p.general_log is not None
		assert p.general_log.startswith("/tmp/logs")


class TestPathConfigEnsureDirs:

	def test_ensure_dirs_creates_directories(self, tmp_path):
		p = PathConfig(data_dir=str(tmp_path / "data"))
		p.ensure_dirs()
		assert os.path.isdir(str(tmp_path / "data" / "logs"))
		assert os.path.isdir(str(tmp_path / "data" / "tracking"))
		assert os.path.isdir(str(tmp_path / "data" / "requests"))

	def test_ensure_dirs_noop_when_no_paths(self):
		p = PathConfig()  # all None
		p.ensure_dirs()  # must not raise

	def test_ensure_dirs_idempotent(self, tmp_path):
		p = PathConfig(data_dir=str(tmp_path / "data"))
		p.ensure_dirs()
		p.ensure_dirs()  # calling twice must not raise

	def test_ensure_dirs_with_log_dir_override(self, tmp_path):
		log_dir = str(tmp_path / "logs")
		p = PathConfig(log_dir_override=log_dir)
		p.ensure_dirs()
		assert os.path.isdir(log_dir)
