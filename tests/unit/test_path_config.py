"""Unit tests for core/path_config.py (PathConfig dataclass)."""

from __future__ import annotations

from pathlib import Path

from core.path_config import _DEFAULT_BASE_DIR, PathConfig


class TestPathConfigDisabled:
	"""PathConfig(base_dir=None) disables all I/O."""

	def test_base_dir_is_none(self):
		assert PathConfig(base_dir=None).base_dir is None

	def test_persistence_disabled(self):
		assert PathConfig(base_dir=None).persistence_enabled is False

	def test_logging_disabled(self):
		assert PathConfig(base_dir=None).logging_enabled is False

	def test_all_dirs_none(self):
		p = PathConfig(base_dir=None)
		assert p.log_dir is None
		assert p.tracking_dir is None
		assert p.requests_dir is None
		assert p.audio_dir is None

	def test_all_log_files_none(self):
		p = PathConfig(base_dir=None)
		assert p.general_log is None
		assert p.error_log is None
		assert p.request_log is None
		assert p.response_log is None
		assert p.trace_log is None
		assert p.tool_calls_log is None
		assert p.performance_log is None
		assert p.storage_log is None
		assert p.debug_log is None

	def test_all_tracking_files_none(self):
		p = PathConfig(base_dir=None)
		assert p.usage_json is None
		assert p.tokens_json is None
		assert p.account_json is None


class TestPathConfigDefault:
	"""PathConfig() uses the hardcoded _DEFAULT_BASE_DIR."""

	def test_default_base_dir_is_path(self):
		assert isinstance(PathConfig().base_dir, Path)

	def test_default_equals_module_constant(self):
		assert PathConfig().base_dir == _DEFAULT_BASE_DIR

	def test_persistence_enabled(self):
		assert PathConfig().persistence_enabled is True

	def test_logging_enabled(self):
		assert PathConfig().logging_enabled is True


class TestPathConfigWithBaseDir:
	"""PathConfig(base_dir=Path(...)) derives all paths from base_dir."""

	def test_tracking_dir_is_path(self, tmp_path):
		p = PathConfig(base_dir=tmp_path)
		assert isinstance(p.tracking_dir, Path)
		assert p.tracking_dir == tmp_path / "tracking"

	def test_log_dir_is_path(self, tmp_path):
		p = PathConfig(base_dir=tmp_path)
		assert p.log_dir == tmp_path / "logs"

	def test_requests_dir_is_path(self, tmp_path):
		p = PathConfig(base_dir=tmp_path)
		assert p.requests_dir == tmp_path / "requests"

	def test_audio_dir_is_path(self, tmp_path):
		p = PathConfig(base_dir=tmp_path)
		assert p.audio_dir == tmp_path / "audio"

	def test_persistence_enabled(self, tmp_path):
		assert PathConfig(base_dir=tmp_path).persistence_enabled is True

	def test_logging_enabled(self, tmp_path):
		assert PathConfig(base_dir=tmp_path).logging_enabled is True

	def test_log_files_under_log_dir(self, tmp_path):
		p = PathConfig(base_dir=tmp_path)
		log_dir = tmp_path / "logs"
		assert p.general_log == log_dir / "general.log"
		assert p.error_log == log_dir / "error.log"
		assert p.request_log == log_dir / "request.log"
		assert p.response_log == log_dir / "response.log"
		assert p.trace_log == log_dir / "trace.log"
		assert p.tool_calls_log == log_dir / "tool_calls.log"
		assert p.performance_log == log_dir / "performance.log"
		assert p.storage_log == log_dir / "storage.log"
		assert p.debug_log == log_dir / "debug.log"

	def test_tracking_files_under_tracking_dir(self, tmp_path):
		p = PathConfig(base_dir=tmp_path)
		tracking_dir = tmp_path / "tracking"
		assert p.usage_json == tracking_dir / "usage.json"
		assert p.tokens_json == tracking_dir / "tokens.json"
		assert p.account_json == tracking_dir / "account.json"

	def test_all_paths_are_path_instances(self, tmp_path):
		p = PathConfig(base_dir=tmp_path)
		path_props = [
			p.log_dir, p.tracking_dir, p.requests_dir, p.audio_dir,
			p.general_log, p.error_log, p.request_log, p.response_log,
			p.trace_log, p.tool_calls_log, p.performance_log, p.storage_log,
			p.debug_log, p.usage_json, p.tokens_json, p.account_json,
			]
		for prop in path_props:
			assert isinstance(prop, Path), f"Expected Path, got {type(prop)}: {prop}"


class TestPathConfigLogDirOverride:

	def test_override_replaces_computed_log_dir(self, tmp_path):
		log_override = tmp_path / "custom_logs"
		p = PathConfig(base_dir=tmp_path, log_dir_override=log_override)
		assert p.log_dir == log_override

	def test_override_without_base_dir(self, tmp_path):
		log_override = tmp_path / "logs"
		p = PathConfig(base_dir=None, log_dir_override=log_override)
		assert p.log_dir == log_override
		assert p.logging_enabled is True
		assert p.persistence_enabled is False

	def test_log_files_under_overridden_log_dir(self, tmp_path):
		log_override = tmp_path / "custom_logs"
		p = PathConfig(base_dir=tmp_path, log_dir_override=log_override)
		assert p.general_log == log_override / "general.log"


class TestPathConfigEnsureDirs:

	def test_creates_expected_directories(self, tmp_path):
		p = PathConfig(base_dir=tmp_path / "data")
		p.ensure_dirs()
		assert (tmp_path / "data" / "logs").is_dir()
		assert (tmp_path / "data" / "tracking").is_dir()
		assert (tmp_path / "data" / "requests").is_dir()
		assert (tmp_path / "data" / "audio").is_dir()

	def test_noop_when_base_dir_none(self):
		PathConfig(base_dir=None).ensure_dirs()  # must not raise

	def test_idempotent(self, tmp_path):
		p = PathConfig(base_dir=tmp_path / "data")
		p.ensure_dirs()
		p.ensure_dirs()  # calling twice must not raise

	def test_with_log_dir_override(self, tmp_path):
		log_dir = tmp_path / "logs"
		p = PathConfig(base_dir=None, log_dir_override=log_dir)
		p.ensure_dirs()
		assert log_dir.is_dir()
