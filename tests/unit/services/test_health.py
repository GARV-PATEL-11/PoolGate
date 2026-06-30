"""Unit tests for services/health_service.py."""

from __future__ import annotations

from poolgate.schemas.common.ops import HealthStatus
from poolgate.services.health import HealthService


def _active_key(key_id: str = "key_0") -> dict:
    return {"key_id": key_id, "status": "active", "requests_per_minute": 0}


def _failed_key(key_id: str = "key_1") -> dict:
    return {"key_id": key_id, "status": "failed", "requests_per_minute": 0}


def _disabled_key(key_id: str = "key_2") -> dict:
    return {"key_id": key_id, "status": "disabled", "requests_per_minute": 0}


class TestHealthServiceSnapshot:

    def test_all_active_keys_returns_healthy(self):
        svc = HealthService()
        result = svc.snapshot(key_status=[_active_key("k0"), _active_key("k1")])
        assert result.status == "healthy"

    def test_one_failed_key_returns_degraded(self):
        svc = HealthService()
        result = svc.snapshot(key_status=[_active_key(), _failed_key()])
        assert result.status == "degraded"

    def test_one_disabled_key_returns_degraded(self):
        svc = HealthService()
        result = svc.snapshot(key_status=[_active_key(), _disabled_key()])
        assert result.status == "degraded"

    def test_all_keys_disabled_returns_unhealthy(self):
        svc = HealthService()
        result = svc.snapshot(key_status=[_disabled_key("k0"), _disabled_key("k1")])
        assert result.status == "unhealthy"

    def test_zero_keys_returns_unhealthy(self):
        svc = HealthService()
        result = svc.snapshot(key_status=[])
        assert result.status == "unhealthy"

    def test_all_failed_returns_unhealthy(self):
        svc = HealthService()
        result = svc.snapshot(key_status=[_failed_key("k0"), _failed_key("k1")])
        assert result.status == "unhealthy"

    def test_active_keys_count_is_correct(self):
        svc = HealthService()
        result = svc.snapshot(key_status=[_active_key("k0"), _active_key("k1"), _failed_key("k2")])
        assert result.active_keys == 2

    def test_disabled_keys_count_is_correct(self):
        svc = HealthService()
        result = svc.snapshot(key_status=[_active_key(), _disabled_key(), _disabled_key("k2")])
        assert result.disabled_keys == 2

    def test_uptime_seconds_is_non_negative(self):
        svc = HealthService()
        result = svc.snapshot(key_status=[_active_key()])
        assert result.uptime_seconds >= 0.0

    def test_returns_health_status_instance(self):
        svc = HealthService()
        result = svc.snapshot(key_status=[_active_key()])
        assert isinstance(result, HealthStatus)

    def test_version_propagated(self):
        svc = HealthService(version="1.2.3")
        result = svc.snapshot(key_status=[_active_key()])
        assert result.version == "1.2.3"
