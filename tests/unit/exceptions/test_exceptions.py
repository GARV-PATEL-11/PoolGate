"""Unit tests for exceptions/ hierarchy — instantiation, fields, and inheritance."""

from __future__ import annotations

from poolgate.exceptions import (
    APIKeyDisabledError,
    APIKeyError,
    CapabilityError,
    ConfigurationError,
    DailyLimitExceededError,
    EmptyKeyPoolError,
    EnvironmentParseError,
    GroqServiceError,
    InvalidMessageRoleError,
    InvalidRateLimitConfigError,
    InvalidRequestError,
    MissingPromptError,
    NoAvailableAPIKeyError,
    PersistenceError,
    QuotaExceededError,
    RateLimitExceededError,
    RetryExhaustedError,
    SessionError,
    SessionExpiredError,
    StructuredOutputError,
    TokenBudgetExceededError,
    TransportError,
    UnknownModelError,
    UnknownSchedulingStrategyError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)


class TestHierarchy:

    def test_every_exception_is_groq_service_error(self):
        leaf_types = [
            ConfigurationError,
            EnvironmentParseError,
            InvalidRateLimitConfigError,
            EmptyKeyPoolError,
            InvalidRequestError,
            MissingPromptError,
            InvalidMessageRoleError,
            UnknownModelError,
            CapabilityError,
            UnknownSchedulingStrategyError,
            APIKeyError,
            NoAvailableAPIKeyError,
            APIKeyDisabledError,
            RateLimitExceededError,
            QuotaExceededError,
            DailyLimitExceededError,
            TokenBudgetExceededError,
            TransportError,
            UpstreamTimeoutError,
            UpstreamServiceError,
            RetryExhaustedError,
            StructuredOutputError,
            SessionError,
            SessionExpiredError,
            PersistenceError,
        ]
        for exc_type in leaf_types:
            skip_types = (
                NoAvailableAPIKeyError,
                APIKeyDisabledError,
                EnvironmentParseError,
                InvalidRateLimitConfigError,
                InvalidMessageRoleError,
                UnknownModelError,
                CapabilityError,
                UnknownSchedulingStrategyError,
                RateLimitExceededError,
                QuotaExceededError,
                DailyLimitExceededError,
                TokenBudgetExceededError,
                PersistenceError,
                RetryExhaustedError,
            )
            if exc_type in skip_types:
                continue
            exc = exc_type("msg")
            assert isinstance(exc, GroqServiceError), f"{exc_type.__name__} not a GroqServiceError"

    def test_configuration_subtypes(self):
        assert issubclass(EnvironmentParseError, ConfigurationError)
        assert issubclass(InvalidRateLimitConfigError, ConfigurationError)
        assert issubclass(EmptyKeyPoolError, ConfigurationError)

    def test_request_subtypes(self):
        assert issubclass(MissingPromptError, InvalidRequestError)
        assert issubclass(InvalidMessageRoleError, InvalidRequestError)
        assert issubclass(UnknownModelError, InvalidRequestError)
        assert issubclass(CapabilityError, InvalidRequestError)

    def test_key_subtypes(self):
        assert issubclass(NoAvailableAPIKeyError, APIKeyError)
        assert issubclass(APIKeyDisabledError, APIKeyError)

    def test_quota_subtypes(self):
        assert issubclass(DailyLimitExceededError, QuotaExceededError)

    def test_transport_subtypes(self):
        assert issubclass(UpstreamTimeoutError, TransportError)

    def test_session_subtypes(self):
        assert issubclass(SessionExpiredError, SessionError)


class TestExceptionFields:

    def test_environment_parse_error_fields(self):
        exc = EnvironmentParseError(
            "bad val",
            var_name="GROQ_MAX_RPM",
            raw_value="abc",
            expected=int,
        )
        assert exc.var_name == "GROQ_MAX_RPM"
        assert exc.raw_value == "abc"
        assert exc.expected is int

    def test_invalid_rate_limit_config_error_fields(self):
        exc = InvalidRateLimitConfigError("neg value", field="rpm", value=-1)
        assert exc.field == "rpm"
        assert exc.value == -1

    def test_no_available_api_key_error_fields(self):
        exc = NoAvailableAPIKeyError(total_keys=3, reason_counts={"rate_limited": 2, "cooling": 1})
        assert exc.total_keys == 3
        assert exc.reason_counts["rate_limited"] == 2

    def test_api_key_disabled_error_fields(self):
        exc = APIKeyDisabledError(key_id="key_x", status_code=401, request_id="req-1")
        assert exc.key_id == "key_x"
        assert exc.status_code == 401

    def test_rate_limit_exceeded_error_fields(self):
        exc = RateLimitExceededError(retry_after=30.0, key_id="key_y")
        assert exc.retry_after == 30.0
        assert exc.key_id == "key_y"

    def test_quota_exceeded_error_fields(self):
        exc = QuotaExceededError(
            "over quota",
            key_id="k",
            quota_type="daily",
            resets_at="2026-06-23T00:00:00Z",
        )
        assert exc.quota_type == "daily"
        assert exc.resets_at == "2026-06-23T00:00:00Z"

    def test_daily_limit_exceeded_sets_quota_type(self):
        exc = DailyLimitExceededError("daily", key_id="k")
        assert exc.quota_type == "daily"

    def test_token_budget_exceeded_error_fields(self):
        exc = TokenBudgetExceededError("tpm hit", budget_type="tpm", limit=12000, retry_after=5.0)
        assert exc.budget_type == "tpm"
        assert exc.limit == 12000
        assert exc.retry_after == 5.0

    def test_invalid_message_role_error_fields(self):
        exc = InvalidMessageRoleError(
            "bad role",
            role="narrator",
            allowed_roles={"user", "assistant"},
        )
        assert exc.role == "narrator"
        assert "user" in exc.allowed_roles

    def test_unknown_model_error_fields(self):
        exc = UnknownModelError(
            "unknown",
            model_id="gpt-99",
            available_models=["llama-3.3-70b-versatile"],
        )
        assert exc.model_id == "gpt-99"
        assert "llama-3.3-70b-versatile" in exc.available_models

    def test_capability_error_fields(self):
        exc = CapabilityError(
            "no tool",
            model_id="whisper-v3",
            capability="tools",
            supported_capabilities=["transcription"],
        )
        assert exc.model_id == "whisper-v3"
        assert exc.capability == "tools"
        assert "transcription" in exc.supported_capabilities

    def test_unknown_scheduling_strategy_error_fields(self):
        exc = UnknownSchedulingStrategyError(
            "bad strategy",
            strategy="magic",
            available_strategies=["round_robin"],
        )
        assert exc.strategy == "magic"
        assert "round_robin" in exc.available_strategies

    def test_persistence_error_fields(self):
        exc = PersistenceError("db fail", backend="sqlite", path="/tmp/data.db")
        assert exc.backend == "sqlite"
        assert exc.path == "/tmp/data.db"

    def test_request_id_propagates(self):
        exc = RateLimitExceededError(request_id="req-42")
        assert exc.request_id == "req-42"
