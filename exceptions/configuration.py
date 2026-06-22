"""
exceptions/configuration.py
────────────────────────────
Exceptions raised during startup when environment variables or
programmatic configuration are invalid.

All four are fatal — they mean the pool cannot be initialised.

Existing
--------
ConfigurationError

EnvironmentParseError       — bad env-var value, e.g. GROQ_MAX_RPM=abc
                              replaces bare ValueError in config.py:59-68
                              and models/base.py:112 (_env_int)
InvalidRateLimitConfigError — ≤0 limit field in ModelRateLimitConfig
                              replaces bare ValueError at models/base.py:76
EmptyKeyPoolError           — no API keys at construction time
                              replaces silent 0-key build at service.py:108-115
"""

from __future__ import annotations

from exceptions.base import GroqServiceError


class ConfigurationError(GroqServiceError):
	"""
	Raised for missing or invalid configuration at startup.

	# No override needed — GroqServiceError.__init__ is sufficient.
	"""


class EnvironmentParseError(ConfigurationError):
	"""
	Raised when an environment variable exists but cannot be parsed into
	the expected type (e.g. GROQ_MAX_RPM=abc fails int()).

	Attributes
	----------
	var_name  : name of the offending environment variable
	raw_value : the string that could not be parsed
	expected  : the Python type that was expected (e.g. int, float)
	"""

	def __init__(
			self,
			message: str,
			var_name: str,
			raw_value: str,
			expected: type,
			request_id: str | None = None,
			) -> None:
		self.var_name = var_name
		self.raw_value = raw_value
		self.expected = expected
		super().__init__(message, request_id)


class InvalidRateLimitConfigError(ConfigurationError):
	"""
	Raised by ModelRateLimitConfig.__post_init__ when any limit field is ≤ 0.

	Attributes
	----------
	field : the limit field that is invalid (e.g. "rpm")
	value : the offending value
	"""

	def __init__(
			self,
			message: str,
			field: str,
			value: int | float,
			request_id: str | None = None,
			) -> None:
		self.field = field
		self.value = value
		super().__init__(message, request_id)


class EmptyKeyPoolError(ConfigurationError):
	"""
	Raised by GroqService.__init__ when the resolved key list is empty.

	Failing fast at construction gives a clearer error at the actual
	misconfiguration site rather than waiting for the first acquire_key call.

	# No override needed — ConfigurationError.__init__ is sufficient.
	"""
