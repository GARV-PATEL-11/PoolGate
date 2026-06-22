"""
exceptions/output.py
─────────────────────
Exceptions for structured-output parsing failures and session lifecycle.

StructuredOutputError — well-handled; sync + async service.structured()
                        already normalise ValidationError / JSONDecodeError /
                        ValueError into this. Kept as-is.

SessionExpiredError   — raised correctly in session_manager.py:162, but
                        service._resolve_session() does NOT catch it, so it
                        leaks through all 6 public methods uncontextualized.
                        Fix required in service.py (see class docstring).

SessionError — base class so callers can catch all session errors in one
               except clause without catching unrelated GroqServiceErrors
"""

from __future__ import annotations

from exceptions.base import GroqServiceError


class StructuredOutputError(GroqServiceError):
	"""
	Raised when structured output parsing / validation fails after retries.

	Attributes
	----------
	raw_response : the last raw text response from the model; useful for
							   debugging prompt / schema mismatches
	"""

	def __init__(
			self,
			message: str,
			raw_response: str | None = None,
			request_id: str | None = None,
			) -> None:
		self.raw_response = raw_response
		super().__init__(message, request_id)


class SessionError(GroqServiceError):
	"""
	Base class for session-lifecycle exceptions.

	Catch this to handle any session-related failure with a single
	except clause.

	# No override needed — GroqServiceError.__init__ is sufficient.
	"""


class SessionExpiredError(SessionError):
	"""
	Raised when an operation is attempted on an expired session.

	Attributes
	----------
	session_id : the identifier of the expired session
	"""

	def __init__(self, session_id: str, request_id: str | None = None) -> None:
		self.session_id = session_id
		super().__init__(f"Session '{session_id}' has expired", request_id)
