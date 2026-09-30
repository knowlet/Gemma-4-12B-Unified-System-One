"""Errors whose provenance is known at a request or backend boundary.

Ordinary ValueError/RuntimeError exceptions remain internal failures; callers
must opt into these types only when they can identify the source of the error.
"""


class RequestValidationError(ValueError):
    """A validly parsed request cannot be processed as supplied (HTTP 422)."""


class BackendResponseError(ValueError):
    """A backend returned an unsuccessful or invalid response (HTTP 502)."""


class BackendTransportError(RuntimeError):
    """A backend could not be reached or its exchange failed (HTTP 502)."""


class BackendTimeoutError(BackendTransportError):
    """A backend exchange exceeded its timeout (HTTP 504)."""
