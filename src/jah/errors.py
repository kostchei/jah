"""Domain errors shared by the engine, scheduler, and HTTP layer."""


class JahError(Exception):
    """Base class for expected service failures."""


class InferenceUnavailableError(JahError):
    """The configured inference artifact cannot serve work."""


class QueueSaturatedError(JahError):
    """The bounded scheduler has no capacity for another request."""


class DeadlineExceededError(JahError):
    """A request could not complete before its deadline."""
