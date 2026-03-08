"""
Base instrumentor class.
"""

from abc import ABC, abstractmethod


class BaseInstrumentor(ABC):
    """Base class for all instrumentors."""

    _instrumented = False

    @abstractmethod
    def instrument(self) -> None:
        """Apply instrumentation."""
        pass

    @abstractmethod
    def uninstrument(self) -> None:
        """Remove instrumentation."""
        pass

    @property
    def is_instrumented(self) -> bool:
        """Check if instrumentation is active."""
        return self._instrumented
