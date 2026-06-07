"""Shared runtime policies for resilient batch construction."""

from __future__ import annotations


class ConsecutiveFailureGuard:
    """Track consecutive hard failures and decide when to stop a batch run."""

    def __init__(self, max_consecutive_failures: int) -> None:
        self.max_consecutive_failures = max(0, int(max_consecutive_failures))
        self.current = 0
        self.last_error: str | None = None

    def note_failure(self, message: str) -> bool:
        """Record one failure and return True if the threshold is reached."""

        self.current += 1
        self.last_error = message
        return self.max_consecutive_failures > 0 and self.current >= self.max_consecutive_failures

    def note_non_failure(self) -> None:
        """Reset the failure streak after a success or a deliberate skip."""

        self.current = 0
        self.last_error = None

    def snapshot(self) -> dict[str, int | str | None]:
        return {
            "max_consecutive_failures": self.max_consecutive_failures,
            "current_consecutive_failures": self.current,
            "last_error": self.last_error,
        }
