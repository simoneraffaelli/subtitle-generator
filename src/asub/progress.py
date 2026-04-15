"""Terminal progress utilities (spinner + inline status)."""

from __future__ import annotations

import sys
import threading


class Spinner:
    """A context-manager that shows an animated spinner with a message.

    Usage::

        with Spinner("Loading model"):
            do_slow_work()

    The spinner runs in a background thread and clears itself on exit.
    """

    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, message: str = "") -> None:
        self._message = message
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_line_len = 0

    # -- public helpers for updating the message mid-spin --

    def update(self, message: str) -> None:
        """Change the displayed message while the spinner is running."""
        self._message = message

    # -- context manager --

    def __enter__(self) -> Spinner:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._clear_line()

    # -- internals --

    def _spin(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            frame = self._FRAMES[idx % len(self._FRAMES)]
            line = f"\r  {frame} {self._message}"
            # Pad with spaces to overwrite any previous longer line
            padded = line.ljust(self._last_line_len)
            sys.stderr.write(padded)
            sys.stderr.flush()
            self._last_line_len = len(line)
            idx += 1
            self._stop_event.wait(0.08)

    def _clear_line(self) -> None:
        sys.stderr.write("\r" + " " * self._last_line_len + "\r")
        sys.stderr.flush()
