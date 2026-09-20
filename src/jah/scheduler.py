"""Bounded asynchronous admission control for blocking inference engines."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from jah.errors import DeadlineExceededError, QueueSaturatedError

T = TypeVar("T")


@dataclass(frozen=True)
class ScheduledResult:
    value: object
    queue_ms: float


class BoundedScheduler:
    """Bound waiting work and execute blocking calls outside the event loop."""

    def __init__(self, *, maximum_concurrency: int = 1, maximum_queue_size: int = 32) -> None:
        if maximum_concurrency < 1:
            raise ValueError("maximum_concurrency must be positive")
        if maximum_queue_size < 0:
            raise ValueError("maximum_queue_size cannot be negative")
        self.maximum_concurrency = maximum_concurrency
        self.maximum_queue_size = maximum_queue_size
        self._semaphore = asyncio.Semaphore(maximum_concurrency)
        self._state_lock = threading.Lock()
        self._admitted = 0

    async def run(self, operation: Callable[[], T], *, timeout_seconds: float) -> tuple[T, float]:
        if timeout_seconds <= 0:
            raise DeadlineExceededError("deadline must be positive")
        capacity = self.maximum_concurrency + self.maximum_queue_size
        with self._state_lock:
            if self._admitted >= capacity:
                raise QueueSaturatedError("inference queue is saturated")
            self._admitted += 1

        queued_at = time.perf_counter()
        acquired = False
        deferred_cleanup = False
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout_seconds)
            acquired = True
            queue_ms = (time.perf_counter() - queued_at) * 1_000
            remaining = timeout_seconds - (time.perf_counter() - queued_at)
            if remaining <= 0:
                raise DeadlineExceededError("request deadline expired in the queue")

            task = asyncio.create_task(asyncio.to_thread(operation))
            try:
                value = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
            except (TimeoutError, asyncio.CancelledError):
                # A Python thread cannot be force-cancelled safely. Keep its capacity reserved
                # until it exits, while allowing the HTTP request to finish immediately.
                deferred_cleanup = True
                task.add_done_callback(self._finish_deferred)
                raise
            return value, queue_ms
        except TimeoutError as exc:
            raise DeadlineExceededError("request deadline expired") from exc
        finally:
            if not deferred_cleanup:
                if acquired:
                    self._semaphore.release()
                with self._state_lock:
                    self._admitted -= 1

    def _release_admission(self) -> None:
        self._semaphore.release()
        with self._state_lock:
            self._admitted -= 1

    def _finish_deferred(self, task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()  # retrieve a late backend error after the caller has departed
        self._release_admission()
