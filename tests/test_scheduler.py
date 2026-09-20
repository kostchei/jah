import asyncio
import time

import pytest

from jah.errors import DeadlineExceededError, QueueSaturatedError
from jah.scheduler import BoundedScheduler


def test_deadline_keeps_capacity_reserved_until_thread_finishes() -> None:
    async def scenario() -> None:
        scheduler = BoundedScheduler(maximum_concurrency=1, maximum_queue_size=0)

        with pytest.raises(DeadlineExceededError):
            await scheduler.run(lambda: time.sleep(0.05), timeout_seconds=0.005)
        with pytest.raises(QueueSaturatedError):
            await scheduler.run(lambda: None, timeout_seconds=1.0)
        await asyncio.sleep(0.07)
        value, _ = await scheduler.run(lambda: "done", timeout_seconds=1.0)
        assert value == "done"

    asyncio.run(scenario())

def test_queued_deadline_releases_admission() -> None:
    async def scenario() -> None:
        scheduler = BoundedScheduler(maximum_concurrency=1, maximum_queue_size=1)
        first = asyncio.create_task(
            scheduler.run(lambda: time.sleep(0.05), timeout_seconds=1.0)
        )
        await asyncio.sleep(0.005)
        with pytest.raises(DeadlineExceededError):
            await scheduler.run(lambda: None, timeout_seconds=0.005)
        await first
        value, _ = await scheduler.run(lambda: 42, timeout_seconds=1.0)
        assert value == 42

    asyncio.run(scenario())
