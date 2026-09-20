"""Tests for resource management, chunking, and resume functionality."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jah.resource import configure_resource_limits, set_process_priority_low, throttle


def test_set_process_priority_low() -> None:
    # Should execute without error on any OS
    result = set_process_priority_low()
    assert isinstance(result, bool)


def test_configure_resource_limits() -> None:
    res = configure_resource_limits(
        max_total_gpu_fraction=0.80,
        max_cpu_fraction=0.80,
        below_normal_priority=True,
    )
    assert "cpu_threads" in res
    assert "gpu_memory_fraction" in res
    assert "priority_low" in res
    assert res["cpu_threads"] is not None
    assert res["cpu_threads"] >= 1


def test_throttle() -> None:
    import time

    start = time.perf_counter()
    throttle(10.0)  # 10ms
    elapsed = (time.perf_counter() - start) * 1000
    assert elapsed >= 8.0  # At least ~10ms
