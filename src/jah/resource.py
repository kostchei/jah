"""Resource management and system throttling utilities."""

from __future__ import annotations

import os
import sys
import time
from typing import Any


def set_process_priority_low() -> bool:
    """Set the current process to below normal or idle priority."""
    if sys.platform == "win32":
        try:
            import psutil

            p = psutil.Process(os.getpid())
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return True
        except Exception:
            try:
                import ctypes

                # 0x00004000 = BELOW_NORMAL_PRIORITY_CLASS
                ctypes.windll.kernel32.SetPriorityClass(-1, 0x00004000)
                return True
            except Exception:
                pass
    else:
        try:
            os.nice(10)
            return True
        except Exception:
            pass
    return False


def get_physical_gpu_info() -> tuple[int, int] | None:
    """Query physical GPU used and total memory in bytes via nvidia-smi."""
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2.0,
        )
        parts = out.strip().splitlines()[0].split(",")
        used_mb = float(parts[0].strip())
        total_mb = float(parts[1].strip())
        return int(used_mb * 1024 * 1024), int(total_mb * 1024 * 1024)
    except Exception:
        return None


def configure_resource_limits(
    *,
    max_total_gpu_fraction: float = 0.80,
    max_cpu_fraction: float = 0.80,
    device_id: int = 0,
    below_normal_priority: bool = True,
) -> dict[str, Any]:
    """Configure CPU thread limits, GPU memory allocation ceiling, and process priority.

    Ensures that total GPU usage (including other running applications like games)
    does not exceed `max_total_gpu_fraction` of physical capacity, and that CPU
    threads are capped at `max_cpu_fraction` of physical cores.
    """
    priority_set = False
    if below_normal_priority:
        priority_set = set_process_priority_low()

    limit_threads = None
    try:
        import torch

        num_cores = os.cpu_count() or 4
        limit_threads = max(1, int(num_cores * max_cpu_fraction))
        torch.set_num_threads(limit_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, limit_threads // 2))
    except Exception:
        pass

    # Enable expandable segments to avoid virtual address space fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    gpu_fraction_applied = None
    gpu_headroom_gb = None
    try:
        import torch

        if torch.cuda.is_available():
            phys_info = get_physical_gpu_info()
            if phys_info is not None:
                phys_used, phys_total = phys_info
                allocated_by_self = torch.cuda.memory_allocated(device_id)
                used_other_bytes = max(0, phys_used - allocated_by_self)
                total_bytes = phys_total
            else:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
                allocated_by_self = torch.cuda.memory_allocated(device_id)
                used_other_bytes = max(0, (total_bytes - free_bytes) - allocated_by_self)

            max_allowed_bytes = max_total_gpu_fraction * total_bytes
            allowed_for_proc_bytes = max(0.0, max_allowed_bytes - used_other_bytes)
            proc_fraction = allowed_for_proc_bytes / total_bytes
            # Never let the process claim a minimum fraction that would push combined
            # use above the configured device budget. A too-small cap should fail the
            # model load visibly rather than silently competing with other GPU apps.
            proc_fraction = min(max_total_gpu_fraction, max(0.0, proc_fraction))
            torch.cuda.set_per_process_memory_fraction(proc_fraction, device_id)
            gpu_fraction_applied = round(proc_fraction, 4)
            gpu_headroom_gb = round(allowed_for_proc_bytes / (1024**3), 2)
    except Exception:
        pass

    return {
        "priority_low": priority_set,
        "cpu_threads": limit_threads,
        "gpu_memory_fraction": gpu_fraction_applied,
        "gpu_headroom_gb": gpu_headroom_gb,
    }


def throttle(delay_ms: float = 0.0) -> None:
    """Sleep for delay_ms milliseconds to yield execution."""
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
