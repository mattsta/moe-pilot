"""Conservative per-process limits for research on a shared workstation.

Estimates are admission controls, not promises about framework allocator peaks.
Keep a host reserve, bound accelerator allocation, and recheck between layers.
"""

import os
import platform
import resource
import subprocess
import tempfile
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import torch
from filelock import FileLock, Timeout

GIB = 1024**3
_COMPUTE_LOCK = FileLock(Path(tempfile.gettempdir()) / f"moe-pilot-compute-{os.getuid()}.lock")


def bounded_inference(function):
    """Allow one cooperating research process per user on the workstation."""

    @wraps(function)
    def execute(*args, **kwargs):
        try:
            with _COMPUTE_LOCK.acquire(timeout=0):
                return function(*args, **kwargs)
        except Timeout as error:
            raise ValueError(
                "Another local MoE job is active; concurrent inference is disabled"
            ) from error

    return execute


def physical_memory_bytes():
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError(
                "Cannot read physical memory; refusing unbudgeted inference"
            ) from error
        return int(result.stdout)
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def available_memory_bytes():
    """Read headroom, including conservatively bounded reclaimable file cache."""
    if platform.system() == "Darwin":
        try:
            pressure = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            if int(pressure.stdout) != 1:
                raise ValueError(
                    "macOS reports memory pressure; research stopped without allocating more memory"
                )
            result = subprocess.run(
                ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True, timeout=2
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError("Cannot read free memory; refusing unbudgeted inference") from error
        lines = result.stdout.splitlines()
        page_size = int(lines[0].split("page size of ")[1].split()[0])
        pages = {}
        for line in lines[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                pages[name] = int(value.strip().rstrip("."))
        # File-backed inactive cache can be reclaimed without compressing an
        # application's anonymous working set. Do not count all inactive pages.
        # Subtract speculative pages from file cache before adding them once.
        speculative = pages.get("Pages speculative", 0)
        file_cache = min(
            pages.get("Pages inactive", 0), max(0, pages.get("File-backed pages", 0) - speculative)
        )
        return page_size * (pages["Pages free"] + speculative + file_cache)
    memory = Path("/proc/meminfo")
    if memory.exists():
        for line in memory.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def peak_process_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


@dataclass(frozen=True)
class ResourcePolicy:
    """Default to an 8 GiB job budget and a 16 GiB host reserve.

    The budget includes a conservative runtime allowance. Raising it is an
    explicit caller decision; automatic device fallback never raises it.
    On small hosts, cap the reserve at one quarter of physical memory.
    """

    max_memory_gib: int = 8
    reserve_memory_gib: int = 16

    def __post_init__(self):
        for name in ("max_memory_gib", "reserve_memory_gib"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(
            self, "reserve_bytes", min(self.reserve_memory_gib * GIB, physical_memory_bytes() // 4)
        )

    def check(self, required_bytes=0):
        available = available_memory_bytes()
        if required_bytes + GIB > self.max_memory_gib * GIB:
            raise ValueError(
                f"Estimated job memory {(required_bytes + GIB) / GIB:.2f} GiB exceeds "
                f"the {self.max_memory_gib} GiB budget; use streaming weights, smaller "
                "batches/condition groups, or explicitly configure a suitable budget"
            )
        if peak_process_bytes() > self.max_memory_gib * GIB:
            raise ValueError("Process memory exceeded the job budget; stop and use a smaller run")
        if available < self.reserve_bytes + required_bytes:
            raise ValueError(
                f"Insufficient free memory ({available / GIB:.2f} GiB available, "
                f"{required_bytes / GIB:.2f} GiB estimated additional allocation) "
                f"to preserve the {self.reserve_bytes / GIB:.1f} GiB "
                "workstation reserve; no CPU fallback or larger allocation was attempted"
            )

    def limit_device(self, device):
        """Bound this process's allocator, without changing another app's limits."""
        budget = self.max_memory_gib * GIB
        if device.type == "mps":
            recommended = torch.mps.recommended_max_memory()
            torch.mps.set_per_process_memory_fraction(min(0.5, budget / recommended))
        elif device.type == "cuda":
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(0.5, budget / total), device)
