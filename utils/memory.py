"""Resident-memory probes (RQ1 reports memory, not just speed).

Pure standard library -- no psutil dependency. Current RSS is read from
``/proc/self/statm`` on Linux; peak RSS from :func:`resource.getrusage`. On
platforms without ``/proc`` the current-RSS probe falls back to the peak.
"""

from __future__ import annotations

import os
import resource

__all__ = ["current_rss_bytes", "peak_rss_bytes"]

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def current_rss_bytes() -> int:
    """Current resident set size in bytes (best-effort)."""
    try:
        with open("/proc/self/statm", "r") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * _PAGE_SIZE
    except (OSError, IndexError, ValueError):
        return peak_rss_bytes()


def peak_rss_bytes() -> int:
    """Peak resident set size in bytes since process start.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS; we assume the Linux
    convention (the reference platform) and scale accordingly.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
