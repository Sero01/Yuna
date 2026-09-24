"""Temporary latency instrumentation for the mic-audio-end -> first-audio-out path.

THROWAWAY: this module exists to measure where a conversation turn spends its
time. Delete it (and its call sites) once the numbers are in.

Single-user assumption: one global probe, one turn at a time.
"""

import time
from typing import List, Optional, Set, Tuple

from loguru import logger


class LatencyProbe:
    """Records monotonic marks across a single conversation turn."""

    def __init__(self) -> None:
        self._t0: Optional[float] = None
        self._marks: List[Tuple[str, float]] = []
        self._seen: Set[str] = set()

    def start(self, label: str) -> None:
        self._t0 = time.monotonic()
        self._marks = []
        self._seen = set()
        logger.info(f"[LATENCY] ===== turn start ({label}) =====")

    def mark(self, label: str, once: bool = True) -> None:
        """Record a stage boundary. `once=True` keeps only the first hit."""
        if self._t0 is None:
            return
        if once:
            if label in self._seen:
                return
            self._seen.add(label)
        self._marks.append((label, time.monotonic() - self._t0))

    def report(self) -> None:
        """Log the per-stage breakdown and disarm until the next start()."""
        if self._t0 is None or not self._marks:
            return
        lines = ["[LATENCY] turn breakdown (delta / cumulative since turn start):"]
        prev = 0.0
        for label, t in self._marks:
            lines.append(f"[LATENCY]   {label:<24} +{t - prev:7.3f}s   @{t:7.3f}s")
            prev = t
        lines.append(f"[LATENCY]   {'TOTAL':<24} {prev:8.3f}s")
        logger.info("\n".join(lines))
        self._t0 = None


probe = LatencyProbe()
