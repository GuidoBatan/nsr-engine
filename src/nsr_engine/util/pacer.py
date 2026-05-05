# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Portable frame pacer.

Fixed-rate clock targeting `target_fps`. Uses coarse sleep + spin-wait
tail for sub-millisecond accuracy. Tracks overruns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class PacerStats:
    frames: int = 0
    overruns: int = 0
    last_elapsed_ms: float = 0.0
    ewma_elapsed_ms: float = 0.0


class FramePacer:
    def __init__(self, target_fps: int) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self._budget_s: float = 1.0 / float(target_fps)
        self._frame_start: float = 0.0
        self.stats = PacerStats()

    def frame_begin(self) -> None:
        self._frame_start = time.perf_counter()

    def frame_end(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._frame_start
        elapsed_ms = elapsed * 1000.0
        self.stats.frames += 1
        self.stats.last_elapsed_ms = elapsed_ms
        a = 0.05
        self.stats.ewma_elapsed_ms = (
            a * elapsed_ms + (1.0 - a) * self.stats.ewma_elapsed_ms
            if self.stats.ewma_elapsed_ms > 0.0
            else elapsed_ms
        )
        remaining = self._budget_s - elapsed
        if remaining <= 0.0:
            self.stats.overruns += 1
            return
        # Coarse sleep with a 2 ms safety margin, then spin.
        coarse = remaining - 0.002
        if coarse > 0.0:
            time.sleep(coarse)
        target = self._frame_start + self._budget_s
        while time.perf_counter() < target:
            pass
