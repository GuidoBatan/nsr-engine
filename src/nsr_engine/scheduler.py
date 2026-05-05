# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Adaptive CPU scheduler for NSR v0.1.0.

Latency-aware admission control for the LivePortrait pipeline. Lives
alongside the RuntimeGuardLayer; does not replace it.

Design goals
------------
- EMA-tracked per-stage latencies (motion, warp, spade, render, total).
- Per-stage adaptive budgets with consistent skip semantics.
- Three CPU mode tiers (HIGH / MED / LOW) with hysteresis switching.
- HARD admission gate: warp and spade may be skipped entirely under
  load. When skipped, the consumer reuses the last valid output.
- Render is real-time. Inference is optional, scheduled, and may be
  skipped on any given frame.
- No threads, no locks held across stage execution. The scheduler is
  cheap to call and safe from any worker.

The scheduler does NOT touch ONNX sessions, model I/O contracts, or
pipeline architecture. It only decides whether to run a stage and
records observed latencies.
"""

from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger("nsr.scheduler")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EMA smoothing factor. 0.2 is a balanced default per spec (0.1–0.3):
# fast enough to react to load shifts within ~10 frames, slow enough
# to not chatter on jitter.
_EMA_ALPHA: Final[float] = 0.2

# Real-time target.
_FRAME_BUDGET_MS: Final[float] = 33.3  # 30 FPS

# Mode-switch hysteresis: ratios of EMA total / 33ms.
# Going DOWN to a worse mode requires ratio above the upper bound;
# going UP to a better mode requires ratio below the lower bound.
# Wide gap → no oscillation.
_HIGH_TO_MED_RATIO: Final[float] = 1.10   # > this → drop to MED
_MED_TO_LOW_RATIO: Final[float] = 1.50    # > this → drop to LOW
_LOW_TO_MED_RATIO: Final[float] = 0.95    # < this for sustained → climb to MED
_MED_TO_HIGH_RATIO: Final[float] = 0.65   # < this for sustained → climb to HIGH

# How many consecutive samples below the climb threshold are required
# before the scheduler is willing to upgrade the mode. Prevents a single
# fast frame from flipping us back to HIGH.
_CLIMB_DWELL: Final[int] = 30

# Minimum number of EMA samples before any mode switch is allowed.
# Prevents premature classification during warm-up.
_WARMUP_SAMPLES: Final[int] = 15

# Per-mode stage budgets (target latency, ms). Used as the soft cap
# before admission control denies a stage. Spec ranges:
#   motion 5–15, warp 10–25, spade 30–80.
_MODE_BUDGETS: Final[dict[str, dict[str, float]]] = {
    "HIGH": {"motion": 15.0, "warp": 25.0, "spade": 80.0},
    "MED":  {"motion": 12.0, "warp": 20.0, "spade": 60.0},
    "LOW":  {"motion": 10.0, "warp": 15.0, "spade": 40.0},
}

# Per-mode stage cadence (run-once-every-N-frames). Spec FPS targets,
# at a 30 FPS pipeline tick:
#   HIGH spade 8–10 FPS → every 3–4 frames; warp 15 FPS → every 2.
#   MED  spade 5 FPS → every 6;             warp 10 FPS → every 3.
#   LOW  spade 0–2 FPS → every 15+;         warp 3–5 FPS → every 6–10.
_MODE_CADENCE: Final[dict[str, dict[str, int]]] = {
    "HIGH": {"warp": 2, "spade": 3},
    "MED":  {"warp": 3, "spade": 6},
    "LOW":  {"warp": 8, "spade": 15},
}


class CPUMode(enum.Enum):
    HIGH = "HIGH"
    MED = "MED"
    LOW = "LOW"


# ---------------------------------------------------------------------------
# EMA primitive
# ---------------------------------------------------------------------------

class LatencyEMA:
    """Exponentially weighted moving average of stage latency.

    EMA = alpha * sample + (1 - alpha) * EMA_prev

    First sample seeds the EMA directly so the first reading is not
    dragged toward zero. Thread-safe: a single float load/store under
    a lock.
    """

    def __init__(self, alpha: float = _EMA_ALPHA, name: str = "") -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._name = name
        self._value: float = 0.0
        self._samples: int = 0
        self._lock = threading.Lock()

    def update(self, sample_ms: float) -> float:
        """Add a new sample. Returns the new EMA value."""
        if sample_ms < 0.0 or sample_ms != sample_ms:  # reject NaN / negative
            return self.value()
        with self._lock:
            if self._samples == 0:
                self._value = sample_ms
            else:
                self._value = self._alpha * sample_ms + (1.0 - self._alpha) * self._value
            self._samples += 1
            return self._value

    def value(self) -> float:
        with self._lock:
            return self._value

    def samples(self) -> int:
        with self._lock:
            return self._samples

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0
            self._samples = 0


# ---------------------------------------------------------------------------
# Admission decision record
# ---------------------------------------------------------------------------

@dataclass
class StageDecision:
    """Decision payload for an admission check."""
    admit: bool
    reason: str  # "ok" | "cadence" | "ema_overload" | "stage_budget" | "no_cache"

    def __bool__(self) -> bool:
        return self.admit


# ---------------------------------------------------------------------------
# Adaptive scheduler
# ---------------------------------------------------------------------------

class AdaptiveScheduler:
    """Latency-aware admission controller and mode selector.

    Used by the render thread to decide whether to run warp / spade for
    the current frame. The decision is based on:
      1. CPU mode cadence (run every Nth frame)
      2. EMA total pipeline latency vs frame budget
      3. Per-stage EMA vs the stage's mode budget
      4. Whether a cached fallback is available (mandatory for the very
         first warp/spade — those must run at least once)
    """

    def __init__(
        self,
        frame_budget_ms: float = _FRAME_BUDGET_MS,
        alpha: float = _EMA_ALPHA,
    ) -> None:
        self._frame_budget_ms = float(frame_budget_ms)

        self.motion_ema = LatencyEMA(alpha, "motion")
        self.warp_ema = LatencyEMA(alpha, "warp")
        self.spade_ema = LatencyEMA(alpha, "spade")
        self.render_ema = LatencyEMA(alpha, "render")
        self.total_ema = LatencyEMA(alpha, "total")

        self._lock = threading.Lock()
        self._mode: CPUMode = CPUMode.HIGH
        self._frame_index: int = 0
        self._climb_streak: int = 0  # consecutive samples below climb threshold

        # Cache availability — once True, scheduler may legally skip
        # the corresponding stage. The first execution of each stage is
        # therefore always admitted regardless of load.
        self._has_warp_cache: bool = False
        self._has_spade_cache: bool = False

        # Counters for observability.
        self._skipped_warp: int = 0
        self._skipped_spade: int = 0
        self._executed_warp: int = 0
        self._executed_spade: int = 0

    # ------------------------------------------------------------------
    # Latency intake
    # ------------------------------------------------------------------

    def record_motion_latency(self, ms: float) -> None:
        self.motion_ema.update(ms)

    def record_warp_latency(self, ms: float) -> None:
        self.warp_ema.update(ms)

    def record_spade_latency(self, ms: float) -> None:
        self.spade_ema.update(ms)

    def record_render_latency(self, ms: float) -> None:
        self.render_ema.update(ms)

    def record_total_latency(self, ms: float) -> None:
        self.total_ema.update(ms)
        self._tick_mode()

    # ------------------------------------------------------------------
    # Cache notifications (called by the engine when a stage successfully
    # produces output that can be reused as fallback).
    # ------------------------------------------------------------------

    def mark_warp_cache_available(self) -> None:
        with self._lock:
            self._has_warp_cache = True

    def mark_spade_cache_available(self) -> None:
        with self._lock:
            self._has_spade_cache = True

    def reset_caches(self) -> None:
        """Called by the engine on RECOVERY so the next frame is forced
        to execute warp+spade end-to-end (no stale cache reuse across a
        camera reopen)."""
        with self._lock:
            self._has_warp_cache = False
            self._has_spade_cache = False
            self._frame_index = 0
            self._climb_streak = 0

    # ------------------------------------------------------------------
    # Frame indexing — caller increments once per render-loop iteration
    # that actually has a fresh motion packet to consume.
    # ------------------------------------------------------------------

    def begin_frame(self) -> int:
        with self._lock:
            self._frame_index += 1
            return self._frame_index

    # ------------------------------------------------------------------
    # Admission control
    # ------------------------------------------------------------------

    def admit_warp(self) -> StageDecision:
        """Decide whether to execute the warping stage on this frame."""
        with self._lock:
            mode = self._mode
            fi = self._frame_index
            has_cache = self._has_warp_cache

        # First-ever warp — must run, no cache to fall back to.
        if not has_cache:
            return StageDecision(True, "no_cache")

        # Mode cadence: warp runs every N frames in this mode.
        cadence = _MODE_CADENCE[mode.value]["warp"]
        if cadence > 1 and (fi % cadence) != 0:
            self._skipped_warp += 1
            return StageDecision(False, "cadence")

        # EMA-based hard admission gate.
        if self.total_ema.value() > self._frame_budget_ms * _MED_TO_LOW_RATIO:
            self._skipped_warp += 1
            return StageDecision(False, "ema_overload")

        # Per-stage budget.
        budget = _MODE_BUDGETS[mode.value]["warp"]
        if self.warp_ema.samples() > _WARMUP_SAMPLES and self.warp_ema.value() > budget * 1.5:
            self._skipped_warp += 1
            return StageDecision(False, "stage_budget")

        return StageDecision(True, "ok")

    def admit_spade(self) -> StageDecision:
        """Decide whether to execute the spade stage on this frame.

        Spade is the heaviest stage on CPU. It is the most likely to be
        skipped under load. When skipped, the caller reuses the last
        valid spade RGB.
        """
        with self._lock:
            mode = self._mode
            fi = self._frame_index
            has_cache = self._has_spade_cache

        # First-ever spade — must run.
        if not has_cache:
            return StageDecision(True, "no_cache")

        # Cadence.
        cadence = _MODE_CADENCE[mode.value]["spade"]
        if cadence > 1 and (fi % cadence) != 0:
            self._skipped_spade += 1
            return StageDecision(False, "cadence")

        # EMA gate. Spade has a tighter EMA threshold than warp because
        # it dominates the budget.
        if self.total_ema.value() > self._frame_budget_ms * _HIGH_TO_MED_RATIO:
            self._skipped_spade += 1
            return StageDecision(False, "ema_overload")

        # Per-stage budget.
        budget = _MODE_BUDGETS[mode.value]["spade"]
        if self.spade_ema.samples() > _WARMUP_SAMPLES and self.spade_ema.value() > budget * 1.25:
            self._skipped_spade += 1
            return StageDecision(False, "stage_budget")

        return StageDecision(True, "ok")

    def admit_motion(self) -> StageDecision:
        """Motion is soft real-time — almost always admitted. Returned
        for symmetry; the engine may bypass calling this and run motion
        unconditionally, which is also correct."""
        return StageDecision(True, "ok")

    # ------------------------------------------------------------------
    # Stage-execution accounting
    # ------------------------------------------------------------------

    def note_warp_executed(self) -> None:
        with self._lock:
            self._executed_warp += 1

    def note_spade_executed(self) -> None:
        with self._lock:
            self._executed_spade += 1

    # ------------------------------------------------------------------
    # CPU mode (with hysteresis)
    # ------------------------------------------------------------------

    def _tick_mode(self) -> None:
        """Re-evaluate CPU mode after each total-latency sample."""
        total = self.total_ema.value()
        samples = self.total_ema.samples()
        if samples < _WARMUP_SAMPLES:
            return

        ratio = total / max(self._frame_budget_ms, 1e-6)

        with self._lock:
            current = self._mode

            # Downgrade path — immediate.
            if current == CPUMode.HIGH and ratio > _HIGH_TO_MED_RATIO:
                self._mode = CPUMode.MED
                self._climb_streak = 0
                logger.info(
                    "scheduler: HIGH → MED (total_ema=%.1fms ratio=%.2f)",
                    total, ratio,
                )
                return
            if current == CPUMode.MED and ratio > _MED_TO_LOW_RATIO:
                self._mode = CPUMode.LOW
                self._climb_streak = 0
                logger.info(
                    "scheduler: MED → LOW (total_ema=%.1fms ratio=%.2f)",
                    total, ratio,
                )
                return

            # Upgrade path — requires sustained low ratio (hysteresis).
            climb_threshold = (
                _MED_TO_HIGH_RATIO if current == CPUMode.MED
                else _LOW_TO_MED_RATIO if current == CPUMode.LOW
                else 0.0
            )
            if current == CPUMode.HIGH:
                self._climb_streak = 0
                return

            if ratio < climb_threshold:
                self._climb_streak += 1
            else:
                self._climb_streak = 0

            if self._climb_streak >= _CLIMB_DWELL:
                if current == CPUMode.LOW:
                    self._mode = CPUMode.MED
                    logger.info(
                        "scheduler: LOW → MED (sustained ratio<%.2f)",
                        climb_threshold,
                    )
                elif current == CPUMode.MED:
                    self._mode = CPUMode.HIGH
                    logger.info(
                        "scheduler: MED → HIGH (sustained ratio<%.2f)",
                        climb_threshold,
                    )
                self._climb_streak = 0

    def mode(self) -> CPUMode:
        with self._lock:
            return self._mode

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def overload_ratio(self) -> float:
        """total_pipeline_latency_ema / frame_budget_ms."""
        return self.total_ema.value() / max(self._frame_budget_ms, 1e-6)

    def cpu_pressure_score(self) -> float:
        """Composite metric in [0, ~3]:
            0.5 * overload_ratio + 0.3 * (spade_ema / spade_budget)
                                 + 0.2 * (warp_ema  / warp_budget)
        """
        with self._lock:
            mode = self._mode.value
        spade_budget = _MODE_BUDGETS[mode]["spade"]
        warp_budget = _MODE_BUDGETS[mode]["warp"]
        return (
            0.5 * self.overload_ratio()
            + 0.3 * (self.spade_ema.value() / max(spade_budget, 1e-6))
            + 0.2 * (self.warp_ema.value() / max(warp_budget, 1e-6))
        )

    # ------------------------------------------------------------------
    # Observability snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, float | str | int]:
        with self._lock:
            mode = self._mode.value
            executed_warp = self._executed_warp
            executed_spade = self._executed_spade
            skipped_warp = self._skipped_warp
            skipped_spade = self._skipped_spade
        return {
            "mode": mode,
            "motion_ema_ms": self.motion_ema.value(),
            "warp_ema_ms": self.warp_ema.value(),
            "spade_ema_ms": self.spade_ema.value(),
            "render_ema_ms": self.render_ema.value(),
            "total_ema_ms": self.total_ema.value(),
            "overload_ratio": self.overload_ratio(),
            "cpu_pressure": self.cpu_pressure_score(),
            "executed_warp": executed_warp,
            "executed_spade": executed_spade,
            "skipped_warp": skipped_warp,
            "skipped_spade": skipped_spade,
        }
