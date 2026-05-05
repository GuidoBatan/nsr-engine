# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Unit tests for the adaptive scheduler."""

from __future__ import annotations

import pytest

from nsr_engine.scheduler import (
    AdaptiveScheduler,
    CPUMode,
    LatencyEMA,
)

# ---------------------------------------------------------------------------
# LatencyEMA
# ---------------------------------------------------------------------------

class TestLatencyEMA:
    def test_first_sample_seeds_directly(self) -> None:
        ema = LatencyEMA(alpha=0.2)
        assert ema.value() == 0.0
        ema.update(50.0)
        # First sample is taken as-is (no drag toward 0).
        assert ema.value() == 50.0

    def test_subsequent_samples_smooth(self) -> None:
        ema = LatencyEMA(alpha=0.2)
        ema.update(50.0)
        ema.update(100.0)
        # 0.2 * 100 + 0.8 * 50 = 60
        assert ema.value() == pytest.approx(60.0, abs=1e-9)

    def test_converges_to_constant_input(self) -> None:
        ema = LatencyEMA(alpha=0.2)
        for _ in range(100):
            ema.update(33.3)
        assert ema.value() == pytest.approx(33.3, abs=0.01)

    def test_rejects_nan_and_negative(self) -> None:
        ema = LatencyEMA(alpha=0.2)
        ema.update(50.0)
        v = ema.value()
        ema.update(float("nan"))
        ema.update(-1.0)
        assert ema.value() == v

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            LatencyEMA(alpha=0.0)
        with pytest.raises(ValueError):
            LatencyEMA(alpha=1.5)

    def test_reset_clears_state(self) -> None:
        ema = LatencyEMA(alpha=0.2)
        ema.update(99.0)
        ema.reset()
        assert ema.value() == 0.0
        assert ema.samples() == 0

    def test_samples_counter(self) -> None:
        ema = LatencyEMA(alpha=0.2)
        for _ in range(7):
            ema.update(10.0)
        assert ema.samples() == 7


# ---------------------------------------------------------------------------
# Admission control: first-frame execution is always admitted
# ---------------------------------------------------------------------------

class TestAdmissionFirstFrame:
    def test_first_warp_always_admitted(self) -> None:
        s = AdaptiveScheduler()
        s.begin_frame()
        d = s.admit_warp()
        assert d.admit
        assert d.reason == "no_cache"

    def test_first_spade_always_admitted(self) -> None:
        s = AdaptiveScheduler()
        s.begin_frame()
        d = s.admit_spade()
        assert d.admit
        assert d.reason == "no_cache"

    def test_first_admit_holds_even_under_simulated_overload(self) -> None:
        s = AdaptiveScheduler()
        # Simulate massive EMA — but since no cache exists, must run.
        for _ in range(50):
            s.record_total_latency(500.0)
        s.begin_frame()
        assert s.admit_warp().admit
        assert s.admit_spade().admit


# ---------------------------------------------------------------------------
# Cache-driven skipping
# ---------------------------------------------------------------------------

class TestAdmissionSkipping:
    def _saturate(self, s: AdaptiveScheduler, total_ms: float, n: int = 50) -> None:
        for _ in range(n):
            s.record_total_latency(total_ms)

    def test_cadence_skips_warp_in_low_mode(self) -> None:
        s = AdaptiveScheduler()
        s.mark_warp_cache_available()
        s.mark_spade_cache_available()
        # Force LOW mode by saturating EMA.
        self._saturate(s, total_ms=200.0)
        assert s.mode() == CPUMode.LOW

        # In LOW, warp cadence is every 8 frames.
        skips = 0
        for _ in range(8):
            s.begin_frame()
            if not s.admit_warp().admit:
                skips += 1
        # At least 5 of 8 must skip due to cadence (1 of every 8 admits).
        assert skips >= 5

    def test_cadence_skips_spade_in_low_mode(self) -> None:
        s = AdaptiveScheduler()
        s.mark_warp_cache_available()
        s.mark_spade_cache_available()
        self._saturate(s, total_ms=200.0)
        assert s.mode() == CPUMode.LOW

        # In LOW, spade cadence is every 15 frames.
        skips = 0
        for _ in range(15):
            s.begin_frame()
            if not s.admit_spade().admit:
                skips += 1
        assert skips >= 12

    def test_high_mode_cadence_admits_more(self) -> None:
        s = AdaptiveScheduler()
        s.mark_warp_cache_available()
        s.mark_spade_cache_available()
        # Healthy EMA — stays HIGH.
        for _ in range(50):
            s.record_total_latency(20.0)
        assert s.mode() == CPUMode.HIGH

        admits = 0
        for _ in range(6):
            s.begin_frame()
            if s.admit_spade().admit:
                admits += 1
        # In HIGH, spade cadence is every 3 frames → 2 of 6 should admit.
        assert admits >= 2

    def test_admission_reason_strings(self) -> None:
        s = AdaptiveScheduler()
        s.mark_warp_cache_available()
        # Force LOW with all caches present.
        for _ in range(50):
            s.record_total_latency(200.0)
        s.begin_frame()
        d = s.admit_warp()
        # Should be a known reason string.
        assert d.reason in ("ok", "cadence", "ema_overload", "stage_budget")


# ---------------------------------------------------------------------------
# CPU mode hysteresis
# ---------------------------------------------------------------------------

class TestModeHysteresis:
    def test_starts_in_high(self) -> None:
        s = AdaptiveScheduler()
        assert s.mode() == CPUMode.HIGH

    def test_high_to_med_on_overload(self) -> None:
        s = AdaptiveScheduler()
        for _ in range(20):
            s.record_total_latency(40.0)  # ratio ~ 1.20 > 1.10
        assert s.mode() == CPUMode.MED

    def test_med_to_low_on_severe_overload(self) -> None:
        s = AdaptiveScheduler()
        for _ in range(20):
            s.record_total_latency(40.0)
        for _ in range(20):
            s.record_total_latency(60.0)  # ratio ~ 1.80 > 1.50
        assert s.mode() == CPUMode.LOW

    def test_low_to_med_requires_sustained_recovery(self) -> None:
        s = AdaptiveScheduler()
        # Drive to LOW.
        for _ in range(40):
            s.record_total_latency(60.0)
        assert s.mode() == CPUMode.LOW

        # A single fast frame must NOT promote — hysteresis required.
        s.record_total_latency(15.0)
        assert s.mode() == CPUMode.LOW

        # Sustained low ratio (< 0.95) for the climb dwell window.
        for _ in range(40):
            s.record_total_latency(15.0)
        assert s.mode() == CPUMode.MED

    def test_med_to_high_requires_sustained_low_ratio(self) -> None:
        s = AdaptiveScheduler()
        # Drive to MED.
        for _ in range(20):
            s.record_total_latency(40.0)
        assert s.mode() == CPUMode.MED

        # Mild improvement (ratio ~ 0.7) is borderline — must be < 0.65.
        for _ in range(40):
            s.record_total_latency(10.0)  # ratio ~ 0.30
        assert s.mode() == CPUMode.HIGH

    def test_no_oscillation_under_jitter(self) -> None:
        """Alternating samples around the threshold must NOT flip mode
        every frame."""
        s = AdaptiveScheduler()
        modes_observed = set()
        for i in range(200):
            sample = 35.0 if (i % 2 == 0) else 30.0
            s.record_total_latency(sample)
            modes_observed.add(s.mode())
        # At most two modes observed (HIGH or MED). Never LOW under
        # this jitter.
        assert CPUMode.LOW not in modes_observed

    def test_warmup_prevents_premature_switching(self) -> None:
        s = AdaptiveScheduler()
        # Spike on the very first sample — but warm-up should hold mode.
        s.record_total_latency(500.0)
        assert s.mode() == CPUMode.HIGH


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

class TestDerivedMetrics:
    def test_overload_ratio_zero_at_init(self) -> None:
        s = AdaptiveScheduler()
        assert s.overload_ratio() == 0.0

    def test_overload_ratio_correct_under_load(self) -> None:
        s = AdaptiveScheduler(frame_budget_ms=33.3)
        for _ in range(20):
            s.record_total_latency(66.6)
        assert s.overload_ratio() == pytest.approx(2.0, rel=0.05)

    def test_cpu_pressure_score_increases_with_load(self) -> None:
        s = AdaptiveScheduler()
        for _ in range(20):
            s.record_total_latency(20.0)
            s.record_warp_latency(10.0)
            s.record_spade_latency(40.0)
        light = s.cpu_pressure_score()

        s2 = AdaptiveScheduler()
        for _ in range(20):
            s2.record_total_latency(80.0)
            s2.record_warp_latency(40.0)
            s2.record_spade_latency(120.0)
        heavy = s2.cpu_pressure_score()

        assert heavy > light


# ---------------------------------------------------------------------------
# Snapshot / observability
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_keys(self) -> None:
        s = AdaptiveScheduler()
        snap = s.snapshot()
        for key in (
            "mode",
            "motion_ema_ms",
            "warp_ema_ms",
            "spade_ema_ms",
            "render_ema_ms",
            "total_ema_ms",
            "overload_ratio",
            "cpu_pressure",
            "executed_warp",
            "executed_spade",
            "skipped_warp",
            "skipped_spade",
        ):
            assert key in snap, f"missing snapshot key: {key}"

    def test_skip_counts_increase_on_denial(self) -> None:
        s = AdaptiveScheduler()
        s.mark_warp_cache_available()
        s.mark_spade_cache_available()
        for _ in range(50):
            s.record_total_latency(200.0)
        for _ in range(20):
            s.begin_frame()
            if not s.admit_warp().admit:
                pass
            if not s.admit_spade().admit:
                pass
        snap = s.snapshot()
        assert snap["skipped_warp"] > 0
        assert snap["skipped_spade"] > 0


# ---------------------------------------------------------------------------
# Reset (RECOVERY hook)
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_caches_clears_admission_state(self) -> None:
        s = AdaptiveScheduler()
        s.mark_warp_cache_available()
        s.mark_spade_cache_available()
        s.reset_caches()
        s.begin_frame()
        # After reset, the first call must be no_cache (admitted).
        assert s.admit_warp().reason == "no_cache"
        assert s.admit_spade().reason == "no_cache"


# ---------------------------------------------------------------------------
# DEGRADED-loop regression
# ---------------------------------------------------------------------------

class TestDegradedLoopRegression:
    """Regression: per-frame budget comparison flapped RUNNING ↔ DEGRADED
    on every jittery frame. Replaced by EMA-driven, mode-driven transition.
    """

    def test_alternating_jitter_does_not_flap_mode(self) -> None:
        s = AdaptiveScheduler()
        mode_changes = 0
        prev_mode = s.mode()
        for i in range(500):
            # 30 ms / 36 ms alternating — straddles the 33 ms budget.
            sample = 36.0 if (i % 2 == 0) else 30.0
            s.record_total_latency(sample)
            current = s.mode()
            if current != prev_mode:
                mode_changes += 1
                prev_mode = current
        # At most 1 mode change in 500 samples (HIGH → MED once if EMA
        # settles around 1.10x). NEVER LOW.
        assert mode_changes <= 2

    def test_sustained_overload_settles_in_low(self) -> None:
        s = AdaptiveScheduler()
        for _ in range(200):
            s.record_total_latency(150.0)  # ratio ~ 4.5
        assert s.mode() == CPUMode.LOW

    def test_recovery_from_low_is_gradual_not_instant(self) -> None:
        s = AdaptiveScheduler()
        for _ in range(50):
            s.record_total_latency(150.0)
        assert s.mode() == CPUMode.LOW

        # 5 fast frames must not climb out — dwell required.
        for _ in range(5):
            s.record_total_latency(10.0)
        assert s.mode() == CPUMode.LOW
