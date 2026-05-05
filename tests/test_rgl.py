# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Unit tests for the Runtime Guard Layer (RGL).

Runs without any ONNX model files.
"""

from __future__ import annotations

import time

import numpy as np

from nsr_engine.rgl import (
    CameraFailureKind,
    CameraSupervisor,
    FrameTelemetry,
    FrameValidator,
    FSMState,
    RuntimeGuardLayer,
    _ThreadHeartbeat,
)

# ---------------------------------------------------------------------------
# FSM
# ---------------------------------------------------------------------------

class TestFSM:
    def test_initial_state_is_init(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        assert rgl.state() == FSMState.INIT

    def test_startup_transitions_to_camera_probing(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        assert rgl.state() == FSMState.CAMERA_PROBING

    def test_camera_stable_transitions_to_running(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        assert rgl.state() == FSMState.RUNNING

    def test_running_to_degraded_on_budget_exceeded(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        rgl.notify_frame_budget_exceeded(100.0)
        assert rgl.state() == FSMState.DEGRADED

    def test_degraded_recovers_to_running_on_budget_ok(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        rgl.notify_frame_budget_exceeded(100.0)
        rgl.notify_frame_budget_ok()
        assert rgl.state() == FSMState.RUNNING

    def test_any_state_to_recovery_on_failure(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        rgl.trigger_recovery("test")
        assert rgl.state() == FSMState.RECOVERY

    def test_recovery_to_camera_probing_after_complete(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        rgl.trigger_recovery("test")
        rgl.notify_recovery_complete()
        assert rgl.state() == FSMState.CAMERA_PROBING

    def test_repeated_recovery_triggers_hard_fail(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        for _ in range(4):
            rgl.trigger_recovery("repeated")
        assert rgl.state() == FSMState.HARD_FAIL

    def test_hard_fail_is_terminal(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        for _ in range(10):
            rgl.trigger_recovery("flood")
        assert rgl.state() == FSMState.HARD_FAIL
        # Further transitions must be no-ops
        rgl.notify_camera_stable()
        rgl.notify_recovery_complete()
        assert rgl.state() == FSMState.HARD_FAIL

    def test_illegal_transition_is_ignored(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        # INIT → RUNNING is not legal
        rgl.transition(FSMState.RUNNING, "illegal")
        assert rgl.state() == FSMState.INIT


# ---------------------------------------------------------------------------
# Pipeline gate
# ---------------------------------------------------------------------------

class TestPipelineGate:
    def test_allow_pipeline_false_in_init(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        assert not rgl.allow_pipeline()

    def test_allow_pipeline_false_in_camera_probing(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        assert not rgl.allow_pipeline()

    def test_allow_pipeline_true_in_running(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        assert rgl.allow_pipeline()

    def test_allow_pipeline_true_in_degraded(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        rgl.notify_frame_budget_exceeded(999.0)
        assert rgl.allow_pipeline()

    def test_allow_pipeline_false_in_recovery(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        rgl.trigger_recovery("test")
        assert not rgl.allow_pipeline()

    def test_allow_pipeline_false_in_hard_fail(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        for _ in range(10):
            rgl.trigger_recovery("flood")
        assert not rgl.allow_pipeline()


# ---------------------------------------------------------------------------
# Frame validation gate
# ---------------------------------------------------------------------------

class TestFrameValidator:
    def _make_frame(self) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def test_rejects_none_frame(self) -> None:
        v = FrameValidator()
        assert not v.validate(None, 1.0, 1)

    def test_rejects_empty_frame(self) -> None:
        v = FrameValidator()
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert not v.validate(empty, 1.0, 1)

    def test_accepts_valid_first_frame(self) -> None:
        v = FrameValidator()
        assert v.validate(self._make_frame(), 1.0, 1)

    def test_rejects_non_monotonic_timestamp(self) -> None:
        v = FrameValidator()
        assert v.validate(self._make_frame(), 1.0, 1)
        assert not v.validate(self._make_frame(), 0.9, 2)

    def test_rejects_same_timestamp(self) -> None:
        v = FrameValidator()
        assert v.validate(self._make_frame(), 1.0, 1)
        assert not v.validate(self._make_frame(), 1.0, 2)

    def test_rejects_non_increasing_frame_id(self) -> None:
        v = FrameValidator()
        assert v.validate(self._make_frame(), 1.0, 5)
        assert not v.validate(self._make_frame(), 2.0, 5)

    def test_rejects_decreasing_frame_id(self) -> None:
        v = FrameValidator()
        assert v.validate(self._make_frame(), 1.0, 5)
        assert not v.validate(self._make_frame(), 2.0, 4)

    def test_accepts_sequential_frames(self) -> None:
        v = FrameValidator()
        for i in range(10):
            assert v.validate(self._make_frame(), float(i + 1), i + 1)

    def test_reset_allows_restart(self) -> None:
        v = FrameValidator()
        assert v.validate(self._make_frame(), 10.0, 100)
        v.reset()
        assert v.validate(self._make_frame(), 1.0, 1)

    def test_rgl_validate_frame_requires_running_state(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        frame = self._make_frame()
        # INIT state → gate must block
        assert not rgl.validate_frame(frame, 1.0, 1)
        rgl.startup()
        # CAMERA_PROBING → gate must block
        assert not rgl.validate_frame(frame, 1.0, 1)
        rgl.notify_camera_stable()
        # RUNNING → gate passes
        assert rgl.validate_frame(frame, 1.0, 1)


# ---------------------------------------------------------------------------
# Thread heartbeat
# ---------------------------------------------------------------------------

class TestHeartbeat:
    def test_fresh_heartbeat_not_stale(self) -> None:
        hb = _ThreadHeartbeat(name="test")
        assert not hb.is_stale()

    def test_beat_resets_staleness(self) -> None:
        hb = _ThreadHeartbeat(name="test")
        # Manually backdate
        hb.last_beat = time.perf_counter() - 100.0
        assert hb.is_stale()
        hb.beat()
        assert not hb.is_stale()

    def test_rgl_stale_heartbeat_triggers_recovery(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        assert rgl.state() == FSMState.RUNNING

        hb = rgl.register_thread("stale_thread")
        hb.last_beat = time.perf_counter() - 100.0  # force stale

        rgl.check_heartbeats()
        assert rgl.state() in (FSMState.RECOVERY, FSMState.HARD_FAIL)

    def test_healthy_heartbeat_does_not_trigger_recovery(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()

        hb = rgl.register_thread("healthy_thread")
        hb.beat()

        rgl.check_heartbeats()
        assert rgl.state() == FSMState.RUNNING


# ---------------------------------------------------------------------------
# Camera failure classification → FSM
# ---------------------------------------------------------------------------

class TestCameraFailureNotification:
    def test_transient_failure_does_not_trigger_recovery(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        rgl.notify_camera_failure(CameraFailureKind.TRANSIENT)
        assert rgl.state() == FSMState.RUNNING

    def test_driver_failure_triggers_recovery(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        rgl.notify_camera_failure(CameraFailureKind.DRIVER)
        assert rgl.state() == FSMState.RECOVERY

    def test_device_failure_triggers_recovery(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        rgl.notify_camera_failure(CameraFailureKind.DEVICE)
        assert rgl.state() == FSMState.RECOVERY

    def test_hardware_failure_triggers_recovery(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        rgl.notify_camera_failure(CameraFailureKind.HARDWARE)
        assert rgl.state() == FSMState.RECOVERY


# ---------------------------------------------------------------------------
# Anti-cheat
# ---------------------------------------------------------------------------

class TestAntiCheat:
    def _kp(self, val: float) -> np.ndarray:
        return np.full((1, 21, 3), val, dtype=np.float32)

    def _rgb(self, val: int) -> np.ndarray:
        return np.full((512, 512, 3), val, dtype=np.uint8)

    def _running_rgl(self) -> RuntimeGuardLayer:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        return rgl

    def test_same_input_no_constraint(self) -> None:
        rgl = self._running_rgl()
        kp = self._kp(1.0)
        rgb = self._rgb(128)
        rgl.check_anti_cheat(kp, kp, rgb, rgb)
        assert rgl.state() == FSMState.RUNNING

    def test_different_kp_different_output_ok(self) -> None:
        rgl = self._running_rgl()
        rgl.check_anti_cheat(self._kp(0.0), self._kp(1.0), self._rgb(0), self._rgb(100))
        assert rgl.state() == FSMState.RUNNING

    def test_different_kp_identical_output_hard_fail(self) -> None:
        rgl = self._running_rgl()
        kp_a = self._kp(0.0)
        kp_b = self._kp(1.0)
        rgb = self._rgb(128)
        rgl.check_anti_cheat(kp_a, kp_b, rgb, rgb)
        assert rgl.state() == FSMState.HARD_FAIL

    def test_none_prev_no_check(self) -> None:
        rgl = self._running_rgl()
        rgl.check_anti_cheat(None, self._kp(1.0), None, self._rgb(0))
        assert rgl.state() == FSMState.RUNNING


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:
    def test_default_telemetry(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        t = rgl.telemetry()
        assert isinstance(t, FrameTelemetry)
        assert t.frame_id == 0

    def test_update_telemetry_stores_raw(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        rgl.notify_camera_stable()
        tel = FrameTelemetry(frame_id=42, timestamp=1.0, motion_ms=5.0, kp_ms=1.0,
                             warp_ms=10.0, spade_ms=20.0, total_ms=36.0)
        rgl.update_telemetry(tel)
        got = rgl.telemetry()
        assert got.frame_id == 42
        assert got.motion_ms == 5.0
        assert got.warp_ms == 10.0
        assert got.spade_ms == 20.0
        assert got.total_ms == 36.0
        assert got.fsm_state == FSMState.RUNNING.value

    def test_telemetry_fsm_state_updated(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.startup()
        tel = FrameTelemetry(frame_id=1)
        rgl.update_telemetry(tel)
        got = rgl.telemetry()
        assert got.fsm_state == FSMState.CAMERA_PROBING.value


# ---------------------------------------------------------------------------
# ONNX validity
# ---------------------------------------------------------------------------

class TestOnnxValidity:
    def test_default_onnx_invalid(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        assert not rgl.onnx_valid()

    def test_set_onnx_valid(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.set_onnx_valid(True)
        assert rgl.onnx_valid()

    def test_clear_onnx_valid(self) -> None:
        rgl = RuntimeGuardLayer(frame_budget_ms=33.3)
        rgl.set_onnx_valid(True)
        rgl.set_onnx_valid(False)
        assert not rgl.onnx_valid()


# ---------------------------------------------------------------------------
# CameraSupervisor unit (no real camera)
# ---------------------------------------------------------------------------

class TestCameraSupervisorBlacklist:
    def test_blacklist_backend_not_used_again(self) -> None:
        sup = CameraSupervisor(640, 480)
        # Simulate a backend being set
        import cv2
        sup._backend = cv2.CAP_ANY
        sup.blacklist_current_backend()
        assert cv2.CAP_ANY in sup._blacklisted_backends

    def test_needs_recovery_not_triggered_below_threshold(self) -> None:
        sup = CameraSupervisor(640, 480)
        sup._consec_failures = 14
        assert not sup.needs_recovery()

    def test_needs_recovery_triggered_at_threshold(self) -> None:
        sup = CameraSupervisor(640, 480)
        sup._consec_failures = 15
        assert sup.needs_recovery()

    def test_stability_ok_requires_full_window(self) -> None:
        sup = CameraSupervisor(640, 480)
        # Partial window — not yet stable
        sup._probe_results = [True] * 20
        assert not sup.stability_ok()

    def test_stability_ok_with_full_window(self) -> None:
        sup = CameraSupervisor(640, 480)
        sup._probe_results = [True] * 30
        assert sup.stability_ok()

    def test_stability_fails_below_rate(self) -> None:
        sup = CameraSupervisor(640, 480)
        # 80% — below 95% threshold
        sup._probe_results = [True] * 24 + [False] * 6
        assert not sup.stability_ok()
