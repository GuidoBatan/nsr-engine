# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Runtime Guard Layer (RGL) — central control plane for NSR v0.1.0.

Responsibilities
----------------
- FSM state machine (INIT→CAMERA_PROBING→RUNNING→DEGRADED→RECOVERY→HARD_FAIL)
- Frame validation gate (monotonic timestamp + strictly increasing frame_id)
- Camera health supervision via CameraSupervisor
- Thread heartbeat monitoring
- ONNX session validity tracking
- Pipeline execution permissioning
- Per-frame raw telemetry (no smoothing)

RULE: All pipeline execution MUST be gated through RGL.allow_pipeline().
HARD_FAIL is terminal — no execution permitted after that state.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Final

import cv2
import numpy as np

from nsr_engine.util.typing import U8

logger = logging.getLogger("nsr.rgl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CAMERA_PROBE_WINDOW: Final[int] = 30         # sample window for stability
_CAMERA_PROBE_MIN_SUCCESS_RATE: Final[float] = 0.95
_CAMERA_CONSEC_FAIL_LIMIT: Final[int] = 15    # consecutive bad frames → recovery
_HEARTBEAT_STALE_S: Final[float] = 5.0        # thread dead if no heartbeat in this window
_MAX_RECOVERY_ATTEMPTS: Final[int] = 3        # repeated recovery → HARD_FAIL
_MONOTONIC_EPSILON: Final[float] = 0.0        # strict: t_new > t_prev required

# Backend priority: DSHOW > MSMF > ANY (spec requirement)
_BACKEND_PRIORITY: Final[list[int]] = []  # populated at module load


def _build_backend_priority() -> list[int]:
    out: list[int] = []
    if hasattr(cv2, "CAP_DSHOW"):
        out.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        out.append(cv2.CAP_MSMF)
    out.append(cv2.CAP_ANY)
    return out


_BACKEND_PRIORITY_LIST: Final[list[int]] = _build_backend_priority()


# ---------------------------------------------------------------------------
# FSM
# ---------------------------------------------------------------------------

class FSMState(enum.Enum):
    INIT = "INIT"
    CAMERA_PROBING = "CAMERA_PROBING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    RECOVERY = "RECOVERY"
    HARD_FAIL = "HARD_FAIL"


# ---------------------------------------------------------------------------
# Camera failure classification
# ---------------------------------------------------------------------------

class CameraFailureKind(enum.Enum):
    TRANSIENT = "TRANSIENT"   # frame drops, jitter — tolerated up to threshold
    DRIVER = "DRIVER"         # backend malfunction → RECOVERY
    DEVICE = "DEVICE"         # invalid device index → RECOVERY
    HARDWARE = "HARDWARE"     # disconnect → RECOVERY


# ---------------------------------------------------------------------------
# Telemetry frame record
# ---------------------------------------------------------------------------

@dataclass
class FrameTelemetry:
    """Raw per-frame telemetry. No smoothing, no averaging."""
    frame_id: int = 0
    timestamp: float = 0.0
    fsm_state: str = FSMState.INIT.value
    motion_ms: float = 0.0
    kp_ms: float = 0.0
    warp_ms: float = 0.0
    spade_ms: float = 0.0
    total_ms: float = 0.0


# ---------------------------------------------------------------------------
# Camera Supervisor
# ---------------------------------------------------------------------------

class CameraSupervisor:
    """Device validation + stream stability + backend blacklisting.

    Valid device criteria:
    - isOpened == True
    - ≥95% successful frames over 30-sample window
    - monotonic timestamps
    - stable resolution
    - bounded latency variance

    Backend priority: DSHOW → MSMF → ANY
    Blacklisting: unstable backend is blacklisted for the session.
    """

    def __init__(self, width: int, height: int) -> None:
        self._width = int(width)
        self._height = int(height)
        self._cap: cv2.VideoCapture | None = None
        self._device_index: int | None = None
        self._backend: int | None = None
        self._blacklisted_backends: set[int] = set()

        # Rolling window stats
        self._probe_results: list[bool] = []
        self._last_ts: float = 0.0
        self._consec_failures: int = 0

        # Failure classification last seen
        self._last_failure_kind: CameraFailureKind = CameraFailureKind.TRANSIENT

    def open_best(self) -> bool:
        """Probe and open the best available camera. Returns True on success."""
        for backend in _BACKEND_PRIORITY_LIST:
            if backend in self._blacklisted_backends:
                continue
            for idx in range(4):
                cap = self._try_open_validated(idx, backend)
                if cap is not None:
                    self._cap = cap
                    self._device_index = idx
                    self._backend = backend
                    self._consec_failures = 0
                    self._probe_results.clear()
                    self._last_ts = 0.0
                    logger.info(
                        "CameraSupervisor: opened dev=%d backend=%d", idx, backend
                    )
                    return True
        logger.error("CameraSupervisor: no valid camera found")
        return False

    def _try_open_validated(
        self, idx: int, backend: int
    ) -> cv2.VideoCapture | None:
        """Open, configure, and validate a camera. Returns cap or None."""
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            return None

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Warm-up flush
        for _ in range(5):
            cap.grab()
            time.sleep(0.005)

        # Validate: 30-sample probe, require ≥95% success
        ok_count = 0
        last_probe_ts = 0.0
        ts_monotonic = True
        w_set: set[int] = set()
        h_set: set[int] = set()

        for _ in range(_CAMERA_PROBE_WINDOW):
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                ok_count += 1
                if t0 <= last_probe_ts:
                    ts_monotonic = False
                last_probe_ts = t0
                h_set.add(frame.shape[0])
                w_set.add(frame.shape[1])
            time.sleep(0.001)

        success_rate = ok_count / _CAMERA_PROBE_WINDOW
        resolution_stable = len(w_set) <= 1 and len(h_set) <= 1

        if (
            success_rate < _CAMERA_PROBE_MIN_SUCCESS_RATE
            or not ts_monotonic
            or not resolution_stable
        ):
            logger.debug(
                "CameraSupervisor: dev=%d backend=%d failed validation "
                "(rate=%.2f, monotonic=%s, res_stable=%s)",
                idx, backend, success_rate, ts_monotonic, resolution_stable,
            )
            cap.release()
            return None

        return cap

    def read(self) -> tuple[U8 | None, float]:
        """Read one frame. Updates consecutive failure counter."""
        if self._cap is None:
            self._consec_failures += 1
            return None, time.perf_counter()

        ok, frame = self._cap.read()
        ts = time.perf_counter()

        if not ok or frame is None or frame.size == 0:
            self._consec_failures += 1
            self._probe_results.append(False)
            if len(self._probe_results) > _CAMERA_PROBE_WINDOW:
                self._probe_results.pop(0)
            self._classify_failure()
            return None, ts

        # Validate monotonic timestamp
        if ts <= self._last_ts:
            self._consec_failures += 1
            self._probe_results.append(False)
            if len(self._probe_results) > _CAMERA_PROBE_WINDOW:
                self._probe_results.pop(0)
            return None, ts

        self._last_ts = ts
        self._consec_failures = 0
        self._probe_results.append(True)
        if len(self._probe_results) > _CAMERA_PROBE_WINDOW:
            self._probe_results.pop(0)
        return frame, ts

    def needs_recovery(self) -> bool:
        """True if ≥15 consecutive invalid frames."""
        return self._consec_failures >= _CAMERA_CONSEC_FAIL_LIMIT

    def stability_ok(self) -> bool:
        """True if 30-sample window success rate ≥ 95%."""
        if len(self._probe_results) < _CAMERA_PROBE_WINDOW:
            return False
        rate = sum(self._probe_results) / len(self._probe_results)
        return rate >= _CAMERA_PROBE_MIN_SUCCESS_RATE

    def _classify_failure(self) -> None:
        if self._cap is None:
            self._last_failure_kind = CameraFailureKind.DEVICE
        elif self._consec_failures >= _CAMERA_CONSEC_FAIL_LIMIT:
            self._last_failure_kind = CameraFailureKind.DRIVER
        else:
            self._last_failure_kind = CameraFailureKind.TRANSIENT

    def last_failure_kind(self) -> CameraFailureKind:
        return self._last_failure_kind

    def blacklist_current_backend(self) -> None:
        if self._backend is not None:
            logger.warning(
                "CameraSupervisor: blacklisting backend=%d", self._backend
            )
            self._blacklisted_backends.add(self._backend)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------------------
# Thread heartbeat registry
# ---------------------------------------------------------------------------

@dataclass
class _ThreadHeartbeat:
    name: str
    last_beat: float = field(default_factory=time.perf_counter)

    def beat(self) -> None:
        self.last_beat = time.perf_counter()

    def is_stale(self) -> bool:
        return (time.perf_counter() - self.last_beat) > _HEARTBEAT_STALE_S


# ---------------------------------------------------------------------------
# Frame validation gate
# ---------------------------------------------------------------------------

class FrameValidator:
    """Hard gate: validates monotonic timestamp + strictly increasing frame_id."""

    def __init__(self) -> None:
        self._last_ts: float = -1.0
        self._last_fid: int = -1

    def validate(self, frame: U8 | None, ts: float, frame_id: int) -> bool:
        """Returns True only if frame passes all checks. No bypass."""
        if frame is None:
            return False
        if frame.size == 0:
            return False
        if ts <= self._last_ts:
            return False
        if frame_id <= self._last_fid:
            return False
        self._last_ts = ts
        self._last_fid = frame_id
        return True

    def reset(self) -> None:
        self._last_ts = -1.0
        self._last_fid = -1


# ---------------------------------------------------------------------------
# Runtime Guard Layer
# ---------------------------------------------------------------------------

class RuntimeGuardLayer:
    """Central runtime control plane.

    All pipeline execution MUST call allow_pipeline() before proceeding.
    HARD_FAIL is terminal.
    """

    def __init__(self, frame_budget_ms: float) -> None:
        self._lock = threading.Lock()
        self._state: FSMState = FSMState.INIT
        self._recovery_attempts: int = 0
        self._frame_budget_ms = frame_budget_ms
        self._validator = FrameValidator()
        self._heartbeats: dict[str, _ThreadHeartbeat] = {}
        self._onnx_valid: bool = False
        self._last_recovery_cause: str | None = None

        # Telemetry (latest frame only, raw)
        self._telemetry = FrameTelemetry()

    # ------------------------------------------------------------------
    # FSM transitions (all guarded)
    # ------------------------------------------------------------------

    def transition(self, target: FSMState, reason: str = "") -> None:
        with self._lock:
            self._transition_locked(target, reason)

    def _transition_locked(self, target: FSMState, reason: str) -> None:
        current = self._state

        # HARD_FAIL is terminal
        if current == FSMState.HARD_FAIL:
            return

        # Validate legal transitions
        legal = self._legal_transitions(current)
        if target not in legal:
            logger.warning(
                "RGL: illegal FSM transition %s→%s ignored (reason=%s)",
                current.value, target.value, reason,
            )
            return

        logger.info(
            "RGL FSM: %s → %s%s",
            current.value, target.value,
            f" ({reason})" if reason else "",
        )
        self._state = target

    @staticmethod
    def _legal_transitions(state: FSMState) -> set[FSMState]:
        return {
            FSMState.INIT: {FSMState.CAMERA_PROBING},
            FSMState.CAMERA_PROBING: {FSMState.RUNNING, FSMState.RECOVERY},
            FSMState.RUNNING: {FSMState.DEGRADED, FSMState.RECOVERY, FSMState.HARD_FAIL},
            FSMState.DEGRADED: {FSMState.RUNNING, FSMState.RECOVERY, FSMState.HARD_FAIL},
            FSMState.RECOVERY: {FSMState.CAMERA_PROBING, FSMState.HARD_FAIL},
            FSMState.HARD_FAIL: set(),
        }[state]

    def state(self) -> FSMState:
        with self._lock:
            return self._state

    # ------------------------------------------------------------------
    # Pipeline permissioning
    # ------------------------------------------------------------------

    def allow_pipeline(self) -> bool:
        """MUST be called before every pipeline cycle. Returns False → skip."""
        with self._lock:
            s = self._state
        return s in (FSMState.RUNNING, FSMState.DEGRADED)

    # ------------------------------------------------------------------
    # Frame validation gate
    # ------------------------------------------------------------------

    def validate_frame(self, frame: U8 | None, ts: float, frame_id: int) -> bool:
        """Hard gate. Returns True only if frame is valid. No bypass."""
        if not self.allow_pipeline():
            return False
        return self._validator.validate(frame, ts, frame_id)

    def reset_validator(self) -> None:
        self._validator.reset()

    # ------------------------------------------------------------------
    # Camera supervision signals
    # ------------------------------------------------------------------

    def notify_camera_failure(self, kind: CameraFailureKind) -> None:
        """Called by capture thread on camera failure events."""
        if kind == CameraFailureKind.TRANSIENT:
            return  # tolerated
        with self._lock:
            self._trigger_recovery_locked(f"camera failure: {kind.value}")

    def notify_camera_stable(self) -> None:
        """Called when camera probe window shows stability."""
        with self._lock:
            if self._state == FSMState.CAMERA_PROBING:
                self._transition_locked(FSMState.RUNNING, "camera stable")

    # ------------------------------------------------------------------
    # Thread heartbeat
    # ------------------------------------------------------------------

    def register_thread(self, name: str) -> _ThreadHeartbeat:
        hb = _ThreadHeartbeat(name=name)
        with self._lock:
            self._heartbeats[name] = hb
        return hb

    def check_heartbeats(self) -> None:
        """Called periodically. Stale heartbeat → RECOVERY."""
        with self._lock:
            for name, hb in self._heartbeats.items():
                if hb.is_stale():
                    logger.error("RGL: thread %s heartbeat stale → RECOVERY", name)
                    self._trigger_recovery_locked(f"thread {name} dead")
                    return

    # ------------------------------------------------------------------
    # ONNX session tracking
    # ------------------------------------------------------------------

    def set_onnx_valid(self, valid: bool) -> None:
        with self._lock:
            self._onnx_valid = valid

    def onnx_valid(self) -> bool:
        with self._lock:
            return self._onnx_valid

    # ------------------------------------------------------------------
    # Budget / DEGRADED
    # ------------------------------------------------------------------

    def notify_frame_budget_exceeded(self, total_ms: float) -> None:
        with self._lock:
            if self._state == FSMState.RUNNING:
                self._transition_locked(
                    FSMState.DEGRADED,
                    f"frame budget exceeded: {total_ms:.1f}ms > {self._frame_budget_ms:.1f}ms",
                )

    def notify_frame_budget_ok(self) -> None:
        with self._lock:
            if self._state == FSMState.DEGRADED:
                self._transition_locked(FSMState.RUNNING, "budget recovered")

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def trigger_recovery(self, reason: str = "") -> None:
        with self._lock:
            self._trigger_recovery_locked(reason)

    def _trigger_recovery_locked(self, reason: str) -> None:
        if self._state == FSMState.HARD_FAIL:
            return
        self._recovery_attempts += 1
        if self._recovery_attempts > _MAX_RECOVERY_ATTEMPTS:
            self._transition_locked(FSMState.HARD_FAIL, "max recovery attempts exceeded")
            return
        self._last_recovery_cause = reason
        self._transition_locked(FSMState.RECOVERY, reason)

    def notify_recovery_complete(self) -> None:
        with self._lock:
            if self._state == FSMState.RECOVERY:
                self._last_recovery_cause = None
                self._transition_locked(FSMState.CAMERA_PROBING, "recovery complete")

    def recovery_attempts(self) -> int:
        with self._lock:
            return self._recovery_attempts

    def reset_recovery_attempts(self) -> None:
        with self._lock:
            self._recovery_attempts = 0


    def recovery_cause(self) -> str | None:
        with self._lock:
            return self._last_recovery_cause


    def stale_thread_name(self) -> str | None:
        """Return the name of the first stale thread, or None."""
        with self._lock:
            for name, hb in self._heartbeats.items():
                if hb.is_stale():
                    return name
        return None

    # ------------------------------------------------------------------
    # Anti-cheat invariants
    # ------------------------------------------------------------------

    def check_anti_cheat(
        self,
        prev_kp: np.ndarray | None,
        curr_kp: np.ndarray | None,
        prev_rgb: np.ndarray | None,
        curr_rgb: np.ndarray | None,
    ) -> None:
        """Enforce: different kp → different output. Violation → HARD_FAIL."""
        if prev_kp is None or curr_kp is None:
            return
        if prev_rgb is None or curr_rgb is None:
            return

        kp_diff = float(np.mean(np.abs(prev_kp.astype(np.float32) - curr_kp.astype(np.float32))))
        if kp_diff < 1e-6:
            return  # same input — no constraint on output

        rgb_diff = float(np.mean(np.abs(prev_rgb.astype(np.float32) - curr_rgb.astype(np.float32))))
        if rgb_diff < 1e-3:
            logger.error(
                "RGL: anti-cheat violation: kp_diff=%.6f but rgb_diff=%.6f → HARD_FAIL",
                kp_diff, rgb_diff,
            )
            with self._lock:
                self._state = FSMState.HARD_FAIL
                logger.info("RGL FSM: * → HARD_FAIL (anti-cheat: no output change)")

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def update_telemetry(self, t: FrameTelemetry) -> None:
        with self._lock:
            t.fsm_state = self._state.value
            self._telemetry = t

    def telemetry(self) -> FrameTelemetry:
        with self._lock:
            return self._telemetry

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Called once at engine start. INIT → CAMERA_PROBING."""
        with self._lock:
            if self._state == FSMState.INIT:
                self._transition_locked(FSMState.CAMERA_PROBING, "startup")
