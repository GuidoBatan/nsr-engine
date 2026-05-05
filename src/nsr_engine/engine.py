# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""NSR v0.1.0 engine orchestration — LivePortrait 5-stage pipeline (§11).

Pipeline
--------

    source (avatar, init only)                driving (webcam, per frame)
      |                                           |
      v                                           v
    AppearanceEncoder (F)                     MotionExtractor (M)
      |                                           |
      |  AppearanceFeature3D                      |  MotionParams_driving
      |  (frozen)                                 |
      |                                           v
      |                                       transform_keypoints
      |                                           |
      |                                           v
      |                                       kp_driving (raw)
      |                                           |
      |   MotionParams_source (computed ONCE     |
      |   at init from avatar crop)               |
      |        |                                  |
      |        v                                  |
      |   transform_keypoints                     |
      |        |                                  |
      |        v                                  |
      |   kp_source (frozen)                      |
      |        |             +--------------------+
      |        |             |
      |        v             v
      |   StitchingRetargeting (optional)
      |        |             |
      |        |             v
      |        |        kp_driving (refined)
      |        |             |
      |        +---+    +----+
      |            v    v
      v        WarpingNetwork (W)
     WarpingNetwork <--+
          |
          v
     WarpedFeature3D
          |
          v
     SpadeGenerator (G)
          |
          v
     uint8 RGB at OUTPUT_RES

Thread topology
---------------------------------------------------------

    [capture] -> cap_slot (size 1, drop-on-full)
    [motion]  -> motion_slot (size 1, drop-on-full)
    [render]  -> render_slot (size 1, drop-on-full)
    [main]    -> OBSWindow, paced at target_fps

All threads emit heartbeats monitored by RGL.
All pipeline execution is gated through RGL.allow_pipeline().
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, TypeVar

import cv2
import numpy as np

from nsr_engine.config import EngineConfig
from nsr_engine.contract.guard import assert_contract_integrity
from nsr_engine.face.appearance_encoder import AppearanceEncoder
from nsr_engine.face.cropper import FaceCropper
from nsr_engine.face.motion_extractor import MotionExtractor
from nsr_engine.motion.keypoint_transform import transform_keypoints
from nsr_engine.motion.latent_flow import LatentFlowWarper
from nsr_engine.output.window import OBSWindow
from nsr_engine.render.io import load_avatar_rgba, rgba_to_bgr
from nsr_engine.render.neural_stitcher import StitchingRetargeting
from nsr_engine.render.spade_generator import SpadeGenerator
from nsr_engine.rgl import (
    CameraSupervisor,
    FrameTelemetry,
    FSMState,
    RuntimeGuardLayer,
)
from nsr_engine.util.latents import (
    AppearanceFeature3D,
    CropResult,
    ImplicitKeypoints,
    MotionParams,
    RenderResult,
)
from nsr_engine.util.pacer import FramePacer
from nsr_engine.util.typing import F32, U8

logger = logging.getLogger("nsr.engine")

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Runtime knobs
# ---------------------------------------------------------------------------

_MIN_FACE_EDGE: float = 40.0
_MOTION_INPUT_MAX_EDGE: int = 256
_CAPTURE_IDLE_SLEEP_S: float = 0.001
_WORKER_IDLE_SLEEP_S: float = 0.0005
_FPS_LOG_INTERVAL_S: float = 1.0
_HEARTBEAT_INTERVAL_S: float = 0.5
_HB_CHECK_INTERVAL_S: float = 2.0

_ALPHA_KP: float = 0.6
_ALPHA_EXP: float = 0.6
_ALPHA_T: float = 0.5
_ALPHA_POSE: float = 0.6
_ALPHA_SCALE: float = 0.6

_LOWER_FACE_KP_INDICES: tuple[int, ...] = (14, 15, 16, 17, 18, 19, 20)
_LOWER_FACE_CONF_THRESHOLD: float = 0.6


class _LastWinsSlot(Generic[T]):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._value: T | None = None
        self._fresh: bool = False
        self._closed: bool = False
        self._version: int = 0

    def put(self, value: T) -> None:
        with self._cond:
            self._value = value
            self._fresh = True
            self._version += 1
            self._cond.notify_all()

    def get(self, timeout_s: float = 0.0) -> tuple[T | None, bool]:
        with self._cond:
            if timeout_s > 0.0 and not self._fresh and not self._closed:
                self._cond.wait(timeout=timeout_s)
            value = self._value
            was_fresh = self._fresh
            self._fresh = False
            return value, was_fresh

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


@dataclass
class _StageTimings:
    face_ms_last: float = 0.0
    motion_ms_last: float = 0.0
    warp_ms_last: float = 0.0
    render_ms_last: float = 0.0


@dataclass
class _GuideState:
    face_present: bool = False
    crop: CropResult | None = None
    camera_frame: U8 | None = None


class NSREngine:
    def __init__(self, cfg: EngineConfig) -> None:
        assert_contract_integrity()

        self._cfg = cfg
        self._inset_scale = 1.0
        self._timings = _StageTimings()
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()

        self._frame_count: int = 0
        self._fps_render: float = 0.0
        self._fps_last_update: float = time.perf_counter()
        self._fps_render_count: int = 0
        self._last_displayed_frame_id: int = 0

        self._last_valid_crop: CropResult | None = None
        self._last_valid_crop_ts: float = 0.0

        self._guide_state = _GuideState()

        # Runtime Guard Layer
        self._rgl = RuntimeGuardLayer(cfg.frame_budget_ms)

        self._cropper = FaceCropper(cfg.face, 1)
        self._appearance_enc = AppearanceEncoder(
            cfg.neural.appearance_encoder,
            cfg.face.ort_intra_threads,
            cfg.face.ort_inter_threads,
        )
        self._motion_ext = MotionExtractor(
            cfg.neural.motion_extractor,
            cfg.face.ort_intra_threads,
            cfg.face.ort_inter_threads,
        )
        self._warper = LatentFlowWarper(
            cfg.neural.warping,
            cfg.face.ort_intra_threads,
            cfg.face.ort_inter_threads,
        )
        self._spade = SpadeGenerator(
            cfg.neural.spade_generator,
            cfg.face.ort_intra_threads,
            cfg.face.ort_inter_threads,
        )

        self._stitcher: StitchingRetargeting | None = None
        if cfg.use_stitching:
            self._stitcher = StitchingRetargeting(
                cfg.neural.stitching,
                cfg.face.ort_intra_threads,
                cfg.face.ort_inter_threads,
            )

        self._rgl.set_onnx_valid(True)

        avatar_rgba = load_avatar_rgba(cfg.avatar_path)
        avatar_bgr = rgba_to_bgr(avatar_rgba)
        avatar_crop_result = self._cropper.crop(avatar_bgr)
        if not avatar_crop_result.valid:
            raise RuntimeError(
                "avatar face detection failed; detector produced no bbox. "
                "Check that the avatar image contains a clearly visible face."
            )

        self._avatar_crop: U8 = avatar_crop_result.bgr.copy()
        self._avatar_feature: AppearanceFeature3D = self._appearance_enc.encode(
            self._avatar_crop
        )
        self._avatar_motion: MotionParams = self._motion_ext.extract(
            self._avatar_crop
        )
        self._kp_source: ImplicitKeypoints = transform_keypoints(self._avatar_motion)

        self._cropper.reset()

        self._cap_slot: _LastWinsSlot[tuple[U8, float, int]] = _LastWinsSlot()
        self._motion_slot: _LastWinsSlot[tuple[MotionParams, CropResult, int]] = _LastWinsSlot()
        self._render_slot: _LastWinsSlot[tuple[RenderResult, int]] = _LastWinsSlot()

        self._camera_supervisor: CameraSupervisor | None = None
        self._window: OBSWindow | None = None
        self._threads: list[threading.Thread] = []

        self._pacer = FramePacer(cfg.target_fps)

        self._last_display_rgb: U8 = cv2.resize(
            self._avatar_crop,
            (cfg.output_width, cfg.output_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Anti-cheat state
        self._prev_kp_data: np.ndarray | None = None
        self._prev_rgb: U8 | None = None

        self._last_hb_check: float = time.perf_counter()

    def run(self) -> None:
        cfg = self._cfg

        self._rgl.startup()

        self._camera_supervisor = CameraSupervisor(cfg.webcam_width, cfg.webcam_height)
        if not self._camera_supervisor.open_best():
            self._rgl.trigger_recovery("camera open failed at startup")
            if self._rgl.state() == FSMState.HARD_FAIL:
                raise RuntimeError("RGL HARD_FAIL: no camera available")

        self._window = OBSWindow(cfg.output_width, cfg.output_height, cfg.headless)

        self._start_threads()

        try:
            while not self._stop_event.is_set():
                if self._rgl.state() == FSMState.HARD_FAIL:
                    logger.error("RGL HARD_FAIL: engine halting")
                    break

                now = time.perf_counter()
                if now - self._last_hb_check >= _HB_CHECK_INTERVAL_S:
                    # Direct thread-aliveness check first — catches a crashed
                    # worker faster than the heartbeat staleness window. The
                    # cause is tagged "thread" so _do_recovery() restarts the
                    # dead worker instead of touching the camera.
                    self._detect_dead_threads()
                    self._rgl.check_heartbeats()
                    self._last_hb_check = now

                if self._rgl.state() == FSMState.RECOVERY:
                    self._do_recovery()
                    continue

                self._pacer.frame_begin()

                drained = self._drain_render_slot()
                if drained is not None and drained[1] > self._last_displayed_frame_id:
                    frame_rgb, frame_id = drained
                    self._last_displayed_frame_id = frame_id
                    self._last_display_rgb = frame_rgb
                    self._fps_render_count += 1

                camera_frame = self._get_latest_camera_frame()
                crop = self._get_latest_crop()
                face_present = self._is_face_present()

                display_rgb = self._compose_display(
                    base_rgb=self._last_display_rgb,
                    camera_bgr=camera_frame,
                    crop=crop,
                    face_present=face_present,
                )

                if self._window is not None:
                    self._window.show(_rgb_to_rgba(display_rgb))
                    if self._window.poll_quit():
                        break

                self._pacer.frame_end()
                self._frame_count += 1
                self._update_fps()

        except KeyboardInterrupt:
            pass
        except Exception:
            logger.exception("engine crashed")
            raise
        finally:
            self._stop_event.set()
            self._cleanup()

    def _detect_dead_threads(self) -> None:
        """If any worker thread has exited, mark its heartbeat name as
        the stale thread and trigger a thread-cause recovery. Faster than
        waiting for heartbeat staleness to elapse."""
        thread_to_hb = {
            "nsr-capture": "capture_thread",
            "nsr-motion": "motion_thread",
            "nsr-render": "render_thread",
        }
        for t in self._threads:
            if t.is_alive():
                continue
            hb_name = thread_to_hb.get(t.name)
            if hb_name is None:
                continue
            # Only notify if not already in RECOVERY (avoid attempt-counter inflation).
            if self._rgl.state() != FSMState.RECOVERY:
                logger.warning("RGL: detected dead thread %s", t.name)
                self._rgl.notify_thread_dead(hb_name)
            return

    def _do_recovery(self) -> None:
        """Cause-aware recovery.

        - cause == "thread": restart only the failed worker thread. Do NOT
          touch the camera or blacklist its backend. A render-thread death
          must not cost the camera backend.
        - cause == "camera": release+reopen the camera. Backend is
          blacklisted ONLY if the reopen on the same backend fails (handled
          inside CameraSupervisor.open_best via its own retry path).
        - cause == "" (legacy / unknown): restart any non-alive worker
          thread, then reset validator. Does not blacklist.
        """
        cause = self._rgl.recovery_cause()
        stale = self._rgl.stale_thread_name()
        logger.info("RGL RECOVERY: cause=%r stale_thread=%r", cause, stale)
        time.sleep(0.2)

        if cause == "camera":
            if self._camera_supervisor is not None:
                self._camera_supervisor.release()
                opened = self._camera_supervisor.open_best()
                if not opened:
                    # Same backend failed → blacklist and try a different one.
                    self._camera_supervisor.blacklist_current_backend()
                    opened = self._camera_supervisor.open_best()
                if not opened:
                    self._rgl.trigger_recovery("camera reopen failed in RECOVERY")
                    return

        # Restart any worker thread that has died, regardless of cause —
        # a thread that exited cannot be re-entered, so this is safe and
        # idempotent for live threads.
        self._restart_dead_threads()

        self._rgl.reset_validator()
        self._cropper.reset()
        self._prev_kp_data = None
        self._prev_rgb = None
        self._rgl.notify_recovery_complete()

        # The system has visibly recovered: drop the cumulative attempt
        # counter so a series of unrelated transient incidents over a long
        # run does not eventually latch into HARD_FAIL.
        self._rgl.reset_recovery_attempts()
        logger.info("RGL RECOVERY: complete")

    _THREAD_TARGETS: dict[str, str] = {
        "capture_thread": "_capture_loop",
        "motion_thread": "_motion_loop",
        "render_thread": "_render_loop",
    }

    _THREAD_NAMES: dict[str, str] = {
        "capture_thread": "nsr-capture",
        "motion_thread": "nsr-motion",
        "render_thread": "nsr-render",
    }

    def _restart_dead_threads(self) -> None:
        """Replace any worker thread that is no longer alive. Live threads
        are left untouched. The old heartbeat record is dropped before
        the new thread re-registers so a stale entry of the dead thread
        does not immediately re-trigger RECOVERY."""
        thread_name_to_target = {
            "nsr-capture": ("capture_thread", self._capture_loop),
            "nsr-motion": ("motion_thread", self._motion_loop),
            "nsr-render": ("render_thread", self._render_loop),
        }
        new_threads: list[threading.Thread] = []
        for t in self._threads:
            if t.is_alive():
                new_threads.append(t)
                continue
            entry = thread_name_to_target.get(t.name)
            if entry is None:
                continue
            hb_name, target = entry
            self._rgl.unregister_thread(hb_name)
            replacement = threading.Thread(target=target, name=t.name, daemon=True)
            replacement.start()
            new_threads.append(replacement)
            logger.warning("RGL: restarted dead thread %s", t.name)
        self._threads = new_threads

    def __enter__(self) -> NSREngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self._cleanup()
        return False

    def _start_threads(self) -> None:
        self._threads = [
            threading.Thread(target=self._capture_loop, name="nsr-capture", daemon=True),
            threading.Thread(target=self._motion_loop, name="nsr-motion", daemon=True),
            threading.Thread(target=self._render_loop, name="nsr-render", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def _capture_loop(self) -> None:
        assert self._camera_supervisor is not None
        hb = self._rgl.register_thread("capture_thread")
        frame_id = 0
        last_hb_emit = time.perf_counter()

        try:
            while not self._stop_event.is_set():
                now = time.perf_counter()
                if now - last_hb_emit >= _HEARTBEAT_INTERVAL_S:
                    hb.beat()
                    last_hb_emit = now

                frame, t_cap = self._camera_supervisor.read()

                if frame is None:
                    if self._camera_supervisor.needs_recovery():
                        fk = self._camera_supervisor.last_failure_kind()
                        self._rgl.notify_camera_failure(fk)
                    time.sleep(_CAPTURE_IDLE_SLEEP_S)
                    continue

                frame_id += 1

                # CRITICAL: stability promotion runs UNCONDITIONALLY on every
                # successful read so bootstrap CAMERA_PROBING → RUNNING fires.
                # If gated behind validate_frame, the gate's allow_pipeline()
                # check (which is False in CAMERA_PROBING) prevents the
                # transition forever — pipeline transition failure regression.
                if self._camera_supervisor.stability_ok():
                    self._rgl.notify_camera_stable()

                # Single controlled copy
                frame_copy = frame.copy()

                # UI preview state always updated — webcam feed must be
                # visible during bootstrap and probing, not only after RUNNING.
                with self._state_lock:
                    self._guide_state.camera_frame = frame_copy

                # Frame validation gate — hard gate for the inference path.
                # Rejected frames do NOT enter the cap_slot, so motion/warp/spade
                # never run on invalid frames. UI preview above is unaffected.
                if not self._rgl.validate_frame(frame, t_cap, frame_id):
                    time.sleep(_CAPTURE_IDLE_SLEEP_S)
                    continue

                self._cap_slot.put((frame_copy, t_cap, frame_id))

        except Exception:
            logger.exception("capture worker crashed")
            self._rgl.trigger_recovery("capture thread exception")
            # Per-thread isolation — main loop will restart this thread alone.

    def _motion_loop(self) -> None:
        hb = self._rgl.register_thread("motion_thread")
        smoother = _MotionSmoother()
        smoother.set_neutral(self._avatar_motion.exp)
        last_processed_frame_id = 0
        last_hb_emit = time.perf_counter()

        try:
            while not self._stop_event.is_set():
                now = time.perf_counter()
                if now - last_hb_emit >= _HEARTBEAT_INTERVAL_S:
                    hb.beat()
                    last_hb_emit = now

                if not self._rgl.allow_pipeline():
                    hb.beat()
                    last_hb_emit = time.perf_counter()
                    time.sleep(_WORKER_IDLE_SLEEP_S)
                    continue

                pkt, _fresh = self._cap_slot.get(0.0)
                if pkt is None:
                    hb.beat()
                    last_hb_emit = time.perf_counter()
                    time.sleep(_WORKER_IDLE_SLEEP_S)
                    continue

                frame_bgr, _t_cap, frame_id = pkt
                if frame_id <= last_processed_frame_id:
                    hb.beat()
                    last_hb_emit = time.perf_counter()
                    time.sleep(_WORKER_IDLE_SLEEP_S)
                    continue

                last_processed_frame_id = frame_id

                t_face = time.perf_counter()
                crop = self._cropper.crop(frame_bgr)
                self._timings.face_ms_last = (time.perf_counter() - t_face) * 1000.0

                if not crop.valid:
                    with self._state_lock:
                        self._guide_state.face_present = False
                        self._guide_state.crop = None
                    continue

                if _bbox_max_edge(crop.bbox_xyxy) < _MIN_FACE_EDGE:
                    with self._state_lock:
                        self._guide_state.face_present = False
                        self._guide_state.crop = None
                    continue

                motion_input = _prepare_motion_input(crop.bgr)

                # Beat before the heaviest stage in this thread.
                hb.beat()

                t_motion = time.perf_counter()
                raw_params = self._motion_ext.extract(motion_input)
                self._timings.motion_ms_last = (time.perf_counter() - t_motion) * 1000.0

                hb.beat()
                last_hb_emit = time.perf_counter()

                params = smoother.smooth(raw_params, confidence=float(crop.score))

                with self._state_lock:
                    self._last_valid_crop = crop
                    self._last_valid_crop_ts = time.perf_counter()
                    self._guide_state.face_present = True
                    self._guide_state.crop = crop

                self._motion_slot.put((params, crop, frame_id))

        except Exception:
            # Per-thread isolation — see _render_loop comment.
            logger.exception("motion worker crashed")
            self._rgl.trigger_recovery("motion thread exception")

    def _render_loop(self) -> None:
        hb = self._rgl.register_thread("render_thread")
        cfg = self._cfg
        last_processed_frame_id = 0
        last_hb_emit = time.perf_counter()

        try:
            while not self._stop_event.is_set():
                now = time.perf_counter()
                if now - last_hb_emit >= _HEARTBEAT_INTERVAL_S:
                    hb.beat()
                    last_hb_emit = now

                if not self._rgl.allow_pipeline():
                    # Heartbeat must reflect liveness, not work availability.
                    # During RECOVERY / CAMERA_PROBING the loop is idle but
                    # alive — beat unconditionally so it is not declared dead.
                    hb.beat()
                    last_hb_emit = time.perf_counter()
                    time.sleep(_WORKER_IDLE_SLEEP_S)
                    continue

                pkt, _fresh = self._motion_slot.get(0.0)
                if pkt is None:
                    # Idle waiting for motion thread — still alive.
                    hb.beat()
                    last_hb_emit = time.perf_counter()
                    time.sleep(_WORKER_IDLE_SLEEP_S)
                    continue

                params, _crop, frame_id = pkt
                if frame_id <= last_processed_frame_id:
                    hb.beat()
                    last_hb_emit = time.perf_counter()
                    time.sleep(_WORKER_IDLE_SLEEP_S)
                    continue

                last_processed_frame_id = frame_id

                t_frame_start = time.perf_counter()

                # kp transform — mandatory per-frame, NO cache bypass
                t_kp = time.perf_counter()
                kp_driving = transform_keypoints(params)
                kp_ms = (time.perf_counter() - t_kp) * 1000.0

                if self._stitcher is not None:
                    kp_driving = self._stitcher.refine(self._kp_source, kp_driving)

                # Beat between long ONNX stages so a slow CPU SPADE / warp
                # never starves the heartbeat for the full _HEARTBEAT_STALE_S
                # window. Critical on CPU-only machines where spade ≈ 200–400 ms.
                hb.beat()

                # warp — mandatory per-frame
                t_warp = time.perf_counter()
                warped = self._warper.warp(
                    self._avatar_feature,
                    self._kp_source,
                    kp_driving,
                )
                warp_ms = (time.perf_counter() - t_warp) * 1000.0
                self._timings.warp_ms_last = warp_ms

                # Beat after warp, before the heaviest stage.
                hb.beat()

                # spade — mandatory per-frame
                t_spade = time.perf_counter()
                rgb = self._spade.decode(warped)
                spade_ms = (time.perf_counter() - t_spade) * 1000.0
                self._timings.render_ms_last = spade_ms

                hb.beat()
                last_hb_emit = time.perf_counter()

                if rgb.shape[0] != cfg.output_height or rgb.shape[1] != cfg.output_width:
                    rgb = cv2.resize(
                        rgb,
                        (cfg.output_width, cfg.output_height),
                        interpolation=cv2.INTER_LINEAR,
                    )

                total_ms = (time.perf_counter() - t_frame_start) * 1000.0

                if total_ms > cfg.frame_budget_ms:
                    self._rgl.notify_frame_budget_exceeded(total_ms)
                else:
                    self._rgl.notify_frame_budget_ok()

                # Anti-cheat
                self._rgl.check_anti_cheat(
                    self._prev_kp_data,
                    kp_driving.data,
                    self._prev_rgb,
                    rgb,
                )
                if self._rgl.state() == FSMState.HARD_FAIL:
                    self._stop_event.set()
                    return

                self._prev_kp_data = kp_driving.data.copy()
                self._prev_rgb = rgb.copy()

                # Raw per-frame telemetry
                tel = FrameTelemetry(
                    frame_id=frame_id,
                    timestamp=time.perf_counter(),
                    motion_ms=self._timings.motion_ms_last,
                    kp_ms=kp_ms,
                    warp_ms=warp_ms,
                    spade_ms=spade_ms,
                    total_ms=total_ms,
                )
                self._rgl.update_telemetry(tel)

                result = RenderResult(
                    rgb=rgb,
                    internal_ms=warp_ms + spade_ms,
                )
                result.validate()
                self._render_slot.put((result, frame_id))

        except Exception:
            # Per-thread isolation: render thread death must not kill the
            # whole system. RGL will mark recovery; the main loop restarts
            # this thread alone in _do_recovery(). DO NOT set _stop_event.
            logger.exception("render worker crashed")
            self._rgl.trigger_recovery("render thread exception")

    def _get_latest_camera_frame(self) -> U8 | None:
        with self._state_lock:
            if self._guide_state.camera_frame is None:
                return None
            return self._guide_state.camera_frame.copy()

    def _get_latest_crop(self) -> CropResult | None:
        with self._state_lock:
            return self._guide_state.crop

    def _is_face_present(self) -> bool:
        with self._state_lock:
            return bool(self._guide_state.face_present)

    def _compose_display(
        self,
        base_rgb: U8,
        camera_bgr: U8 | None,
        crop: CropResult | None,
        face_present: bool,
    ) -> U8:
        canvas_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

        if camera_bgr is not None and self._cfg.show_debug:
            inset = self._build_camera_inset(camera_bgr, crop, face_present)
            self._paste_inset(canvas_bgr, inset)

        return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

    def _build_camera_inset(
        self,
        camera_bgr: U8,
        crop: CropResult | None,
        face_present: bool,
    ) -> U8:
        h, w = camera_bgr.shape[:2]
        # Scale inset to % of output canvas
        target_w = max(120, int(self._cfg.output_width * 0.20))
        scale = target_w / max(1.0, float(w))
        target_h = int(round(h * scale))

        inset = cv2.resize(camera_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if crop is not None and crop.valid:
            bbox = crop.bbox_xyxy.astype(np.float32).copy()
            bbox[0] *= target_w / float(w)
            bbox[2] *= target_w / float(w)
            bbox[1] *= target_h / float(h)
            bbox[3] *= target_h / float(h)
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            x1 = max(0, min(target_w - 1, x1))
            y1 = max(0, min(target_h - 1, y1))
            x2 = max(0, min(target_w - 1, x2))
            y2 = max(0, min(target_h - 1, y2))
            color = (0, 255, 0) if face_present else (0, 0, 255)
            cv2.rectangle(inset, (x1, y1), (x2, y2), color, 2)

        self._draw_center_guide(inset, face_present, compact=True)
        cv2.rectangle(inset, (0, 0), (target_w - 1, target_h - 1), (255, 255, 255), 1)

        return inset

    def _paste_inset(self, canvas_bgr: U8, inset_bgr: U8) -> None:
        H, W = canvas_bgr.shape[:2]
        h, w = inset_bgr.shape[:2]
        margin = 8
        x1 = W - w - margin
        y1 = H - h - margin
        x2 = min(W, x1 + w)
        y2 = min(H, y1 + h)
        canvas_bgr[y1:y2, x1:x2] = inset_bgr[: y2 - y1, : x2 - x1]

    def _draw_center_guide(self, frame_bgr: U8, face_present: bool, compact: bool = False) -> None:
        h, w = frame_bgr.shape[:2]
        color = (0, 220, 0) if face_present else (0, 0, 255)
        thick = max(1, int(w / 100)) if compact else max(1, int(w / 50))

        cx = w // 2
        cy = int(h * 0.42)
        box_w = int(w * (0.34 if not compact else 0.42))
        box_h = int(h * (0.46 if not compact else 0.58))

        x1 = max(0, cx - box_w // 2)
        y1 = max(0, cy - box_h // 2)
        x2 = min(w - 1, cx + box_w // 2)
        y2 = min(h - 1, cy + box_h // 2)

        l = max(6, min(w, h) // 10)

        cv2.line(frame_bgr, (x1, y1), (x1 + l, y1), color, thick)
        cv2.line(frame_bgr, (x1, y1), (x1, y1 + l), color, thick)
        cv2.line(frame_bgr, (x2, y1), (x2 - l, y1), color, thick)
        cv2.line(frame_bgr, (x2, y1), (x2, y1 + l), color, thick)

        cv2.line(frame_bgr, (x1, y2), (x1 + l, y2), color, thick)
        cv2.line(frame_bgr, (x1, y2), (x1, y2 - l), color, thick)
        cv2.line(frame_bgr, (x2, y2), (x2 - l, y2), color, thick)
        cv2.line(frame_bgr, (x2, y2), (x2, y2 - l), color, thick)

        axes = (max(30, box_w // 3), max(40, box_h // 2))
        cv2.ellipse(frame_bgr, (cx, cy), axes, 0.0, 0.0, 360.0, color, thick)
        cv2.circle(frame_bgr, (cx, cy), 3 if not compact else 2, color, -1)

        label = "ROSTRO EN CUADRE" if face_present else "ACERCA TU ROSTRO"
        cv2.putText(
            frame_bgr,
            label,
            (max(12, x1), max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 if not compact else 0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    def _drain_render_slot(self) -> tuple[U8, int] | None:
        pkt, _fresh = self._render_slot.get(0.0)
        if pkt is None:
            return None
        result, frame_id = pkt
        return result.rgb, frame_id

    def _update_fps(self) -> None:
        now = time.perf_counter()
        if now - self._fps_last_update >= _FPS_LOG_INTERVAL_S:
            elapsed = now - self._fps_last_update
            self._fps_render = self._fps_render_count / elapsed if elapsed > 0 else 0.0
            self._fps_render_count = 0
            self._fps_last_update = now

    def _cleanup(self) -> None:
        self._stop_event.set()
        self._cap_slot.close()
        self._motion_slot.close()
        self._render_slot.close()

        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)

        if self._camera_supervisor is not None:
            self._camera_supervisor.release()
            self._camera_supervisor = None

        if self._window is not None:
            self._window.close()
            self._window = None


def _rgb_to_rgba(rgb: U8) -> U8:
    h, w = rgb.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = rgb
    out[:, :, 3] = 255
    return out


def _blend(prev: np.ndarray, new: np.ndarray, alpha_new: float) -> np.ndarray:
    return (alpha_new * new + (1.0 - alpha_new) * prev).astype(np.float32, copy=False)


def _bbox_max_edge(bbox_xyxy: np.ndarray) -> float:
    return float(
        max(
            float(bbox_xyxy[2] - bbox_xyxy[0]),
            float(bbox_xyxy[3] - bbox_xyxy[1]),
        )
    )


def _prepare_motion_input(crop_bgr: U8) -> U8:
    h, w = crop_bgr.shape[:2]
    max_edge = max(h, w)
    if max_edge <= _MOTION_INPUT_MAX_EDGE:
        return crop_bgr
    scale = _MOTION_INPUT_MAX_EDGE / float(max_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


class _MotionSmoother:
    def __init__(self) -> None:
        self._prev: MotionParams | None = None
        self._neutral_exp: F32 | None = None

    def set_neutral(self, neutral_exp: F32) -> None:
        self._neutral_exp = neutral_exp.astype(np.float32, copy=True)

    def smooth(self, params: MotionParams, confidence: float = 1.0) -> MotionParams:
        if self._prev is None:
            self._prev = params
            return params

        out_exp = _blend(self._prev.exp, params.exp, _ALPHA_EXP)

        if confidence < _LOWER_FACE_CONF_THRESHOLD and self._neutral_exp is not None:
            out_exp[:, _LOWER_FACE_KP_INDICES, :] = (
                0.85 * out_exp[:, _LOWER_FACE_KP_INDICES, :]
                + 0.15 * self._neutral_exp[:, _LOWER_FACE_KP_INDICES, :]
            )

        out = MotionParams(
            pitch=_blend(self._prev.pitch, params.pitch, _ALPHA_POSE),
            yaw=_blend(self._prev.yaw, params.yaw, _ALPHA_POSE),
            roll=_blend(self._prev.roll, params.roll, _ALPHA_POSE),
            t=_blend(self._prev.t, params.t, _ALPHA_T),
            exp=out_exp,
            scale=_blend(self._prev.scale, params.scale, _ALPHA_SCALE),
            kp_canonical=_blend(self._prev.kp_canonical, params.kp_canonical, _ALPHA_KP),
        )
        out.validate()
        self._prev = out
        return out
