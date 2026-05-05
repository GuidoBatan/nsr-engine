# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import cast

import cv2

from nsr_engine.util.typing import U8

logger = logging.getLogger("nsr.capture")

_IS_WINDOWS = sys.platform == "win32"

_PROBE_FRAMES = 20
_PROBE_TIMEOUT_S = 1.5
_DEVICE_RANGE = 6


def _backends() -> list[int]:
    if _IS_WINDOWS:
        return [cv2.CAP_DSHOW, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def _try_open(dev: int, backend: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(int(dev), backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _score(dev: int, width: int, height: int) -> float | None:
    for backend in _backends():
        cap = _try_open(dev, backend)
        if cap is None:
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ok_frames = 0
        lat: list[float] = []

        t_end = time.perf_counter() + _PROBE_TIMEOUT_S

        for _ in range(_PROBE_FRAMES):
            if time.perf_counter() > t_end:
                break

            t0 = time.perf_counter()
            ok, frame = cap.read()
            dt = (time.perf_counter() - t0) * 1000.0

            if ok and frame is not None:
                ok_frames += 1
                lat.append(dt)

        cap.release()

        if ok_frames < 5:
            continue

        return ok_frames / (1.0 + (sum(lat) / len(lat)))

    return None


def _select(width: int, height: int, preferred: int) -> int:
    best_dev: int | None = None
    best_score: float = -1.0

    for dev in range(_DEVICE_RANGE):
        sc = _score(dev, width, height)
        if sc is None:
            continue
        if sc > best_score:
            best_score = sc
            best_dev = dev

    if best_dev is None:
        best_dev = int(preferred)

    logger.info("camera selected device=%s", best_dev)
    return best_dev


class WebcamCapture:
    def __init__(self, device: int, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._device = _select(width, height, device)

        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()

        self._frame: U8 | None = None
        self._ts = 0.0

        self._running = True

        self._open()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open(self) -> None:
        for backend in _backends():
            cap = _try_open(self._device, backend)
            if cap:
                self._cap = cap
                logger.info("camera opened dev=%s backend=%s", self._device, backend)
                return

        logger.warning("camera open failed, retrying...")
        self._cap = None

    def _reopen(self) -> None:
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass
        self._open()

    def _loop(self) -> None:
        fail_count = 0

        while self._running:
            if self._cap is None:
                time.sleep(0.5)
                self._reopen()
                continue

            try:
                ok, frame = self._cap.read()

                if not ok or frame is None:
                    fail_count += 1
                    if fail_count > 10:
                        self._reopen()
                        fail_count = 0
                    time.sleep(0.01)
                    continue

                fail_count = 0

                with self._lock:
                    # cv2.VideoCapture.read returns BGR uint8 at runtime;
                    # cv2 stubs widen the dtype to `integer | floating`.
                    self._frame = cast(U8, frame)
                    self._ts = time.perf_counter()

            except Exception:
                self._reopen()
                time.sleep(0.05)

    def read(self) -> tuple[U8 | None, float]:
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame, self._ts

    def release(self) -> None:
        self._running = False
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass
