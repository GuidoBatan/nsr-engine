# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""OBS Window Capture target.

Plain cv2 window with a stable title. OBS reads via "Window Capture"
source. Headless mode is a no-op sink. We intentionally do NOT expose a
virtual-camera path — that would drag in extra deps (pyvirtualcam)
outside the minimal `numpy / opencv / onnxruntime` stack.
"""

from __future__ import annotations

import cv2

from nsr_engine.util.typing import U8

WINDOW_TITLE: str = "NSR Engine"


class OBSWindow:
    def __init__(self, width: int, height: int, headless: bool = False) -> None:
        self._headless = bool(headless)
        self._w = int(width)
        self._h = int(height)
        if not self._headless:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_TITLE, self._w, self._h)

    def show(self, rgba: U8) -> None:
        if self._headless:
            return
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        if bgra.shape[1] != self._w or bgra.shape[0] != self._h:
            bgra = cv2.resize(bgra, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(WINDOW_TITLE, bgra)

    def poll_quit(self) -> bool:
        """Return True if user requested exit (q/esc or window closed)."""
        if self._headless:
            return False
        key = cv2.waitKey(1) & 0xFF
        return (
            key == ord("q")
            or key == 27
            or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
        )

    def close(self) -> None:
        if not self._headless:
            cv2.destroyAllWindows()
