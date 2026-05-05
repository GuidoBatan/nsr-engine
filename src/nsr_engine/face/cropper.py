# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Face cropping built on MediaPipe FaceDetector.

This is the *only* geometric operation in v1.0.0. Its job is narrow:
given a webcam BGR frame, produce a CROP_RES x CROP_RES BGR crop
centred on the face, padded per `FacePipelineConfig.crop_padding_frac`.

No landmarks, no pose, no rigid/non-rigid decomposition. Everything
downstream consumes the crop and treats it as an opaque identity +
motion source.

Detection throttling (§10 real-time): the face detector is called every
`detect_every_n` frames. Between detections the previous bbox is
re-cropped against the *current* webcam frame, so the face stays
centred during motion even when we skip a detection. This is the
same throttling strategy as v0.3.1 but without landmarks running
on off-frames.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from nsr_engine.config import FacePipelineConfig
from nsr_engine.face.detector import FaceDetector
from nsr_engine.util.latents import CROP_RES, CropResult
from nsr_engine.util.typing import F32, U8

logger = logging.getLogger("nsr.face.crop")


class FaceCropper:
    """Stateful face cropper. Keeps the last valid bbox for throttled re-crop."""

    def __init__(self, cfg: FacePipelineConfig, detect_every_n: int) -> None:
        if detect_every_n < 1:
            raise ValueError(f"detect_every_n must be >= 1, got {detect_every_n}")
        self._cfg = cfg
        self._detector = FaceDetector(
            model_selection=1,
            min_detection_confidence=cfg.det_score_threshold,
        )
        self._detect_every_n = int(detect_every_n)
        self._frame_idx: int = 0
        self._last_bbox: F32 | None = None
        self._last_score: float = 0.0

    def reset(self) -> None:
        self._frame_idx = 0
        self._last_bbox = None
        self._last_score = 0.0

    def crop(self, frame_bgr: U8) -> CropResult:
        """Produce a `CropResult`. Invalid result if no face has ever been seen."""
        run_detect = (self._last_bbox is None) or (
            self._frame_idx % self._detect_every_n == 0
        )
        self._frame_idx += 1

        if run_detect:
            dets = self._detector.detect(
                frame_bgr,
                self._cfg.det_score_threshold,
                self._cfg.det_nms_iou,
            )
            if dets.shape[0] > 0:
                # Pick the largest-area detection (most likely the primary face).
                areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
                best = int(np.argmax(areas))
                self._last_bbox = dets[best, :4].astype(np.float32, copy=True)
                self._last_score = float(dets[best, 4])

        if self._last_bbox is None:
            return CropResult(
                bgr=np.zeros((CROP_RES, CROP_RES, 3), dtype=np.uint8),
                bbox_xyxy=np.zeros((4,), dtype=np.float32),
                score=0.0,
                valid=False,
            )

        crop = _crop_square_padded(
            frame_bgr,
            self._last_bbox,
            self._cfg.crop_padding_frac,
            CROP_RES,
        )
        result = CropResult(
            bgr=crop,
            bbox_xyxy=self._last_bbox.copy(),
            score=self._last_score,
            valid=True,
        )
        result.validate()
        return result


def _crop_square_padded(
    frame_bgr: U8,
    bbox_xyxy: F32,
    pad_frac: float,
    out_size: int,
) -> U8:
    """Centre-crop a square region around `bbox_xyxy` and resize to `out_size`.

    Strategy:
      1. Compute the target square region around the bbox centre.
      2. Clamp the region to frame bounds.
      3. Extract the valid (clamped) sub-rectangle via array slicing only.
      4. Pad with cv2.copyMakeBorder (BORDER_CONSTANT, value=0) to restore
         the target square shape.
      5. cv2.resize to (out_size, out_size) if necessary.

    No manual slice assignment. Output shape is guaranteed
    (out_size, out_size, 3) uint8.
    """
    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * pad_frac)
    half = 0.5 * side

    # Target square region in source-frame coordinates (may exit frame).
    sx1 = int(round(cx - half))
    sy1 = int(round(cy - half))
    sx2 = int(round(cx + half))
    sy2 = int(round(cy + half))
    side_px = max(1, sx2 - sx1)

    # Clamp to frame bounds.
    ix1 = max(0, sx1)
    iy1 = max(0, sy1)
    ix2 = min(frame_w, sx2)
    iy2 = min(frame_h, sy2)

    # Pad amounts on each side (black borders for out-of-frame regions).
    pad_top = max(0, iy1 - sy1)
    pad_bottom = max(0, sy2 - iy2)
    pad_left = max(0, ix1 - sx1)
    pad_right = max(0, sx2 - ix2)

    if ix2 > ix1 and iy2 > iy1:
        valid = frame_bgr[iy1:iy2, ix1:ix2]
    else:
        # No intersection with frame; synthesize a 1x1 black pixel so
        # copyMakeBorder has a non-empty input. We correct via resize below.
        valid = np.zeros((1, 1, 3), dtype=np.uint8)
        pad_top = pad_bottom = pad_left = pad_right = 0

    padded = cv2.copyMakeBorder(
        valid,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # Guarantee the canvas is (side_px, side_px, 3). If sums-of-valid-plus-
    # padding don't line up (can happen with degenerate bboxes), force it
    # via resize to the expected square before the final resize.
    if padded.shape[:2] != (side_px, side_px):
        padded = cv2.resize(padded, (side_px, side_px), interpolation=cv2.INTER_LINEAR)

    if padded.shape[:2] != (out_size, out_size):
        padded = cv2.resize(padded, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    return padded
