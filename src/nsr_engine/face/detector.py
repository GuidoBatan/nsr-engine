# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Face detection module using MediaPipe Tasks Vision API.

Replaces InsightFace SCRFD (det_10g.onnx) with MediaPipe for Apache 2.0 compliance.

This module provides a drop-in replacement for the previous SCRFD-based
face detector. The public API (detect method signature and output format)
is preserved: detect() returns (N, 5) float32 arrays of [x1, y1, x2, y2, score]
in the original frame coordinate system.

MediaPipe Tasks FaceDetector returns bounding boxes in absolute pixel
coordinates (origin_x, origin_y, width, height). We convert to
[x1, y1, x2, y2, score] to match the downstream contract.

The bundled short-range TFLite model is resolved automatically from the
mediapipe package installation directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from nsr_engine.util.typing import F32, U8

_MP_MODULE_DIR: Path = Path(mp.__file__).resolve().parent / "modules" / "face_detection"
_SHORT_RANGE_MODEL: str = str(_MP_MODULE_DIR / "face_detection_short_range.tflite")
_FULL_RANGE_MODEL: str = str(
    _MP_MODULE_DIR / "face_detection_full_range_sparse.tflite"
)


def _resolve_model_path(model_selection: int) -> str:
    """Return the absolute path to the bundled MediaPipe TFLite model.

    model_selection 0 or 1 both use the short-range model, which is
    compatible with the Tasks Vision API and bundled with mediapipe.
    The full-range sparse model is not compatible with the Tasks API
    calculator graph in mediapipe <=0.10.14.
    """
    path = _SHORT_RANGE_MODEL
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"MediaPipe face detection model not found at {path}. "
            "Reinstall mediapipe: pip install mediapipe>=0.10.14"
        )
    return path


class FaceDetector:
    """MediaPipe Tasks-based face detector. Returns (N, 5) bbox+score arrays.

    Drop-in replacement for the previous SCRFD class. The detect() method
    signature and return format are identical.
    """

    def __init__(
        self,
        model_selection: int = 1,
        min_detection_confidence: float = 0.5,
    ) -> None:
        model_path = _resolve_model_path(model_selection)
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def detect(
        self,
        frame_bgr: U8,
        score_threshold: float,
        nms_iou: float,
    ) -> F32:
        """Return (N, 5) array of [x1, y1, x2, y2, score] in ORIGINAL frame coords.

        Empty detection returns shape (0, 5).

        Parameters
        ----------
        frame_bgr : numpy array
            HxWx3 BGR uint8 image.
        score_threshold : float
            Minimum detection score to keep.
        nms_iou : float
            Unused (MediaPipe performs its own NMS), kept for API compatibility.
        """
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                f"FaceDetector expects HxWx3 BGR uint8, got {frame_bgr.shape}"
            )

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.detections:
            return np.zeros((0, 5), dtype=np.float32)

        boxes: list[F32] = []
        for det in result.detections:
            score = det.categories[0].score if det.categories else 0.0
            if score < score_threshold:
                continue
            bb = det.bounding_box
            x1 = float(bb.origin_x)
            y1 = float(bb.origin_y)
            x2 = float(bb.origin_x + bb.width)
            y2 = float(bb.origin_y + bb.height)
            boxes.append(np.array([x1, y1, x2, y2, score], dtype=np.float32))

        if not boxes:
            return np.zeros((0, 5), dtype=np.float32)

        return np.stack(boxes, axis=0).astype(np.float32)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()

    def __del__(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self.close()
