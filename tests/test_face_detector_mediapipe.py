# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Integration test: MediaPipe FaceDetector initialises and returns expected format."""

from __future__ import annotations

import numpy as np

from nsr_engine.face.detector import FaceDetector


class TestFaceDetectorMediaPipe:
    def test_detector_initialises(self) -> None:
        """Verify MediaPipe FaceDetector can be constructed."""
        detector = FaceDetector()
        assert detector is not None

    def test_detect_returns_correct_shape(self) -> None:
        """Verify detect() returns (N, 5) float32 array."""
        detector = FaceDetector()
        # Synthetic grey image — may or may not contain a face detection.
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        detections = detector.detect(image, score_threshold=0.5, nms_iou=0.4)
        assert isinstance(detections, np.ndarray)
        assert detections.dtype == np.float32
        assert detections.ndim == 2
        assert detections.shape[1] == 5

    def test_detect_empty_on_blank_image(self) -> None:
        """A solid black image should produce zero detections."""
        detector = FaceDetector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(image, score_threshold=0.5, nms_iou=0.4)
        assert detections.shape == (0, 5)

    def test_detect_rejects_bad_input(self) -> None:
        """Non-3-channel input should raise ValueError."""
        detector = FaceDetector()
        image = np.zeros((480, 640), dtype=np.uint8)
        raised = False
        try:
            detector.detect(image, score_threshold=0.5, nms_iou=0.4)
        except ValueError:
            raised = True
        assert raised, "Should have raised ValueError"
