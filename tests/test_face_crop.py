# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Tests for the face crop helper `_crop_square_padded`.

Runs without any ONNX models — the padded crop is pure numpy/cv2.
"""

from __future__ import annotations

import numpy as np

from nsr_engine.face.cropper import _crop_square_padded
from nsr_engine.util.latents import CROP_RES


class TestCropSquarePadded:
    def test_output_shape_and_dtype(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 300, 300], dtype=np.float32)
        out = _crop_square_padded(frame, bbox, 0.25, CROP_RES)
        assert out.shape == (CROP_RES, CROP_RES, 3)
        assert out.dtype == np.uint8

    def test_centre_pixel_preserved(self) -> None:
        """A bright pixel at the bbox centre must survive a padded crop+resize."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([200, 200, 400, 400], dtype=np.float32)  # centre (300, 300)
        frame[299:302, 299:302] = np.array([10, 20, 200], dtype=np.uint8)
        out = _crop_square_padded(frame, bbox, 0.25, CROP_RES)
        cy, cx = CROP_RES // 2, CROP_RES // 2
        # After INTER_LINEAR the exact integer values may have been smoothed;
        # assert the red channel at the centre is meaningfully elevated.
        assert int(out[cy, cx, 2]) > 50, f"centre pixel lost: {out[cy, cx]}"

    def test_letterbox_when_bbox_exits_frame(self) -> None:
        """bbox off the right edge must produce black padding, not crash."""
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        bbox = np.array([600, 200, 700, 300], dtype=np.float32)  # partly outside
        out = _crop_square_padded(frame, bbox, 0.25, CROP_RES)
        assert out.shape == (CROP_RES, CROP_RES, 3)
        # Some black must appear where the bbox exited the frame.
        assert (out == 0).any()

    def test_padding_frac_zero_gives_tight_crop(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Mark the tight bbox region white.
        frame[200:400, 100:300] = 255
        bbox = np.array([100, 200, 300, 400], dtype=np.float32)
        out = _crop_square_padded(frame, bbox, 0.0, CROP_RES)
        # With zero padding the crop should be uniformly near-white.
        assert out.mean() > 200
