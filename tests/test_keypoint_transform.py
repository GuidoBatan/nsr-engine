# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Unit tests for the implicit keypoint transform math.

Runs without any ONNX models. Verifies pose-bin decoding, rotation
matrix construction, canonical->implicit keypoint transform, and
stitching delta application.
"""

from __future__ import annotations

import numpy as np
import pytest

from nsr_engine.motion.keypoint_transform import (
    apply_stitching_delta,
    decode_pose_bins,
    rotation_matrix,
    transform_keypoints,
)
from nsr_engine.util.latents import (
    KEYPOINT_DIMS,
    NUM_KEYPOINTS,
    POSE_BINS,
    ImplicitKeypoints,
    MotionParams,
)


class TestDecodePoseBins:
    def test_uniform_gives_zero(self) -> None:
        logits = np.zeros((1, POSE_BINS), dtype=np.float32)
        degrees = decode_pose_bins(logits)
        assert degrees.shape == (1,)
        assert abs(float(degrees[0])) < 1e-4

    def test_one_hot_bin_zero_gives_min_degree(self) -> None:
        logits = np.full((1, POSE_BINS), -1e9, dtype=np.float32)
        logits[0, 0] = 0.0
        degrees = decode_pose_bins(logits)
        assert abs(float(degrees[0]) - (-97.5)) < 1e-3

    def test_one_hot_bin_last_gives_max_degree(self) -> None:
        logits = np.full((1, POSE_BINS), -1e9, dtype=np.float32)
        logits[0, POSE_BINS - 1] = 0.0
        degrees = decode_pose_bins(logits)
        # (POSE_BINS - 1) * 3 - 97.5 = 65 * 3 - 97.5 = 97.5
        assert abs(float(degrees[0]) - 97.5) < 1e-3

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            decode_pose_bins(np.zeros((1, 30), dtype=np.float32))


class TestRotationMatrix:
    def test_zero_gives_identity(self) -> None:
        R = rotation_matrix(
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        )
        assert R.shape == (1, 3, 3)
        assert np.allclose(R[0], np.eye(3), atol=1e-6)

    def test_orthogonality(self) -> None:
        R = rotation_matrix(
            np.array([15.0], dtype=np.float32),
            np.array([-20.0], dtype=np.float32),
            np.array([7.0], dtype=np.float32),
        )
        RtR = R[0] @ R[0].T
        assert np.allclose(RtR, np.eye(3), atol=1e-5)

    def test_det_is_one(self) -> None:
        R = rotation_matrix(
            np.array([30.0], dtype=np.float32),
            np.array([45.0], dtype=np.float32),
            np.array([-10.0], dtype=np.float32),
        )
        assert abs(float(np.linalg.det(R[0])) - 1.0) < 1e-5

    def test_90_deg_yaw(self) -> None:
        # Yaw is R_y; rotates x-axis to -z-axis (matching np convention below).
        R = rotation_matrix(
            np.array([0.0], dtype=np.float32),
            np.array([90.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        )
        expected = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        assert np.allclose(R[0], expected, atol=1e-5)

    def test_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError):
            rotation_matrix(
                np.array([[0.0]], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )


class TestTransformKeypoints:
    def _neutral_params(self, kp_canonical: np.ndarray) -> MotionParams:
        return MotionParams(
            pitch=np.zeros((1,), dtype=np.float32),
            yaw=np.zeros((1,), dtype=np.float32),
            roll=np.zeros((1,), dtype=np.float32),
            t=np.zeros((1, 3), dtype=np.float32),
            exp=np.zeros((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32),
            scale=np.ones((1, 1), dtype=np.float32),
            kp_canonical=kp_canonical.astype(np.float32),
        )

    def test_neutral_params_preserve_canonical(self) -> None:
        rng = np.random.default_rng(0)
        kp = rng.standard_normal((1, NUM_KEYPOINTS, KEYPOINT_DIMS)).astype(np.float32)
        params = self._neutral_params(kp)
        out = transform_keypoints(params)
        assert np.allclose(out.data, kp, atol=1e-6)

    def test_scale_applied_before_translation(self) -> None:
        kp = np.ones((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32)
        params = MotionParams(
            pitch=np.zeros((1,), dtype=np.float32),
            yaw=np.zeros((1,), dtype=np.float32),
            roll=np.zeros((1,), dtype=np.float32),
            t=np.array([[10.0, 20.0, 999.0]], dtype=np.float32),
            exp=np.zeros((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32),
            scale=np.array([[2.0]], dtype=np.float32),
            kp_canonical=kp,
        )
        out = transform_keypoints(params)
        # Expected: scale * kp (=2) + t_xy0 (=[10, 20, 0])
        expected = np.zeros_like(kp)
        expected[..., 0] = 2.0 + 10.0
        expected[..., 1] = 2.0 + 20.0
        expected[..., 2] = 2.0 + 0.0  # t_z zeroed
        assert np.allclose(out.data, expected, atol=1e-5)

    def test_expression_adds_after_rotation(self) -> None:
        kp = np.zeros((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32)
        exp = np.ones((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32)
        params = MotionParams(
            pitch=np.zeros((1,), dtype=np.float32),
            yaw=np.zeros((1,), dtype=np.float32),
            roll=np.zeros((1,), dtype=np.float32),
            t=np.zeros((1, 3), dtype=np.float32),
            exp=exp,
            scale=np.ones((1, 1), dtype=np.float32),
            kp_canonical=kp,
        )
        out = transform_keypoints(params)
        assert np.allclose(out.data, exp, atol=1e-6)


class TestApplyStitchingDelta:
    def test_adds_delta(self) -> None:
        kp_driving = ImplicitKeypoints(
            data=np.zeros((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32)
        )
        delta = np.ones((1, NUM_KEYPOINTS * KEYPOINT_DIMS), dtype=np.float32)
        out = apply_stitching_delta(kp_driving, delta)
        assert np.allclose(
            out.data, np.ones((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32),
            atol=1e-6,
        )

    def test_ignores_trailing_entries(self) -> None:
        kp_driving = ImplicitKeypoints(
            data=np.zeros((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32)
        )
        # 65 = 63 kp delta + 2 retargeting scalars (eye, lip).
        delta = np.concatenate(
            [
                np.ones((1, NUM_KEYPOINTS * KEYPOINT_DIMS), dtype=np.float32),
                np.full((1, 2), 999.0, dtype=np.float32),
            ],
            axis=1,
        )
        out = apply_stitching_delta(kp_driving, delta)
        # Only the first 63 should be used; trailing 999.0 must not leak.
        assert float(out.data.max()) == 1.0
        assert float(out.data.min()) == 1.0

    def test_rejects_too_narrow(self) -> None:
        kp_driving = ImplicitKeypoints(
            data=np.zeros((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32)
        )
        with pytest.raises(ValueError):
            apply_stitching_delta(
                kp_driving, np.zeros((1, 10), dtype=np.float32)
            )

    def test_does_not_mutate_input(self) -> None:
        data = np.ones((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32)
        kp_driving = ImplicitKeypoints(data=data.copy())
        delta = np.ones((1, NUM_KEYPOINTS * KEYPOINT_DIMS), dtype=np.float32)
        _ = apply_stitching_delta(kp_driving, delta)
        assert np.array_equal(kp_driving.data, data)
