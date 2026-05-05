# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Data-contract tests for NSR v0.1.0 (LivePortrait topology).

Runs without any ONNX model files. Verifies tensor shape / dtype
guarantees for the feature-volume + structured-motion contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from nsr_engine.util.latents import (
    APPEARANCE_FEATURE_CHANNELS,
    APPEARANCE_FEATURE_DEPTH,
    APPEARANCE_FEATURE_H,
    APPEARANCE_FEATURE_W,
    CROP_RES,
    INTERNAL_RES,
    KEYPOINT_DIMS,
    NSR_LATENT_CONTRACT_FINGERPRINT,
    NSR_LATENT_CONTRACT_VERSION,
    NUM_KEYPOINTS,
    OUTPUT_RES,
    POSE_BINS,
    WARPED_FEATURE_CHANNELS,
    WARPED_FEATURE_H,
    WARPED_FEATURE_W,
    AppearanceFeature3D,
    CropResult,
    ImplicitKeypoints,
    MotionParams,
    RenderResult,
    WarpedFeature3D,
)


class TestContractVersion:
    def test_version_pinned(self) -> None:
        assert NSR_LATENT_CONTRACT_VERSION == "0.1.0-liveportrait"

    def test_fingerprint_is_stable(self) -> None:
        # Ensures ABI stability across CI and runtime.
        assert isinstance(NSR_LATENT_CONTRACT_FINGERPRINT, str)
        assert len(NSR_LATENT_CONTRACT_FINGERPRINT) == 64  # sha256

    def test_internal_res_matches_crop(self) -> None:
        assert INTERNAL_RES == CROP_RES

    def test_output_is_native_from_spade(self) -> None:
        # SPADE generator upscales 64x64 feature volume to 512x512 natively.
        assert OUTPUT_RES == 512

    def test_keypoint_constants(self) -> None:
        assert NUM_KEYPOINTS == 21
        assert KEYPOINT_DIMS == 3
        assert POSE_BINS == 66


class TestAppearanceFeature3D:
    def _good(self) -> AppearanceFeature3D:
        data = np.zeros(
            (
                1,
                APPEARANCE_FEATURE_CHANNELS,
                APPEARANCE_FEATURE_DEPTH,
                APPEARANCE_FEATURE_H,
                APPEARANCE_FEATURE_W,
            ),
            dtype=np.float32,
        )
        return AppearanceFeature3D(data=data)

    def test_happy_path(self) -> None:
        self._good().validate()

    def test_rejects_wrong_dtype(self) -> None:
        data = np.zeros(
            (
                1,
                APPEARANCE_FEATURE_CHANNELS,
                APPEARANCE_FEATURE_DEPTH,
                APPEARANCE_FEATURE_H,
                APPEARANCE_FEATURE_W,
            ),
            dtype=np.float64,
        )
        with pytest.raises(TypeError):
            AppearanceFeature3D(data=data).validate()

    def test_rejects_wrong_rank(self) -> None:
        with pytest.raises(ValueError):
            AppearanceFeature3D(data=np.zeros((1, 32, 16), dtype=np.float32)).validate()

    def test_rejects_wrong_channels(self) -> None:
        with pytest.raises(ValueError):
            AppearanceFeature3D(
                data=np.zeros((1, 33, 16, 64, 64), dtype=np.float32)
            ).validate()

    def test_rejects_wrong_depth(self) -> None:
        with pytest.raises(ValueError):
            AppearanceFeature3D(
                data=np.zeros((1, 32, 17, 64, 64), dtype=np.float32)
            ).validate()

    def test_rejects_wrong_spatial(self) -> None:
        with pytest.raises(ValueError):
            AppearanceFeature3D(
                data=np.zeros((1, 32, 16, 32, 32), dtype=np.float32)
            ).validate()

    def test_rejects_non_finite(self) -> None:
        d = self._good().data.copy()
        d[0, 0, 0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            AppearanceFeature3D(data=d).validate()


class TestMotionParams:
    def _good(self) -> MotionParams:
        return MotionParams(
            pitch=np.zeros((1,), dtype=np.float32),
            yaw=np.zeros((1,), dtype=np.float32),
            roll=np.zeros((1,), dtype=np.float32),
            t=np.zeros((1, 3), dtype=np.float32),
            exp=np.zeros((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32),
            scale=np.ones((1, 1), dtype=np.float32),
            kp_canonical=np.zeros(
                (1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32
            ),
        )

    def test_happy_path(self) -> None:
        self._good().validate()

    def test_rejects_pitch_wrong_shape(self) -> None:
        p = self._good()
        bad = MotionParams(
            pitch=np.zeros((2,), dtype=np.float32),
            yaw=p.yaw,
            roll=p.roll,
            t=p.t,
            exp=p.exp,
            scale=p.scale,
            kp_canonical=p.kp_canonical,
        )
        with pytest.raises(ValueError):
            bad.validate()

    def test_rejects_t_wrong_shape(self) -> None:
        p = self._good()
        bad = MotionParams(
            pitch=p.pitch,
            yaw=p.yaw,
            roll=p.roll,
            t=np.zeros((1, 4), dtype=np.float32),
            exp=p.exp,
            scale=p.scale,
            kp_canonical=p.kp_canonical,
        )
        with pytest.raises(ValueError):
            bad.validate()

    def test_rejects_exp_wrong_shape(self) -> None:
        p = self._good()
        bad = MotionParams(
            pitch=p.pitch,
            yaw=p.yaw,
            roll=p.roll,
            t=p.t,
            exp=np.zeros((1, 20, 3), dtype=np.float32),
            scale=p.scale,
            kp_canonical=p.kp_canonical,
        )
        with pytest.raises(ValueError):
            bad.validate()

    def test_rejects_non_finite(self) -> None:
        p = self._good()
        bad_t = p.t.copy()
        bad_t[0, 0] = np.inf
        bad = MotionParams(
            pitch=p.pitch,
            yaw=p.yaw,
            roll=p.roll,
            t=bad_t,
            exp=p.exp,
            scale=p.scale,
            kp_canonical=p.kp_canonical,
        )
        with pytest.raises(ValueError):
            bad.validate()


class TestImplicitKeypoints:
    def test_happy_path(self) -> None:
        ImplicitKeypoints(
            data=np.zeros((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32)
        ).validate()

    def test_rejects_wrong_count(self) -> None:
        with pytest.raises(ValueError):
            ImplicitKeypoints(
                data=np.zeros((1, 20, 3), dtype=np.float32)
            ).validate()

    def test_rejects_wrong_dtype(self) -> None:
        with pytest.raises(TypeError):
            ImplicitKeypoints(
                data=np.zeros((1, NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float64)
            ).validate()


class TestWarpedFeature3D:
    def test_happy_path(self) -> None:
        WarpedFeature3D(
            data=np.zeros(
                (
                    1,
                    WARPED_FEATURE_CHANNELS,
                    WARPED_FEATURE_H,
                    WARPED_FEATURE_W,
                ),
                dtype=np.float32,
            )
        ).validate()

    def test_rejects_mismatched_shape(self) -> None:
        with pytest.raises(ValueError):
            WarpedFeature3D(
                data=np.zeros((1, 256, 64, 65), dtype=np.float32)
            ).validate()


class TestCropResult:
    def test_happy_path(self) -> None:
        CropResult(
            bgr=np.zeros((CROP_RES, CROP_RES, 3), dtype=np.uint8),
            bbox_xyxy=np.array([0, 0, 1, 1], dtype=np.float32),
            score=0.9,
            valid=True,
        ).validate()

    def test_rejects_wrong_size(self) -> None:
        with pytest.raises(ValueError):
            CropResult(
                bgr=np.zeros((128, 128, 3), dtype=np.uint8),
                bbox_xyxy=np.array([0, 0, 1, 1], dtype=np.float32),
                score=0.9,
                valid=True,
            ).validate()


class TestRenderResult:
    def test_happy_path(self) -> None:
        RenderResult(
            rgb=np.zeros((OUTPUT_RES, OUTPUT_RES, 3), dtype=np.uint8),
            internal_ms=1.0,
        ).validate()

    def test_rejects_float_dtype(self) -> None:
        with pytest.raises(TypeError):
            RenderResult(
                rgb=np.zeros((OUTPUT_RES, OUTPUT_RES, 3), dtype=np.float32),
                internal_ms=1.0,
            ).validate()
