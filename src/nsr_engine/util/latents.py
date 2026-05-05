# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Latent-space data contracts for NSR v0.1.0 — LivePortrait ONNX topology.

Pipeline (five stages, §11):

    Face detector crop (256x256 BGR u8)
        |
        |  source path (init only)         |  driving path (per frame)
        v                                   v
    AppearanceFeatureExtractor          MotionExtractor
        |                                   |
        |  AppearanceFeature3D               |  MotionParams
        |    (1, 32, 16, 64, 64) f32         |    (pitch, yaw, roll in degrees,
        |                                   |     t (1,3), exp (1,21,3),
        |                                   |     scale (1,1), kp_canonical (1,21,3))
        |                                   |
        |  [transform canonical kp to        |
        |   source kp via MotionParams_source; |
        |   driving kp via MotionParams_driving] |
        |         ^                         |
        |         |                         |
        |  SourceKeypoints                 DrivingKeypoints
        |   (1, 21, 3) f32                  (1, 21, 3) f32
        |                                   |
        |  [optional] StitchingRetargeting
        |      kp_driving := kp_driving + delta(kp_source, kp_driving)
        |                                   |
        +-------->  WarpingNetwork  <-------+
                         |
                         v
                   WarpedFeature3D
                     (1, 32, 16, 64, 64) f32
                         |
                         v
                   SpadeGenerator
                         |
                         v
                   RGB image (1, 3, 512, 512) f32 in [0, 1]
                         |
                         v
                   uint8 RGB OUTPUT_RES x OUTPUT_RES

Design notes
------------
- Implicit keypoints (21 x 3) are NOT legacy 2D landmarks. They are
  internal keypoints of the LivePortrait motion model, consumed only
  by the warping network and the stitching MLP. They are never used
  for explicit geometric operations at the Python level (no TPS, no
  cv2.remap).
- All feature-volume shapes are the published LivePortrait reference
  shapes. Validators in this module check them strictly. The ONNX
  loaders re-validate against the actual graph at load time and fail
  loudly on mismatch.
- Coordinate / normalization conventions:
    * Input image to encoder/motion: RGB float32 in [0, 1], NCHW.
    * Canonical kp scale/unit: implicit keypoint coordinate frame;
      not in pixels. Do NOT interpret these as image coordinates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from nsr_engine.util.typing import F32, U8

# ---------------------------------------------------------------------------
# Versioning and contract identity
# ---------------------------------------------------------------------------

# Runtime / implementation version of the engine package.
NSR_ENGINE_VERSION: Final[str] = "0.1.0"

# Model family / topology lineage.
NSR_MODEL_FAMILY: Final[str] = "liveportrait"

# Hard ABI contract version for latent-space tensors.
# This is the compatibility boundary enforced by tests, loaders, and validators.
NSR_LATENT_CONTRACT_VERSION: Final[str] = "0.1.0-liveportrait"

# Human-readable contract label for logs and diagnostics.
NSR_LATENT_CONTRACT_NAME: Final[str] = "nsr.latent.liveportrait"

# ---------------------------------------------------------------------------
# Processing resolutions and architectural constants
# ---------------------------------------------------------------------------

# Processing resolutions. Internal encoder/motion input is 256x256; the SPADE
# generator upscales to 512x512 natively (no cv2 upscale in the default path).
CROP_RES: Final[int] = 256
INTERNAL_RES: Final[int] = 256
OUTPUT_RES: Final[int] = 512

# LivePortrait reference architecture constants. Validated against the loaded
# ONNX graphs at init. These are the expected values; loaders must agree.
# Source: upstream LivePortrait (KwaiVGI/LivePortrait) base_models configs.
APPEARANCE_FEATURE_CHANNELS: Final[int] = 32
APPEARANCE_FEATURE_DEPTH: Final[int] = 16
APPEARANCE_FEATURE_H: Final[int] = 64
APPEARANCE_FEATURE_W: Final[int] = 64

# Warped feature volume is flattened to 4-D in this export family
# (depth and channels fused). The SPADE generator consumes this 4-D
# tensor directly; it is not the same shape as AppearanceFeature3D.
WARPED_FEATURE_CHANNELS: Final[int] = 256
WARPED_FEATURE_H: Final[int] = 64
WARPED_FEATURE_W: Final[int] = 64

NUM_KEYPOINTS: Final[int] = 21
KEYPOINT_DIMS: Final[int] = 3
KEYPOINT_FLAT: Final[int] = NUM_KEYPOINTS * KEYPOINT_DIMS  # 63

# Head pose bin count for pitch/yaw/roll classification outputs of the motion
# extractor. Upstream uses 66 bins spanning -99..99 degrees, step 3.
POSE_BINS: Final[int] = 66


def _shape_spec() -> dict[str, object]:
    """
    Deterministic latent ABI description used for fingerprinting.

    This is intentionally narrow: only properties that define compatibility
    are included.
    """
    return {
        "engine_version": NSR_ENGINE_VERSION,
        "model_family": NSR_MODEL_FAMILY,
        "contract_version": NSR_LATENT_CONTRACT_VERSION,
        "constants": {
            "crop_res": CROP_RES,
            "internal_res": INTERNAL_RES,
            "output_res": OUTPUT_RES,
            "appearance_feature": {
                "shape": (1, APPEARANCE_FEATURE_CHANNELS, APPEARANCE_FEATURE_DEPTH,
                          APPEARANCE_FEATURE_H, APPEARANCE_FEATURE_W),
                "dtype": "float32",
            },
            "motion_params": {
                "pitch": (1,),
                "yaw": (1,),
                "roll": (1,),
                "t": (1, 3),
                "exp": (1, NUM_KEYPOINTS, KEYPOINT_DIMS),
                "scale": (1, 1),
                "kp_canonical": (1, NUM_KEYPOINTS, KEYPOINT_DIMS),
            },
            "implicit_keypoints": {
                "shape": (1, NUM_KEYPOINTS, KEYPOINT_DIMS),
                "dtype": "float32",
            },
            "warped_feature": {
                "shape": (1, WARPED_FEATURE_CHANNELS, WARPED_FEATURE_H, WARPED_FEATURE_W),
                "dtype": "float32",
            },
            "crop": {
                "shape": (CROP_RES, CROP_RES, 3),
                "dtype": "uint8",
            },
            "render": {
                "shape": (OUTPUT_RES, OUTPUT_RES, 3),
                "dtype": "uint8",
            },
            "num_keypoints": NUM_KEYPOINTS,
            "keypoint_dims": KEYPOINT_DIMS,
            "pose_bins": POSE_BINS,
        },
    }


def compute_latent_contract_fingerprint() -> str:
    """
    Deterministic fingerprint of the latent ABI.

    Any change in compatibility-relevant shapes, dtypes, or semantic constants
    changes this digest.
    """
    raw = json.dumps(_shape_spec(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


NSR_LATENT_CONTRACT_FINGERPRINT: Final[str] = compute_latent_contract_fingerprint()


def assert_contract_compatibility() -> None:
    """
    Hard guardrail for startup and CI.

    This module is the source of truth for the latent ABI.
    """
    if NSR_ENGINE_VERSION != "0.1.0":
        raise RuntimeError(f"Unexpected engine version: {NSR_ENGINE_VERSION}")

    if NSR_MODEL_FAMILY != "liveportrait":
        raise RuntimeError(f"Unsupported model family: {NSR_MODEL_FAMILY}")

    if NSR_LATENT_CONTRACT_VERSION != "0.1.0-liveportrait":
        raise RuntimeError(
            "Latent ABI contract mismatch: "
            f"expected 0.1.0-liveportrait, got {NSR_LATENT_CONTRACT_VERSION}"
        )


@dataclass(frozen=True)
class AppearanceFeature3D:
    """Frozen 3D feature volume from the appearance feature extractor (§4.1).

    Produced ONCE per session on the avatar image at engine init.
    Consumed only by the warping network.
    """

    data: F32  # (1, C=32, D=16, H=64, W=64) float32

    def validate(self) -> None:
        if self.data.dtype != np.float32:
            raise TypeError(
                f"appearance_feature must be float32, got {self.data.dtype}"
            )
        if self.data.ndim != 5:
            raise ValueError(
                f"appearance_feature must be 5-D (1,C,D,H,W), got {self.data.shape}"
            )
        n, c, d, h, w = self.data.shape
        if n != 1:
            raise ValueError(f"appearance_feature batch must be 1, got {n}")
        if c != APPEARANCE_FEATURE_CHANNELS:
            raise ValueError(
                f"appearance_feature channels must be {APPEARANCE_FEATURE_CHANNELS}, "
                f"got {c}"
            )
        if d != APPEARANCE_FEATURE_DEPTH:
            raise ValueError(
                f"appearance_feature depth must be {APPEARANCE_FEATURE_DEPTH}, got {d}"
            )
        if (h, w) != (APPEARANCE_FEATURE_H, APPEARANCE_FEATURE_W):
            raise ValueError(
                f"appearance_feature spatial must be "
                f"({APPEARANCE_FEATURE_H},{APPEARANCE_FEATURE_W}), got ({h},{w})"
            )
        if not np.all(np.isfinite(self.data)):
            raise ValueError("appearance_feature contains non-finite values")


@dataclass(frozen=True)
class MotionParams:
    """Structured per-frame motion from the motion extractor (§4.2).

    This replaces the 1-D MotionLatent of the skeleton. Each field has
    the shape produced by the upstream LivePortrait motion_extractor,
    after the pose-bin softmax has been converted to Euler degrees.

    pitch, yaw, roll:  (1,) float32, Euler angles in degrees.
    t:                 (1, 3) float32, translation vector.
    exp:               (1, 21, 3) float32, per-keypoint expression delta.
    scale:             (1, 1) float32, global scale.
    kp_canonical:      (1, 21, 3) float32, canonical implicit keypoints.
    """

    pitch: F32
    yaw: F32
    roll: F32
    t: F32
    exp: F32
    scale: F32
    kp_canonical: F32

    def validate(self) -> None:
        _assert_shape("pitch", self.pitch, (1,), np.float32)
        _assert_shape("yaw", self.yaw, (1,), np.float32)
        _assert_shape("roll", self.roll, (1,), np.float32)
        _assert_shape("t", self.t, (1, 3), np.float32)
        _assert_shape("exp", self.exp, (1, NUM_KEYPOINTS, KEYPOINT_DIMS), np.float32)
        _assert_shape("scale", self.scale, (1, 1), np.float32)
        _assert_shape(
            "kp_canonical",
            self.kp_canonical,
            (1, NUM_KEYPOINTS, KEYPOINT_DIMS),
            np.float32,
        )
        for name, arr in (
            ("pitch", self.pitch),
            ("yaw", self.yaw),
            ("roll", self.roll),
            ("t", self.t),
            ("exp", self.exp),
            ("scale", self.scale),
            ("kp_canonical", self.kp_canonical),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"MotionParams.{name} contains non-finite values")


@dataclass(frozen=True)
class ImplicitKeypoints:
    """Transformed implicit keypoints (1, 21, 3) float32.

    Derived from MotionParams via the canonical transform:

        kp = scale * (kp_canonical @ R(pitch, yaw, roll).T + exp) + t

    The warping network consumes one of these for the source path and
    one for the driving path. The stitching MLP consumes both
    concatenated to produce a retargeting delta on kp_driving.

    These are *not* 2D landmarks. They live in the implicit-keypoint
    coordinate frame of the LivePortrait model.
    """

    data: F32  # (1, 21, 3) float32

    def validate(self) -> None:
        _assert_shape(
            "keypoints",
            self.data,
            (1, NUM_KEYPOINTS, KEYPOINT_DIMS),
            np.float32,
        )
        if not np.all(np.isfinite(self.data)):
            raise ValueError("implicit keypoints contain non-finite values")


@dataclass(frozen=True)
class WarpedFeature3D:
    """Output of the warping network — 4-D flattened feature map.

    Shape: (1, 256, 64, 64) in this export family. The warping ONNX
    flattens the 3D feature volume into a 4-D tensor before emitting it,
    so the SPADE generator can consume it as a standard 2-D feature map.
    The name `WarpedFeature3D` is retained for source-stability; the tensor
    is 4-D.
    """

    data: F32

    def validate(self) -> None:
        if self.data.dtype != np.float32:
            raise TypeError(
                f"warped_feature must be float32, got {self.data.dtype}"
            )
        if self.data.shape != (
            1,
            WARPED_FEATURE_CHANNELS,
            WARPED_FEATURE_H,
            WARPED_FEATURE_W,
        ):
            raise ValueError(
                f"warped_feature shape must be "
                f"(1,{WARPED_FEATURE_CHANNELS},"
                f"{WARPED_FEATURE_H},{WARPED_FEATURE_W}), "
                f"got {self.data.shape}"
            )
        if not np.all(np.isfinite(self.data)):
            raise ValueError("warped_feature contains non-finite values")


@dataclass(frozen=True)
class CropResult:
    """Face-detector-derived face crop in BGR uint8 at CROP_RES."""

    bgr: U8
    bbox_xyxy: F32
    score: float
    valid: bool

    def validate(self) -> None:
        if self.bgr.dtype != np.uint8:
            raise TypeError(f"crop.bgr must be uint8, got {self.bgr.dtype}")
        if self.bgr.shape != (CROP_RES, CROP_RES, 3):
            raise ValueError(
                f"crop.bgr must be ({CROP_RES},{CROP_RES},3), got {self.bgr.shape}"
            )
        if self.bbox_xyxy.shape != (4,):
            raise ValueError(f"crop.bbox_xyxy must be (4,), got {self.bbox_xyxy.shape}")

    @classmethod
    def invalid(cls) -> CropResult:
        return cls(
            bgr=np.zeros((CROP_RES, CROP_RES, 3), dtype=np.uint8),
            bbox_xyxy=np.zeros((4,), dtype=np.float32),
            score=0.0,
            valid=False,
        )


@dataclass(frozen=True)
class RenderResult:
    """Final synthesized frame. RGB per §2 system I/O contract."""

    rgb: U8
    internal_ms: float

    def validate(self) -> None:
        if self.rgb.dtype != np.uint8:
            raise TypeError(f"render.rgb must be uint8, got {self.rgb.dtype}")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"render.rgb must be HxWx3, got {self.rgb.shape}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_shape(
    name: str,
    arr: np.ndarray[Any, np.dtype[Any]],
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
) -> None:
    expected = np.dtype(dtype)
    if arr.dtype != expected:
        raise TypeError(f"{name} must be {expected}, got {arr.dtype}")
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
