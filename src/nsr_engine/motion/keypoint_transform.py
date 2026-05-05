# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Implicit keypoint transform and motion extractor output decoding.

Transforms canonical implicit keypoints into source/driving keypoints
using the pose / expression / scale / translation outputs of the
motion extractor.

Reference: upstream LivePortrait (KwaiVGI/LivePortrait),
`src/utils/helper.py` (headpose_pred_to_degree, get_rotation_matrix)
and `src/live_portrait_wrapper.py` (transform_keypoint).

Pose-bin decoding
-----------------
The motion extractor emits pitch/yaw/roll as a softmax over 66 bins.
Each bin represents 3 degrees, spanning [-99, 99). Reference formula:

    degree = sum(softmax(pred) * idx) * 3 - 97.5

This uniquely pins the bin convention; changing the 3-degree step or
the 97.5 offset silently shifts the entire head-pose distribution.

Keypoint transform
------------------
    R = R_z(roll) @ R_y(yaw) @ R_x(pitch)      (ZYX intrinsic order, degrees)
    kp = kp_canonical @ R.T + exp
    kp = scale * kp
    kp = kp + t                                 (only x,y translated; t_z == 0)

Note on translation: upstream zeros the z-component of t before
adding (`t_new[..., 2] = 0`), because the motion extractor predicts
a 3-D t but the 3-D feature volume has no meaningful z-offset.
"""

from __future__ import annotations

from typing import Final, cast

import numpy as np

from nsr_engine.util.latents import (
    KEYPOINT_DIMS,
    NUM_KEYPOINTS,
    POSE_BINS,
    ImplicitKeypoints,
    MotionParams,
)
from nsr_engine.util.typing import F32

# Pose-bin decoding constants (upstream-fixed).
_POSE_BIN_DEG: Final[float] = 3.0
_POSE_BIN_OFFSET: Final[float] = 97.5
_POSE_IDX: Final[F32] = np.arange(POSE_BINS, dtype=np.float32)


def decode_pose_bins(logits: F32) -> F32:
    """Convert (1, 66) pose logits to a (1,) degree prediction.

    Expected logits shape: (1, 66). Returns (1,) float32 degrees.
    """
    if logits.shape != (1, POSE_BINS):
        raise ValueError(
            f"pose logits must have shape (1, {POSE_BINS}), got {logits.shape}"
        )
    # Softmax along the bin axis.
    x = logits.astype(np.float32, copy=False)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    probs = e / np.sum(e, axis=1, keepdims=True)
    # Weighted sum over bins, then scale + offset.
    degrees = np.sum(probs * _POSE_IDX[None, :], axis=1) * _POSE_BIN_DEG - _POSE_BIN_OFFSET
    return cast(F32, degrees.astype(np.float32, copy=False))


def rotation_matrix(pitch_deg: F32, yaw_deg: F32, roll_deg: F32) -> F32:
    """Build a (1, 3, 3) rotation matrix from Euler degrees.

    Uses the ZYX intrinsic convention matching upstream
    `get_rotation_matrix(pitch, yaw, roll)` in
    KwaiVGI/LivePortrait `src/utils/helper.py`:

        R = R_z(roll) @ R_y(yaw) @ R_x(pitch)

    Inputs: each (1,) float32 in degrees.
    Output: (1, 3, 3) float32.
    """
    if pitch_deg.shape != (1,) or yaw_deg.shape != (1,) or roll_deg.shape != (1,):
        raise ValueError(
            "pitch, yaw, roll must each have shape (1,); "
            f"got {pitch_deg.shape}, {yaw_deg.shape}, {roll_deg.shape}"
        )

    deg2rad = np.float32(np.pi / 180.0)
    p = pitch_deg * deg2rad
    y = yaw_deg * deg2rad
    r = roll_deg * deg2rad

    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    cr, sr = np.cos(r), np.sin(r)

    zero = np.zeros_like(cp)
    one = np.ones_like(cp)

    # R_x(pitch)
    rx = np.stack([
        np.stack([one, zero, zero], axis=-1),
        np.stack([zero, cp, -sp], axis=-1),
        np.stack([zero, sp, cp], axis=-1),
    ], axis=-2)  # (1, 3, 3)

    # R_y(yaw)
    ry = np.stack([
        np.stack([cy, zero, sy], axis=-1),
        np.stack([zero, one, zero], axis=-1),
        np.stack([-sy, zero, cy], axis=-1),
    ], axis=-2)

    # R_z(roll)
    rz = np.stack([
        np.stack([cr, -sr, zero], axis=-1),
        np.stack([sr, cr, zero], axis=-1),
        np.stack([zero, zero, one], axis=-1),
    ], axis=-2)

    R = np.matmul(np.matmul(rz, ry), rx).astype(np.float32, copy=False)
    return cast(F32, R)


def transform_keypoints(params: MotionParams) -> ImplicitKeypoints:
    """Compute implicit keypoints from MotionParams.

        kp = scale * (kp_canonical @ R.T + exp) + t_xy0

    Matches upstream `transform_keypoint` in
    KwaiVGI/LivePortrait `src/live_portrait_wrapper.py`.
    """
    params.validate()

    R = rotation_matrix(params.pitch, params.yaw, params.roll)  # (1, 3, 3)

    # kp_canonical @ R.T  -> (1, 21, 3) @ (1, 3, 3).T
    kp_rotated = np.matmul(params.kp_canonical, R.transpose(0, 2, 1))
    kp = kp_rotated + params.exp
    kp = kp * params.scale[:, :, None]  # scale is (1, 1) -> broadcast to (1, 1, 1)

    # Zero out z-translation; upstream convention.
    t = params.t.copy()
    t[:, 2] = 0.0
    kp = kp + t[:, None, :]  # (1, 1, 3) broadcast to (1, 21, 3)

    kp = kp.astype(np.float32, copy=False)

    result = ImplicitKeypoints(data=kp)
    result.validate()
    return result


def apply_stitching_delta(
    kp_driving: ImplicitKeypoints,
    delta: F32,
) -> ImplicitKeypoints:
    """Apply a (1, 63) stitching delta to driving keypoints.

    Upstream stitching_retargeting_module emits a (1, N) prediction
    where the first 63 entries are a delta on driving keypoints
    (and the optional trailing 2 entries are eye/lip retargeting
    scalars, unused in the minimal engine).

    Returns a new ImplicitKeypoints; input is not mutated.
    """
    if delta.ndim != 2 or delta.shape[0] != 1:
        raise ValueError(f"stitching delta must be (1, N>=63), got {delta.shape}")
    if delta.shape[1] < NUM_KEYPOINTS * KEYPOINT_DIMS:
        raise ValueError(
            f"stitching delta must have at least "
            f"{NUM_KEYPOINTS * KEYPOINT_DIMS} entries, got {delta.shape[1]}"
        )
    kp_delta = delta[:, : NUM_KEYPOINTS * KEYPOINT_DIMS].reshape(
        1, NUM_KEYPOINTS, KEYPOINT_DIMS
    ).astype(np.float32, copy=False)
    kp_new = (kp_driving.data + kp_delta).astype(np.float32, copy=False)
    result = ImplicitKeypoints(data=kp_new)
    result.validate()
    return result
