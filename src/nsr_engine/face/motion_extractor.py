# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Motion extractor — structured per-frame motion (§4.2).

Wraps `motion_extractor.onnx` (LivePortrait). Consumes a CROP_RES BGR
crop of the driving webcam frame and produces a `MotionParams` with
pitch/yaw/roll (degrees), translation, expression, scale, and canonical
implicit keypoints.

Output resolution
-----------------
Seven output tensors, resolved at init time by the shared
`nsr_engine.onnx.resolver`:

    pitch:        (1, 66)  -- softmax bins over pitch
    yaw:          (1, 66)  -- softmax bins over yaw
    roll:         (1, 66)  -- softmax bins over roll
    t:            (1, 3)   -- translation
    exp:          (1, 63)  -- 21x3 expression delta (flat)
    scale:        (1, 1)   -- global scale
    kp:           (1, 63)  -- 21x3 canonical keypoints (flat)

The resolver handles the fact that three outputs share shape (1, 66)
and two share (1, 63) — it disambiguates by name hints first, then
falls back to declared-source order within each shape-group.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    class _FakeSession:
        def __init__(self,*a,**k): pass
        def run(self,*a,**k):
            import numpy as np
            return [np.zeros((1,256),dtype=np.float32)]
    class ort:
        InferenceSession=_FakeSession


from nsr_engine.face.onnx_util import (
    describe_io,
    make_session,
    shape_compatible,
)
from nsr_engine.motion.keypoint_transform import decode_pose_bins
from nsr_engine.onnx.resolver import (
    InputSignature,
    OutputSignature,
    resolve_inputs,
    resolve_outputs,
)
from nsr_engine.util.latents import (
    CROP_RES,
    KEYPOINT_DIMS,
    KEYPOINT_FLAT,
    NUM_KEYPOINTS,
    POSE_BINS,
    MotionParams,
)
from nsr_engine.util.typing import F32, U8

logger = logging.getLogger("nsr.face.motion")

_EXPECTED_INPUT_SHAPE = (1, 3, CROP_RES, CROP_RES)

_SHAPE_POSE = (1, POSE_BINS)
_SHAPE_T = (1, 3)
_SHAPE_SCALE = (1, 1)
_SHAPE_EXP_OR_KP = (1, KEYPOINT_FLAT)

_INPUT_SPECS = (
    InputSignature(role="crop", shape=_EXPECTED_INPUT_SHAPE),
)

_OUTPUT_SPECS = (
    OutputSignature(role="pitch", shape=_SHAPE_POSE, name_hints=("pitch",)),
    OutputSignature(role="yaw",   shape=_SHAPE_POSE, name_hints=("yaw",)),
    OutputSignature(role="roll",  shape=_SHAPE_POSE, name_hints=("roll",)),
    OutputSignature(
        role="t", shape=_SHAPE_T,
        name_hints=("t", "translation", "trans"),
    ),
    OutputSignature(
        role="exp", shape=_SHAPE_EXP_OR_KP,
        name_hints=("exp", "expression", "delta"),
    ),
    OutputSignature(role="scale", shape=_SHAPE_SCALE, name_hints=("scale",)),
    OutputSignature(
        role="kp", shape=_SHAPE_EXP_OR_KP,
        name_hints=("kp", "keypoint", "canonical"),
    ),
)


class MotionExtractor:
    """Loads motion_extractor.onnx and extracts per-frame MotionParams."""

    def __init__(
        self,
        model_path: Path,
        intra_threads: int,
        inter_threads: int,
    ) -> None:
        self._session: ort.InferenceSession = make_session(
            model_path, intra_threads, inter_threads
        )

        in_map = resolve_inputs(self._session, _INPUT_SPECS, context="motion_extractor")
        out_map = resolve_outputs(self._session, _OUTPUT_SPECS, context="motion_extractor")

        self._input_name: str = in_map.roles_to_names["crop"]

        # role -> output tensor name, used at every extract() call.
        self._role_to_output_name: dict[str, str] = dict(out_map.roles_to_names)
        # Canonical order of output names for session.run (arbitrary — we index
        # by name anyway, but session.run wants a deterministic list).
        self._role_order: tuple[str, ...] = tuple(s.role for s in _OUTPUT_SPECS)
        self._all_output_names: list[str] = [
            self._role_to_output_name[r] for r in self._role_order
        ]

        logger.info(
            "MotionExtractor loaded: %s (%s); role map: %s",
            model_path.name,
            describe_io(self._session),
            {r: self._role_to_output_name[r] for r in self._role_order},
        )

    def extract(self, crop_bgr: U8) -> MotionParams:
        """Extract MotionParams from one driving crop."""
        if crop_bgr.shape != (CROP_RES, CROP_RES, 3) or crop_bgr.dtype != np.uint8:
            raise ValueError(
                f"motion extractor crop must be uint8 ({CROP_RES},{CROP_RES},3), "
                f"got {crop_bgr.shape} {crop_bgr.dtype}"
            )

        blob = _preprocess_crop(crop_bgr)
        raw_outputs = self._session.run(
            self._all_output_names, {self._input_name: blob}
        )
        by_role: dict[str, F32] = {}
        for role, arr in zip(self._role_order, raw_outputs, strict=True):
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            by_role[role] = arr

        pitch_logits = by_role["pitch"]
        yaw_logits = by_role["yaw"]
        roll_logits = by_role["roll"]
        t = by_role["t"]
        exp_flat = by_role["exp"]
        scale = by_role["scale"]
        kp_flat = by_role["kp"]

        # Runtime shape sanity (in case dynamic axes in the graph fooled
        # the declared-shape check the resolver did at init time).
        _check_shape("pitch_logits", pitch_logits, _SHAPE_POSE)
        _check_shape("yaw_logits", yaw_logits, _SHAPE_POSE)
        _check_shape("roll_logits", roll_logits, _SHAPE_POSE)
        _check_shape("t", t, _SHAPE_T)
        _check_shape("exp_flat", exp_flat, _SHAPE_EXP_OR_KP)
        _check_shape("scale", scale, _SHAPE_SCALE)
        _check_shape("kp_flat", kp_flat, _SHAPE_EXP_OR_KP)

        pitch = decode_pose_bins(pitch_logits)
        yaw = decode_pose_bins(yaw_logits)
        roll = decode_pose_bins(roll_logits)
        exp = exp_flat.reshape(1, NUM_KEYPOINTS, KEYPOINT_DIMS)
        kp_canonical = kp_flat.reshape(1, NUM_KEYPOINTS, KEYPOINT_DIMS)

        params = MotionParams(
            pitch=pitch.astype(np.float32, copy=False),
            yaw=yaw.astype(np.float32, copy=False),
            roll=roll.astype(np.float32, copy=False),
            t=t.astype(np.float32, copy=False),
            exp=exp.astype(np.float32, copy=False),
            scale=scale.astype(np.float32, copy=False),
            kp_canonical=kp_canonical.astype(np.float32, copy=False),
        )
        params.validate()
        return params


def _check_shape(name: str, arr: np.ndarray, expected: tuple) -> None:
    if not shape_compatible(arr.shape, expected):
        raise RuntimeError(
            f"motion_extractor runtime shape mismatch for {name}: "
            f"expected {expected}, got {arr.shape}"
        )


def _preprocess_crop(crop_bgr: U8) -> F32:
    """BGR uint8 (256, 256, 3) -> NCHW float32 RGB in [0, 1].

    Same preprocessing as AppearanceEncoder; upstream applies identical
    normalization to source and driving crops.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
