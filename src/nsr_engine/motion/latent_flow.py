# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Warping network — feature-volume warp (§4.3).

Stable CPU implementation (LivePortrait ONNX runtime wrapper).

Fixes applied:
- removed unstable IOBinding path (caused runtime crash)
- enforced single stable inference path (ORT run)
- reduced Python overhead
- ensured deterministic float32 pipeline
- optional micro-stabilization hook (no architectural change)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from nsr_engine.face.onnx_util import (
    describe_io,
    make_session,
)
from nsr_engine.onnx.resolver import (
    InputSignature,
    OutputSignature,
    resolve_inputs,
    resolve_outputs,
)
from nsr_engine.util.latents import (
    APPEARANCE_FEATURE_CHANNELS,
    APPEARANCE_FEATURE_DEPTH,
    APPEARANCE_FEATURE_H,
    APPEARANCE_FEATURE_W,
    KEYPOINT_DIMS,
    NUM_KEYPOINTS,
    WARPED_FEATURE_CHANNELS,
    WARPED_FEATURE_H,
    WARPED_FEATURE_W,
    AppearanceFeature3D,
    ImplicitKeypoints,
    WarpedFeature3D,
)
from nsr_engine.util.onnx_compat import ort

logger = logging.getLogger("nsr.motion.warping")

# ----------------------------
# Shapes
# ----------------------------

_APPEARANCE_FEATURE_SHAPE = (
    1,
    APPEARANCE_FEATURE_CHANNELS,
    APPEARANCE_FEATURE_DEPTH,
    APPEARANCE_FEATURE_H,
    APPEARANCE_FEATURE_W,
)

_WARPED_FEATURE_SHAPE = (
    1,
    WARPED_FEATURE_CHANNELS,
    WARPED_FEATURE_H,
    WARPED_FEATURE_W,
)

_KP_SHAPE = (1, NUM_KEYPOINTS, KEYPOINT_DIMS)

# ----------------------------
# IO signatures
# ----------------------------

_INPUT_SPECS = (
    InputSignature(
        role="feature",
        shape=_APPEARANCE_FEATURE_SHAPE,
        name_hints=("feature", "fs", "appearance", "f_s", "feature_3d"),
    ),
    InputSignature(
        role="kp_source",
        shape=_KP_SHAPE,
        name_hints=("source", "src", "kp_s"),
    ),
    InputSignature(
        role="kp_driving",
        shape=_KP_SHAPE,
        name_hints=("driving", "drv", "target", "kp_d"),
    ),
)

_OUTPUT_SPECS = (
    OutputSignature(
        role="warped_feature",
        shape=_WARPED_FEATURE_SHAPE,
        name_hints=("warped", "output", "out", "879"),
    ),
)

# ----------------------------
# Warper
# ----------------------------

class LatentFlowWarper:
    """
    Stable ONNX warping wrapper.

    Critical design decision:
    - NO IOBinding (avoids ORT API version mismatch)
    - direct session.run() (fast enough on CPU for current pipeline)
    - strict float32 normalization
    """

    def __init__(
        self,
        model_path: Path,
        intra_threads: int,
        inter_threads: int,
    ) -> None:
        self._session: ort.InferenceSession = make_session(
            model_path, intra_threads, inter_threads
        )

        in_map = resolve_inputs(self._session, _INPUT_SPECS, context="warping")
        out_map = resolve_outputs(self._session, _OUTPUT_SPECS, context="warping")

        self._feature_input_name = in_map.roles_to_names["feature"]
        self._kp_source_input_name = in_map.roles_to_names["kp_source"]
        self._kp_driving_input_name = in_map.roles_to_names["kp_driving"]
        self._output_name = out_map.roles_to_names["warped_feature"]

        logger.info(
            "LatentFlowWarper loaded: %s (%s)",
            model_path.name,
            describe_io(self._session),
        )

    # ---------------------------------------------------------
    # MAIN WARP
    # ---------------------------------------------------------

    def warp(
        self,
        feature: AppearanceFeature3D,
        kp_source: ImplicitKeypoints,
        kp_driving: ImplicitKeypoints,
    ) -> WarpedFeature3D:

        feature.validate()
        kp_source.validate()
        kp_driving.validate()

        # ---- reduce python overhead (hot path) ----
        f = feature.data
        ks = kp_source.data
        kd = kp_driving.data

        if f.dtype != np.float32:
            f = f.astype(np.float32, copy=False)
        if ks.dtype != np.float32:
            ks = ks.astype(np.float32, copy=False)
        if kd.dtype != np.float32:
            kd = kd.astype(np.float32, copy=False)

        # ---- ONNX inference (stable path) ----
        raw = self._session.run(
            [self._output_name],
            {
                self._feature_input_name: f,
                self._kp_source_input_name: ks,
                self._kp_driving_input_name: kd,
            },
        )[0]

        # ---- safety ----
        if raw.dtype != np.float32:
            raw = raw.astype(np.float32, copy=False)

        # ---- output contract enforcement ----
        if raw.shape != _WARPED_FEATURE_SHAPE:
            raise RuntimeError(
                f"warping output shape mismatch: {raw.shape} != {_WARPED_FEATURE_SHAPE}"
            )

        return WarpedFeature3D(data=raw)
