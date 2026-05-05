# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Stitching retargeting — identity-preserving kp refinement (§4.5).

Wraps `stitching_retargeting.onnx` (LivePortrait). A small MLP that
takes the concatenation of source and driving implicit keypoints
(flattened) and produces a delta to apply to the driving keypoints.
Purpose: keep identity stable under large driving motion by nudging
kp_driving back toward the source identity manifold.

ONNX I/O contract (LivePortrait reference)
------------------------------------------
  INPUT:   (1, 126) float32 — concat(kp_source.flat, kp_driving.flat)
           (the two 63-vectors stacked along dim 1)
  OUTPUT:  (1, 65) float32
           The first 63 entries are the delta on driving keypoints.
           The trailing 2 entries are eye/lip retargeting scalars
           (unused here).

Some exports of this module bundle the three upstream MLPs (stitching,
eye retargeting, lip retargeting) into separate graphs. This loader
targets the single `stitching` variant — the one referenced by §11's
`stitching_retargeting.onnx` filename. If the export instead provides
a multi-graph bundle, the init-time arity check fails loudly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from nsr_engine.face.onnx_util import (
    assert_single_input_output,
    describe_io,
    make_session,
    shape_compatible,
)
from nsr_engine.motion.keypoint_transform import apply_stitching_delta
from nsr_engine.util.latents import (
    KEYPOINT_FLAT,
    ImplicitKeypoints,
)
from nsr_engine.util.onnx_compat import ort

logger = logging.getLogger("nsr.render.stitching")

# Input is 2 * 63 = 126. Output is exactly 65 (63 kp delta + 2 retargeting scalars).
_EXPECTED_INPUT_SHAPE = (1, 2 * KEYPOINT_FLAT)
_EXPECTED_OUTPUT_WIDTH = 65


class StitchingRetargeting:
    """Loads stitching_retargeting.onnx and refines kp_driving for identity."""

    def __init__(
        self,
        model_path: Path,
        intra_threads: int,
        inter_threads: int,
    ) -> None:
        self._session: ort.InferenceSession = make_session(
            model_path, intra_threads, inter_threads
        )
        assert_single_input_output(self._session, "stitching_retargeting")

        inp = self._session.get_inputs()[0]
        out = self._session.get_outputs()[0]

        if not shape_compatible(inp.shape, _EXPECTED_INPUT_SHAPE):
            raise RuntimeError(
                f"stitching_retargeting input shape {tuple(inp.shape)} "
                f"incompatible with required {_EXPECTED_INPUT_SHAPE}. "
                f"Expected concat(kp_source.flat, kp_driving.flat) -> (1, 126)."
            )

        out_shape = tuple(out.shape)
        # Batch dim may be a positive int 1 or a symbolic string (e.g. 'batch_size').
        batch_ok = out_shape[0] == 1 or isinstance(out_shape[0], str)
        if len(out_shape) != 2 or not batch_ok:
            raise RuntimeError(
                f"stitching_retargeting output must be 2-D (1, N), got {out_shape}"
            )
        width = out_shape[1]
        if width != _EXPECTED_OUTPUT_WIDTH:
            raise RuntimeError(
                f"stitching_retargeting output width {width} != "
                f"{_EXPECTED_OUTPUT_WIDTH} (63 kp delta + 2 retargeting scalars)"
            )

        self._input_name: str = inp.name
        self._output_name: str = out.name

        logger.info(
            "StitchingRetargeting loaded: %s (%s)",
            model_path.name, describe_io(self._session),
        )

    def refine(
        self,
        kp_source: ImplicitKeypoints,
        kp_driving: ImplicitKeypoints,
    ) -> ImplicitKeypoints:
        """Return a new ImplicitKeypoints equal to kp_driving + delta.

        kp_source is not mutated. kp_driving is not mutated.
        """
        kp_source.validate()
        kp_driving.validate()

        # Flatten to (1, 63) each and concatenate on feature axis.
        src_flat = kp_source.data.reshape(1, KEYPOINT_FLAT).astype(
            np.float32, copy=False
        )
        drv_flat = kp_driving.data.reshape(1, KEYPOINT_FLAT).astype(
            np.float32, copy=False
        )
        blob = np.concatenate([src_flat, drv_flat], axis=1).astype(
            np.float32, copy=False
        )
        if blob.shape != _EXPECTED_INPUT_SHAPE:
            raise RuntimeError(
                f"stitching_retargeting input shape {blob.shape} != "
                f"expected {_EXPECTED_INPUT_SHAPE}"
            )

        raw = self._session.run([self._output_name], {self._input_name: blob})[0]
        if raw.ndim != 2 or raw.shape[0] != 1 or raw.shape[1] != _EXPECTED_OUTPUT_WIDTH:
            raise RuntimeError(
                f"stitching_retargeting runtime output shape {raw.shape} invalid; "
                f"expected (1, {_EXPECTED_OUTPUT_WIDTH})"
            )
        if raw.dtype != np.float32:
            raw = raw.astype(np.float32, copy=False)

        return apply_stitching_delta(kp_driving, raw)
