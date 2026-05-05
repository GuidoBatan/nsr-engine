# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""SPADE generator — neural decoder (§4.4).

Wraps `spade_generator.onnx` (LivePortrait). Consumes a warped 3D
feature volume and produces an RGB image at OUTPUT_RES.

ONNX I/O contract (LivePortrait reference)
------------------------------------------
  INPUT:   (1, 32, 16, 64, 64) float32 — warped 3D feature volume
  OUTPUT:  (1, 3, 512, 512)    float32 — RGB in [0, 1]

The generator upscales natively. No cv2.resize in the default path.
Output range is [0, 1]; postprocess clips and converts to uint8.
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
from nsr_engine.util.latents import (
    OUTPUT_RES,
    WARPED_FEATURE_CHANNELS,
    WARPED_FEATURE_H,
    WARPED_FEATURE_W,
    WarpedFeature3D,
)
from nsr_engine.util.onnx_compat import ort
from nsr_engine.util.typing import F32, U8

logger = logging.getLogger("nsr.render.spade")

_EXPECTED_INPUT_SHAPE = (
    1,
    WARPED_FEATURE_CHANNELS,
    WARPED_FEATURE_H,
    WARPED_FEATURE_W,
)
_EXPECTED_OUTPUT_SHAPE = (1, 3, OUTPUT_RES, OUTPUT_RES)


class SpadeGenerator:
    """Loads spade_generator.onnx and decodes warped features to RGB."""

    def __init__(
        self,
        model_path: Path,
        intra_threads: int,
        inter_threads: int,
    ) -> None:
        self._session: ort.InferenceSession = make_session(
            model_path, intra_threads, inter_threads
        )
        assert_single_input_output(self._session, "spade_generator")

        inp = self._session.get_inputs()[0]
        out = self._session.get_outputs()[0]

        if not shape_compatible(inp.shape, _EXPECTED_INPUT_SHAPE):
            raise RuntimeError(
                f"spade_generator input shape {tuple(inp.shape)} incompatible "
                f"with required {_EXPECTED_INPUT_SHAPE}"
            )
        if not shape_compatible(out.shape, _EXPECTED_OUTPUT_SHAPE):
            raise RuntimeError(
                f"spade_generator output shape {tuple(out.shape)} incompatible "
                f"with required {_EXPECTED_OUTPUT_SHAPE}"
            )

        self._input_name: str = inp.name
        self._output_name: str = out.name

        logger.info(
            "SpadeGenerator loaded: %s (%s)",
            model_path.name, describe_io(self._session),
        )

    def decode(self, warped: WarpedFeature3D) -> U8:
        """Decode a warped 3D feature volume to HWC uint8 RGB at OUTPUT_RES."""
        warped.validate()

        raw = self._session.run(
            [self._output_name], {self._input_name: warped.data}
        )[0]

        if raw.shape != _EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError(
                f"spade_generator runtime output shape {raw.shape} != expected "
                f"{_EXPECTED_OUTPUT_SHAPE}"
            )

        return _postprocess(raw)


def _postprocess(raw: F32) -> U8:
    """NCHW float32 RGB in [0, 1] -> HWC uint8 RGB."""
    if raw.dtype != np.float32:
        raw = raw.astype(np.float32, copy=False)
    img = np.transpose(raw[0], (1, 2, 0))  # CHW -> HWC
    img = np.clip(img * 255.0, 0.0, 255.0)
    return img.astype(np.uint8)
