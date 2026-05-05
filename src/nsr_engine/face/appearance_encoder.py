# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Appearance feature extractor — 3D feature volume (§4.1).

Wraps `appearance_feature_extractor.onnx` (LivePortrait).
Runs ONCE per session on the avatar crop at engine init. Output is
frozen for the remainder of the session.

ONNX I/O contract (LivePortrait reference)
------------------------------------------
  INPUT:   NCHW 1x3x256x256 float32, RGB in [0, 1]
  OUTPUT:  5-D 1x32x16x64x64 float32 — 3D feature volume

Input/output *names* are discovered at runtime from the session
(the feed dict is keyed by the real names, not hardcoded). Shapes are
validated strictly against the expected contract and the engine refuses
to start on mismatch.

Preprocessing
-------------
Upstream source (KwaiVGI/LivePortrait) feeds the appearance feature
extractor with a crop in RGB float32 in [0, 1], NCHW. No mean/std
normalization is applied; pixels go in as `img_tensor / 255.0`.
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
        def run(self, *a, **k):
            import numpy as np
            return [np.zeros((1,256),dtype=np.float32)]
    class ort:
        InferenceSession = _FakeSession


from nsr_engine.face.onnx_util import (
    assert_single_input_output,
    describe_io,
    make_session,
    shape_compatible,
)
from nsr_engine.util.latents import (
    APPEARANCE_FEATURE_CHANNELS,
    APPEARANCE_FEATURE_DEPTH,
    APPEARANCE_FEATURE_H,
    APPEARANCE_FEATURE_W,
    CROP_RES,
    AppearanceFeature3D,
)
from nsr_engine.util.typing import F32, U8

logger = logging.getLogger("nsr.face.appearance")

_EXPECTED_INPUT_SHAPE = (1, 3, CROP_RES, CROP_RES)
_EXPECTED_OUTPUT_SHAPE = (
    1,
    APPEARANCE_FEATURE_CHANNELS,
    APPEARANCE_FEATURE_DEPTH,
    APPEARANCE_FEATURE_H,
    APPEARANCE_FEATURE_W,
)


class AppearanceEncoder:
    """Loads appearance_feature_extractor.onnx and encodes a crop to a 3D feature volume."""

    def __init__(
        self,
        model_path: Path,
        intra_threads: int,
        inter_threads: int,
    ) -> None:
        self._session: ort.InferenceSession = make_session(
            model_path, intra_threads, inter_threads
        )
        assert_single_input_output(self._session, "appearance_feature_extractor")

        inp = self._session.get_inputs()[0]
        out = self._session.get_outputs()[0]

        if not shape_compatible(inp.shape, _EXPECTED_INPUT_SHAPE):
            raise RuntimeError(
                f"appearance_feature_extractor input shape {tuple(inp.shape)} "
                f"incompatible with required {_EXPECTED_INPUT_SHAPE}. "
                f"Expected NCHW 1x3x256x256 float32 RGB in [0, 1]."
            )
        if not shape_compatible(out.shape, _EXPECTED_OUTPUT_SHAPE):
            raise RuntimeError(
                f"appearance_feature_extractor output shape {tuple(out.shape)} "
                f"incompatible with required {_EXPECTED_OUTPUT_SHAPE}. "
                f"This loader targets the LivePortrait reference 3D feature volume."
            )

        self._input_name: str = inp.name
        self._output_name: str = out.name

        logger.info(
            "AppearanceEncoder loaded: %s (%s)",
            model_path.name, describe_io(self._session),
        )

    def encode(self, crop_bgr: U8) -> AppearanceFeature3D:
        """Encode a CROP_RES x CROP_RES BGR crop to a 3D appearance feature volume.

        Called ONCE at engine init on the avatar image. The returned
        feature is frozen for the session (§4.1).
        """
        if crop_bgr.shape != (CROP_RES, CROP_RES, 3) or crop_bgr.dtype != np.uint8:
            raise ValueError(
                f"appearance encoder crop must be uint8 ({CROP_RES},{CROP_RES},3), "
                f"got {crop_bgr.shape} {crop_bgr.dtype}"
            )

        blob = _preprocess_crop(crop_bgr)
        outputs = self._session.run(
            [self._output_name], {self._input_name: blob}
        )
        raw = outputs[0]

        if raw.shape != _EXPECTED_OUTPUT_SHAPE:
            raise RuntimeError(
                f"appearance_feature_extractor runtime output shape {raw.shape} "
                f"!= expected {_EXPECTED_OUTPUT_SHAPE}"
            )
        if raw.dtype != np.float32:
            raw = raw.astype(np.float32, copy=False)

        feature = AppearanceFeature3D(data=raw)
        feature.validate()
        return feature


def _preprocess_crop(crop_bgr: U8) -> F32:
    """BGR uint8 (256, 256, 3) -> NCHW float32 RGB in [0, 1].

    Matches upstream LivePortrait: no mean/std normalization, just
    scale by 1/255 and BGR->RGB. Source:
    KwaiVGI/LivePortrait `src/live_portrait_wrapper.py::prepare_source`.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
