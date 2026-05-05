# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Regression: the five ONNX wrappers must contain real inference, not stubs.

In the skeleton (v0.1.0 as shipped), every ONNX loader had a
`_run_inference` method that raised `NotImplementedError`. The
LivePortrait rewrite replaces all of them with real `session.run`
calls. If anyone reintroduces a stub, this test catches it.

Runs without any ONNX model files.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import nsr_engine.face.appearance_encoder as _appearance_encoder
import nsr_engine.face.motion_extractor as _motion_extractor
import nsr_engine.motion.latent_flow as _latent_flow
import nsr_engine.render.neural_stitcher as _neural_stitcher
import nsr_engine.render.spade_generator as _spade_generator

_MODULES = (
    _appearance_encoder,
    _motion_extractor,
    _latent_flow,
    _neural_stitcher,
    _spade_generator,
)


class TestNoInferenceStubs:
    def test_no_not_implemented_error_raised_in_source(self) -> None:
        """Scan source text of each ONNX wrapper for `raise NotImplementedError`.

        The skeleton flagged unfilled stubs this way; real inference paths
        must not reintroduce the pattern.
        """
        for mod in _MODULES:
            path = Path(inspect.getsourcefile(mod))
            src = path.read_text(encoding="utf-8")
            assert "raise NotImplementedError" not in src, (
                f"{path.name} still contains a NotImplementedError stub. "
                f"The ONNX inference path must be real, not stubbed."
            )

    def test_each_wrapper_class_exists(self) -> None:
        """Confirm the five wrapper class names are exported."""
        assert hasattr(_appearance_encoder, "AppearanceEncoder")
        assert hasattr(_motion_extractor, "MotionExtractor")
        assert hasattr(_latent_flow, "LatentFlowWarper")
        assert hasattr(_spade_generator, "SpadeGenerator")
        assert hasattr(_neural_stitcher, "StitchingRetargeting")
