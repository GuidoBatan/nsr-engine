# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Pytest configuration for NSR v0.1.0 (LivePortrait topology).

Two execution modes:

  - SKELETON mode (default): one or more of the five LivePortrait ONNX
    files are missing from models/face/. Tests marked `requires_onnx`
    are skipped. Tests that verify data contracts, module wiring, shape
    assertions, and the async slot run unconditionally.

  - INTEGRATED mode: all required ONNX files are present. The
    `requires_onnx` marker activates.

Detection is conservative: we only switch modes when every expected
file exists. Partial presence fails closed (skeleton mode).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Put src/ on sys.path so `import nsr_engine` works without install.
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_MODELS = _REPO / "models" / "face"
_REQUIRED = (
    "appearance_feature_extractor.onnx",
    "motion_extractor.onnx",
    "warping.onnx",
    "spade_generator.onnx",
    "stitching_retargeting.onnx",
)


def _all_models_present() -> bool:
    return all((_MODELS / name).exists() for name in _REQUIRED)


ONNX_AVAILABLE = _all_models_present()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_onnx: test needs all v0.1.0 LivePortrait ONNX files "
        "(skipped in skeleton mode)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if ONNX_AVAILABLE:
        return
    skip_onnx = pytest.mark.skip(
        reason="SKELETON MODE: LivePortrait ONNX models not present in models/face/"
    )
    for item in items:
        if "requires_onnx" in item.keywords:
            item.add_marker(skip_onnx)
