# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Regression: micro-expression realism (§13, §14).

Full-stack test (skipped in skeleton mode — `requires_onnx`).

Operational definition
----------------------
We cannot measure "realism" numerically. We CAN measure that changes
to the driving signal cause corresponding changes in the rendered
eye/mouth regions — i.e. the pipeline is *expressive* rather than
stuck producing a rigid-body response (which was the v0.3.1 failure
mode).

Protocol:
  1. Feed a static driver (= avatar crop) -> record eye & mouth ROI.
  2. Feed a driver with eye/mouth ROIs noised -> record eye & mouth ROI.
  3. Assert the rendered ROIs differ from (1) by more than a
     MICRO_FLOOR — the pipeline is responding.
  4. Assert the NON-ROI regions differ by less than a STATIC_CEILING —
     the pipeline is not globally smearing motion.

ROIs are declared as rectangles in OUTPUT_RES coords. They are
approximate and intentionally broad; this is a FLOOR test ("there
exists motion in eye/mouth"), not a localization test.
"""

from __future__ import annotations

import numpy as np
import pytest

from nsr_engine.config import ASSETS_DIR, EngineConfig
from nsr_engine.engine import NSREngine
from nsr_engine.motion.keypoint_transform import transform_keypoints
from nsr_engine.util.latents import OUTPUT_RES

# ROIs in OUTPUT_RES (512) coords. Wide enough to cover reasonable
# face poses within a centred avatar crop.
EYE_ROI = (
    int(OUTPUT_RES * 0.30), int(OUTPUT_RES * 0.30),
    int(OUTPUT_RES * 0.70), int(OUTPUT_RES * 0.50),
)
MOUTH_ROI = (
    int(OUTPUT_RES * 0.30), int(OUTPUT_RES * 0.60),
    int(OUTPUT_RES * 0.70), int(OUTPUT_RES * 0.85),
)

MICRO_FLOOR = 2.0      # mean |Δ| in uint8 space must exceed this in ROI
STATIC_CEILING = 6.0   # mean |Δ| outside ROI must stay below this


def _roi(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2].astype(np.float32)


def _non_roi_mean_abs(
    img_a: np.ndarray,
    img_b: np.ndarray,
    rois: list[tuple[int, int, int, int]],
) -> float:
    mask = np.ones(img_a.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in rois:
        mask[y1:y2, x1:x2] = False
    a = img_a.astype(np.float32)[mask]
    b = img_b.astype(np.float32)[mask]
    return float(np.mean(np.abs(a - b)))


def _render(engine: NSREngine, driver: np.ndarray) -> np.ndarray:
    params = engine._motion_ext.extract(driver)
    kp_driving = transform_keypoints(params)
    if engine._stitcher is not None:
        kp_driving = engine._stitcher.refine(engine._kp_source, kp_driving)
    warped = engine._warper.warp(
        engine._avatar_feature, engine._kp_source, kp_driving
    )
    return engine._spade.decode(warped)


def _perturb_eye_mouth(crop: np.ndarray) -> np.ndarray:
    """Simulate eye + mouth motion by noising those ROIs in the 256x256 crop."""
    h, w = crop.shape[:2]
    out = crop.copy()
    eye = (int(w * 0.30), int(h * 0.30), int(w * 0.70), int(h * 0.50))
    mouth = (int(w * 0.30), int(h * 0.60), int(w * 0.70), int(h * 0.85))
    rng = np.random.default_rng(42)
    for x1, y1, x2, y2 in (eye, mouth):
        patch = out[y1:y2, x1:x2].astype(np.int16)
        noise = rng.integers(-40, 41, size=patch.shape, dtype=np.int16)
        out[y1:y2, x1:x2] = np.clip(patch + noise, 0, 255).astype(np.uint8)
    return out


@pytest.mark.requires_onnx
def test_micro_expression_roi_responds_to_driver() -> None:
    cfg = EngineConfig(headless=True, avatar_path=ASSETS_DIR / "avatar.png")
    engine = NSREngine(cfg)

    static_driver = engine._avatar_crop.copy()
    perturbed_driver = _perturb_eye_mouth(engine._avatar_crop)

    # Warm up (if any stage has internal state).
    _render(engine, static_driver)

    out_static = _render(engine, static_driver)
    out_perturb = _render(engine, perturbed_driver)

    eye_delta = float(np.mean(np.abs(
        _roi(out_perturb, EYE_ROI) - _roi(out_static, EYE_ROI)
    )))
    mouth_delta = float(np.mean(np.abs(
        _roi(out_perturb, MOUTH_ROI) - _roi(out_static, MOUTH_ROI)
    )))
    non_roi_delta = _non_roi_mean_abs(
        out_perturb, out_static, [EYE_ROI, MOUTH_ROI]
    )

    assert eye_delta > MICRO_FLOOR, (
        f"eye ROI did not respond: mean |Δ| = {eye_delta:.3f}, "
        f"floor = {MICRO_FLOOR:.3f}"
    )
    assert mouth_delta > MICRO_FLOOR, (
        f"mouth ROI did not respond: mean |Δ| = {mouth_delta:.3f}, "
        f"floor = {MICRO_FLOOR:.3f}"
    )
    assert non_roi_delta < STATIC_CEILING, (
        f"non-ROI regions changed too much: mean |Δ| = {non_roi_delta:.3f}, "
        f"ceiling = {STATIC_CEILING:.3f}. Pipeline may be globally smearing motion."
    )
