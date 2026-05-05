# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Regression: temporal stability — "no jitter or temporal tearing" (§14).

Full-stack test (skipped in skeleton mode — `requires_onnx`).

Two sub-protocols:

  (a) STATIC DRIVER: feed the same driver crop N times. Pairwise
      frame L2 distance must be below a small threshold. Any drift
      indicates non-determinism in ORT (check session options) or
      stateful contamination in the motion extractor (it should be
      stateless).

  (b) SLOW DRIFT: interpolate linearly between two driver crops over
      N frames. The first discrete difference of rendered frames
      should itself vary smoothly — we measure it via the standard
      deviation of the per-frame diff norm, normalized by the mean.
      A well-behaved pipeline produces a near-constant diff (CV < 0.5).
      A tearing pipeline produces sudden spikes (high CV).
"""

from __future__ import annotations

import numpy as np
import pytest

from nsr_engine.config import ASSETS_DIR, EngineConfig
from nsr_engine.engine import NSREngine
from nsr_engine.motion.keypoint_transform import transform_keypoints

STATIC_L2_THRESHOLD = 1.5   # mean per-pixel |Δ| in [0, 255] space
DRIFT_CV_THRESHOLD = 0.5


def _render_one(engine: NSREngine, driver: np.ndarray) -> np.ndarray:
    params = engine._motion_ext.extract(driver)
    kp_driving = transform_keypoints(params)
    if engine._stitcher is not None:
        kp_driving = engine._stitcher.refine(engine._kp_source, kp_driving)
    warped = engine._warper.warp(
        engine._avatar_feature, engine._kp_source, kp_driving
    )
    return engine._spade.decode(warped).astype(np.float32)


@pytest.mark.requires_onnx
def test_static_driver_is_stable() -> None:
    cfg = EngineConfig(headless=True, avatar_path=ASSETS_DIR / "avatar.png")
    engine = NSREngine(cfg)

    driver = engine._avatar_crop.copy()
    first = _render_one(engine, driver)
    diffs: list[float] = []
    # 20 iterations: CPU SPADE is ~200+ ms/frame; keep test tractable.
    for _ in range(20):
        cur = _render_one(engine, driver)
        d = float(np.mean(np.abs(cur - first)))
        diffs.append(d)
    mean_diff = float(np.mean(diffs))
    assert mean_diff < STATIC_L2_THRESHOLD, (
        f"static-driver instability: mean per-pixel |Δ| = {mean_diff:.3f} "
        f">= {STATIC_L2_THRESHOLD:.3f}"
    )


@pytest.mark.requires_onnx
def test_slow_drift_has_smooth_temporal_gradient() -> None:
    cfg = EngineConfig(headless=True, avatar_path=ASSETS_DIR / "avatar.png")
    engine = NSREngine(cfg)

    rng = np.random.default_rng(1)
    start = engine._avatar_crop.astype(np.float32)
    end = np.clip(
        start + rng.standard_normal(start.shape).astype(np.float32) * 3.0,
        0.0, 255.0,
    )

    frames: list[np.ndarray] = []
    N = 15
    for i in range(N):
        alpha = i / (N - 1)
        driver = ((1.0 - alpha) * start + alpha * end).astype(np.uint8)
        frames.append(_render_one(engine, driver))

    diffs = np.array([
        float(np.mean(np.abs(frames[i + 1] - frames[i])))
        for i in range(N - 1)
    ])
    mean_d = float(diffs.mean())
    std_d = float(diffs.std())
    cv = std_d / (mean_d + 1e-6)
    assert cv < DRIFT_CV_THRESHOLD, (
        f"frame-to-frame diff is non-smooth (tearing likely): "
        f"CV={cv:.3f} (mean={mean_d:.3f}, std={std_d:.3f})"
    )
