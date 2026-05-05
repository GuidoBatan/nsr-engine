# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Regression: identity preservation (§13, §14).

Full-stack test (skipped in skeleton mode — `requires_onnx`).

Protocol (LivePortrait pipeline):
  1. Construct NSREngine; capture the frozen avatar appearance feature
     and source keypoints.
  2. Over N driving frames (avatar with small gaussian noise), run
     motion -> kp_driving -> (stitching) -> warp -> decode.
  3. For each rendered frame, re-encode it to a 3D appearance feature
     (via the appearance encoder) and measure cosine similarity of the
     FLATTENED feature volume against the reference.
  4. Assert mean similarity stays above IDENTITY_THRESHOLD and that the
     minimum never drops more than 0.05 below the mean.

The threshold is conservative (0.85). It is a TESTABLE CONTRACT: if a
new ONNX export fails this test, the export is rejected — the threshold
is not moved to accommodate it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from nsr_engine.config import ASSETS_DIR, EngineConfig
from nsr_engine.engine import NSREngine
from nsr_engine.motion.keypoint_transform import transform_keypoints
from nsr_engine.util.latents import CROP_RES

IDENTITY_THRESHOLD = 0.85
N_FRAMES = 50  # Reduced from 200: CPU SPADE is ~200+ ms/frame, keep test tractable.


def _render_one(engine: NSREngine, driver: np.ndarray) -> np.ndarray:
    params = engine._motion_ext.extract(driver)
    kp_driving = transform_keypoints(params)
    if engine._stitcher is not None:
        kp_driving = engine._stitcher.refine(engine._kp_source, kp_driving)
    warped = engine._warper.warp(
        engine._avatar_feature, engine._kp_source, kp_driving
    )
    return engine._spade.decode(warped)


@pytest.mark.requires_onnx
def test_identity_preservation() -> None:
    cfg = EngineConfig(headless=True, avatar_path=ASSETS_DIR / "avatar.png")
    engine = NSREngine(cfg)

    ref = engine._avatar_feature.data.reshape(-1).astype(np.float64)
    ref_n = ref / (np.linalg.norm(ref) + 1e-12)

    rng = np.random.default_rng(0)
    sims: list[float] = []

    for _ in range(N_FRAMES):
        noise = rng.integers(-5, 6, size=engine._avatar_crop.shape, dtype=np.int16)
        driver = np.clip(
            engine._avatar_crop.astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)

        out_rgb = _render_one(engine, driver)  # uint8 (OUTPUT_RES, OUTPUT_RES, 3)

        # Re-encode the output: downsample to CROP_RES, convert RGB->BGR for encoder.
        crop_rgb = cv2.resize(
            out_rgb, (CROP_RES, CROP_RES), interpolation=cv2.INTER_LINEAR
        )
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        feat = engine._appearance_enc.encode(crop_bgr).data.reshape(-1).astype(
            np.float64
        )
        feat_n = feat / (np.linalg.norm(feat) + 1e-12)
        sims.append(float(np.dot(ref_n, feat_n)))

    sims_arr = np.array(sims, dtype=np.float64)
    mean_sim = float(sims_arr.mean())
    min_sim = float(sims_arr.min())

    assert mean_sim >= IDENTITY_THRESHOLD, (
        f"identity drift: mean cosine sim {mean_sim:.3f} < "
        f"threshold {IDENTITY_THRESHOLD:.3f}"
    )
    assert (mean_sim - min_sim) <= 0.05, (
        f"identity instability: min sim {min_sim:.3f} is "
        f"{mean_sim - min_sim:.3f} below the mean {mean_sim:.3f}"
    )
