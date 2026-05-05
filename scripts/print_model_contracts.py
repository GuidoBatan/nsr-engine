#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Model acquisition instructions for NSR v1.0.0 (LivePortrait ONNX).

This script does NOT download anything. Per project policy, the NSR
engine must not fetch anything at runtime. This file exists to document
exactly what must be placed in `models/face/` before the engine will
start.

Face detection: MediaPipe (bundled via pip install mediapipe).
No external model file needed for face detection.

Required files (v1.0.0)
=======================

    models/face/appearance_feature_extractor.onnx  # §4.1 3D appearance feature
    models/face/motion_extractor.onnx              # §4.2 structured motion
    models/face/warping.onnx                       # §4.3 feature-volume warp
    models/face/spade_generator.onnx               # §4.4 SPADE decoder
    models/face/stitching_retargeting.onnx         # §4.5 identity stitching MLP

File source (§11)
=================

Download the LivePortrait ONNX exports from:

    https://huggingface.co/myn0908/Live-Portrait-ONNX/tree/main

Place them in `models/face/` under the EXACT filenames listed above.
Do not rename.

REMOVED in v1.0.0:
    models/face/det_10g.onnx   -- replaced by MediaPipe FaceDetector (Apache 2.0)

REMOVED from v0.3.1:
    models/face/landmark_106.onnx  -- no longer used (no 2D landmarks)
    models/face/arcface_r100.onnx  -- no longer used (no identity vector)

REMOVED from v0.4.0 skeleton:
    models/face/appearance_encoder.onnx  -- renamed to appearance_feature_extractor.onnx
    models/face/flow_warper.onnx         -- replaced by warping.onnx (feature-volume)
    models/face/decoder.onnx             -- replaced by spade_generator.onnx

Export contracts (LivePortrait reference)
=========================================

All neural inputs use RGB float32 in [0, 1], NCHW. No mean/std
normalization. Source: KwaiVGI/LivePortrait `src/live_portrait_wrapper.py`.

appearance_feature_extractor.onnx
---------------------------------
  input  : NCHW 1x3x256x256 float32 (RGB, [0,1])
  output : 5-D 1x32x16x64x64 float32 (3D feature volume)

motion_extractor.onnx
---------------------
  input  : NCHW 1x3x256x256 float32 (RGB, [0,1])
  outputs: 7 tensors (order may vary by export; names or shapes disambiguate)
           pitch : (1, 66) float32   -- softmax bins, decoded to degrees via
                                        sum(softmax(p) * idx) * 3 - 97.5
           yaw   : (1, 66) float32
           roll  : (1, 66) float32
           t     : (1, 3)  float32   -- translation (z component zeroed downstream)
           exp   : (1, 63) float32   -- 21x3 expression delta (flattened)
           scale : (1, 1)  float32   -- global scale
           kp    : (1, 63) float32   -- 21x3 canonical keypoints (flattened)

warping.onnx
------------
  inputs : 3 tensors (order may vary by export; shapes disambiguate)
           feature_3d: (1, 32, 16, 64, 64) float32  (from appearance extractor)
           kp_source : (1, 21, 3)          float32  (source implicit keypoints)
           kp_driving: (1, 21, 3)          float32  (driving implicit keypoints)
  output : (1, 32, 16, 64, 64) float32 warped feature volume

spade_generator.onnx
--------------------
  input  : (1, 32, 16, 64, 64) float32 (warped feature volume)
  output : (1, 3, 512, 512)    float32 RGB in [0, 1]

stitching_retargeting.onnx
--------------------------
  input  : (1, 126) float32  (concat(kp_source.flat, kp_driving.flat))
  output : (1, N>=63) float32 (first 63 = delta on kp_driving)

Export verification
===================

After placing the files, construct the engine (do NOT run `run()`):

    from nsr_engine.config import EngineConfig
    from nsr_engine.engine import NSREngine
    cfg = EngineConfig(headless=True)
    NSREngine(cfg)

Construction will:
  - open every ONNX session with CPUExecutionProvider (explicitly)
  - validate input/output arities against the contracts above
  - validate shape compatibility strictly (dynamic axes allowed, pinned
    axes must match)
  - compute the avatar's appearance feature + source keypoints ONCE
  - fail loudly on any mismatch

If construction completes without error, the pipeline is wired and
`run()` will start the 3-thread async loop (capture -> motion -> render)
driving the OBS output window.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
