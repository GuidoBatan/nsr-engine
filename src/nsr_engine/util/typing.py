# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Shared type aliases. Keep cv2/numpy out of the public surface elsewhere."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

U8 = NDArray[np.uint8]
F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
I32 = NDArray[np.int32]

# Conventional shape annotations (documentation only, not enforced):
# Frame:      U8  (H, W, 3)   BGR  — webcam / internal
# FrameRGBA:  U8  (H, W, 4)   RGBA — avatar / output
# Lm2D:       F32 (N, 2)      pixel coords in input frame
# Lm2DNorm:   F32 (N, 2)      normalized to proc_res, range [0,1]
# BBox:       F32 (4,)        x1, y1, x2, y2 in pixels
# Affine2D:   F32 (2, 3)      2D affine matrix (cv2.warpAffine)
# DeltaL:     F32 (N, 2)      non-rigid landmark residual (expression)
