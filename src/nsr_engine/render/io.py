# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Avatar image IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np

from nsr_engine.util.typing import U8


def load_avatar_rgba(path: Path) -> U8:
    """Load avatar as RGBA uint8 HxWx4. Validates format."""
    if not path.exists():
        raise FileNotFoundError(f"avatar not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"failed to read image: {path}")

    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        out = np.dstack([rgb, alpha]).astype(np.uint8, copy=False)
    elif img.ndim == 3 and img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        out = np.dstack([rgb, alpha]).astype(np.uint8, copy=False)
    elif img.ndim == 3 and img.shape[2] == 4:
        # cv2.cvtColor stub widens dtype to `integer | floating`; runtime
        # always returns uint8 for this conversion.
        out = cast(U8, cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    else:
        raise ValueError(f"unsupported image shape: {img.shape}")

    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def rgba_to_bgr(rgba: U8) -> U8:
    """For pipelines that need BGR (e.g. running face pipeline on avatar)."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("expected RGBA input")
    # cv2.cvtColor stub widens dtype; uint8→uint8 conversion at runtime.
    return cast(U8, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
