# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Determinism controls. MUST be imported before numpy/cv2/onnxruntime
do any heavy allocation if thread caps are to take effect. See
`apply_thread_env`.
"""

from __future__ import annotations

import os
import random

import numpy as np


def apply_thread_env(threads: int) -> None:
    """Pin BLAS/OpenMP thread counts. Call BEFORE importing numpy-heavy modules."""
    t = str(int(threads))
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = t


def seed_all(seed: int) -> None:
    """Seed python + numpy. No torch. No cudnn. Nothing hidden."""
    random.seed(seed)
    np.random.seed(seed)
