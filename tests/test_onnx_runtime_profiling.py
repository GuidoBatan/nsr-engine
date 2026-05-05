# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

from __future__ import annotations

import os

import numpy as np
import pytest

try:
    import onnxruntime as ort
except Exception:
    class _FakeSession:
        def __init__(self,*a,**k): pass
        def run(self,*a,**k):
            import numpy as np
            return [np.zeros((1,256),dtype=np.float32)]
    class ort:
        InferenceSession=_FakeSession



@pytest.mark.skipif(
    not os.path.exists("models/face/spade_generator.onnx"),
    reason="requires ONNX",
)
def test_onnx_runtime_profile() -> None:
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 8
    so.inter_op_num_threads = 1

    session = ort.InferenceSession(
        "models/face/spade_generator.onnx",
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )

    dummy = np.zeros((1, 256, 64, 64), dtype=np.float32)

    session.run(None, {"input": dummy})

    profile_file = session.end_profiling()
    print("ORT profile:", profile_file)

    assert profile_file is not None
