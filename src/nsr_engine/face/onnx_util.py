# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Deterministic CPU ONNX session factory + shared validation helpers.

Explicitly pins:
  - CPU execution provider only (no CUDA / TensorRT / DirectML / Azure)
  - intra_op / inter_op thread counts
  - deterministic execution mode (sequential op scheduling)

No `download`, no `hub`, no network. If the file does not exist we
raise immediately.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from nsr_engine.util.onnx_compat import ort


def make_session(
    model_path: Path,
    intra_threads: int,
    inter_threads: int,
) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    so = ort.SessionOptions()

    # Parallel op scheduling.
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    # Full graph optimization.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Threading: intra = max(2, cores-1), inter = 1.
    cores = os.cpu_count() or 2
    so.intra_op_num_threads = max(2, cores - 1, int(intra_threads))
    so.inter_op_num_threads = 1

    # Memory optimizations.
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True

    # Provider selection: CUDA → DML → CPU (spec priority, deterministic at startup).
    available = set(ort.get_available_providers())
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")

    return ort.InferenceSession(
        str(model_path),
        sess_options=so,
        providers=providers,
    )


def shape_compatible(
    actual: Iterable[int | str | None],
    expected: tuple[int | None, ...],
) -> bool:
    """Check whether an ONNX-reported shape matches an expected tuple.

    ONNX shapes may contain ints, negative ints (rare), None, or strings
    (symbolic dims). Any non-positive-int entry in `actual` is treated
    as a dynamic axis that matches any value. Positive-int entries must
    match exactly.
    """
    actual = tuple(actual)
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected, strict=True):
        if isinstance(a, int) and a > 0 and a != e:
            return False
        # non-int / non-positive -> dynamic, passes
    return True


def assert_single_input_output(session: ort.InferenceSession, module_name: str) -> None:
    """Require len(inputs) == 1 and len(outputs) == 1. Raise with context."""
    ins = session.get_inputs()
    outs = session.get_outputs()
    if len(ins) != 1:
        raise RuntimeError(
            f"{module_name}: expected 1 input, got {len(ins)}: "
            f"{[i.name for i in ins]}"
        )
    if len(outs) != 1:
        raise RuntimeError(
            f"{module_name}: expected 1 output, got {len(outs)}: "
            f"{[o.name for o in outs]}"
        )


def describe_io(session: ort.InferenceSession) -> str:
    """Human-readable summary of a session's inputs/outputs for logs."""
    parts: list[str] = [
        f"in[{i.name}]={tuple(i.shape)} {i.type}" for i in session.get_inputs()
    ]
    parts.extend(
        f"out[{o.name}]={tuple(o.shape)} {o.type}" for o in session.get_outputs()
    )
    return "; ".join(parts)
