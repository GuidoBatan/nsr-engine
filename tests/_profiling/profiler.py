# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class Profiler:
    """
    Profiler de alta resolución (ns -> ms) con agregación por etiqueta.
    Produce métricas: avg, p95, max, n.
    """

    def __init__(self) -> None:
        self.data: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def track(self, name: str):
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            self.data[name].append(dt_ms)

    def stats(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for k, v in self.data.items():
            if not v:
                continue
            v_sorted = sorted(v)
            n = len(v)
            p95 = v_sorted[int(0.95 * (n - 1))]
            out[k] = {
                "avg": sum(v) / n,
                "p95": p95,
                "max": max(v),
                "n": n,
            }
        return out

    def dump(self) -> str:
        lines = []
        for k, s in sorted(self.stats().items()):
            lines.append(
                f"{k:30s} avg={s['avg']:8.3f}ms p95={s['p95']:8.3f}ms "
                f"max={s['max']:8.3f}ms n={s['n']}"
            )
        return "\n".join(lines)
