# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Unit tests for the last-state-wins slot used by the async pipeline.

These run without any ONNX model files.
"""

from __future__ import annotations

import threading
import time

from nsr_engine.engine import _LastWinsSlot


class TestLastWinsSlot:
    def test_empty_get_times_out(self) -> None:
        slot: _LastWinsSlot[int] = _LastWinsSlot()
        t0 = time.perf_counter()
        value, fresh = slot.get(timeout_s=0.02)
        elapsed = time.perf_counter() - t0
        assert value is None
        assert fresh is False
        assert 0.015 <= elapsed <= 0.2, f"timeout drift: {elapsed:.3f}s"

    def test_put_then_get_is_fresh(self) -> None:
        slot: _LastWinsSlot[int] = _LastWinsSlot()
        slot.put(42)
        v, fresh = slot.get(timeout_s=0.01)
        assert v == 42
        assert fresh is True

    def test_second_get_without_put_is_stale(self) -> None:
        slot: _LastWinsSlot[int] = _LastWinsSlot()
        slot.put(42)
        slot.get(timeout_s=0.01)
        v, fresh = slot.get(timeout_s=0.01)
        assert v == 42        # last_value retained
        assert fresh is False  # but not fresh

    def test_drop_on_full_keeps_latest(self) -> None:
        """Producer overwrites; consumer sees only the final value."""
        slot: _LastWinsSlot[int] = _LastWinsSlot()
        for i in range(100):
            slot.put(i)
        v, fresh = slot.get(timeout_s=0.01)
        assert v == 99
        assert fresh is True

    def test_producer_never_blocks(self) -> None:
        """Verify that put() does not wait on a consumer.

        We fire 10_000 puts in a tight loop with no consumer. If put()
        were to block, this would hang. We assert it completes quickly.
        """
        slot: _LastWinsSlot[int] = _LastWinsSlot()
        t0 = time.perf_counter()
        for i in range(10_000):
            slot.put(i)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"put() appears to block: {elapsed:.3f}s for 10k puts"

    def test_close_unblocks_waiter(self) -> None:
        slot: _LastWinsSlot[int] = _LastWinsSlot()
        result: dict[str, tuple[int | None, bool]] = {}

        def consumer() -> None:
            result["v"] = slot.get(timeout_s=5.0)

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        time.sleep(0.05)
        slot.close()
        t.join(timeout=1.0)
        assert not t.is_alive(), "consumer did not return after close()"
        assert result["v"] == (None, False)
