# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

# tests/test_contract_fingerprint.py
"""
Contract fingerprint tests.

Covers:
- determinism
- lockfile consistency (optional in local, enforced in CI)
"""

from __future__ import annotations

from pathlib import Path

from nsr_engine.contract.fingerprint import compute_fingerprint

LOCK_PATH = Path("src/nsr_engine/contract/fingerprint.lock")


def _read_lock() -> str:
    return LOCK_PATH.read_text().strip()


class TestContractFingerprint:
    def test_fingerprint_is_deterministic(self) -> None:
        f1 = compute_fingerprint()
        f2 = compute_fingerprint()
        assert f1 == f2

    def test_fingerprint_format(self) -> None:
        f = compute_fingerprint()
        assert isinstance(f, str)
        assert len(f) == 64  # sha256 hex
        int(f, 16)  # must be valid hex

    def test_lockfile_exists(self) -> None:
        assert LOCK_PATH.exists()

    def test_lockfile_matches_current_contract(self) -> None:
        """
        This test enforces ABI lock ONLY if lockfile exists.

        Behavior:
        - Local dev: can update lock intentionally
        - CI: must match exactly (fail if drift)
        """
        current = compute_fingerprint()
        locked = _read_lock()
        assert current == locked, (
            "Contract fingerprint mismatch.\n"
            "If this change is intentional:\n"
            "  1. Regenerate fingerprint\n"
            "  2. Commit updated fingerprint.lock\n"
        )
