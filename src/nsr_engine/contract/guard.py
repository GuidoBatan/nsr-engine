# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Hard contract gate — runtime + CI.

Computes the runtime fingerprint and compares it to the locked value
on disk. On mismatch: HARD FAIL. No fallback, no silent bypass.
"""

from __future__ import annotations

from pathlib import Path

from nsr_engine.contract.fingerprint import compute_fingerprint
from nsr_engine.util.latents import NSR_LATENT_CONTRACT_VERSION

_LOCK_PATH: Path = Path(__file__).resolve().parent / "fingerprint.lock"


def _read_lock() -> str:
    if not _LOCK_PATH.exists():
        raise RuntimeError(
            f"contract fingerprint lock missing: {_LOCK_PATH}"
        )
    return _LOCK_PATH.read_text(encoding="utf-8").strip()


def assert_contract_integrity() -> None:
    if NSR_LATENT_CONTRACT_VERSION != "0.1.0-liveportrait":
        raise RuntimeError(
            f"ABI version mismatch: {NSR_LATENT_CONTRACT_VERSION!r} "
            "!= '0.1.0-liveportrait'"
        )

    current = compute_fingerprint()
    locked = _read_lock()
    if current != locked:
        raise RuntimeError(
            "LATENT ABI BREAKING CHANGE DETECTED: "
            f"runtime={current} locked={locked}"
        )
