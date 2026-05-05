# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Contract fingerprint — single source of truth.

Delegates to nsr_engine.util.latents.compute_latent_contract_fingerprint
so the latent ABI is described in exactly one place. There is no
second, divergent spec.
"""

from __future__ import annotations

from nsr_engine.util.latents import (
    NSR_LATENT_CONTRACT_FINGERPRINT,
    compute_latent_contract_fingerprint,
)


def compute_fingerprint() -> str:
    return compute_latent_contract_fingerprint()


__all__ = ["compute_fingerprint", "NSR_LATENT_CONTRACT_FINGERPRINT"]
