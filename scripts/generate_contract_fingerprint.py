# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

from nsr_engine.contract.fingerprint import compute_fingerprint


def main() -> None:
    print(compute_fingerprint())


if __name__ == "__main__":
    main()
