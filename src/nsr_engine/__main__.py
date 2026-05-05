# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""CLI entrypoint. Thread-env MUST be set before heavy imports."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from nsr_engine.util.determinism import apply_thread_env, seed_all


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nsr-engine",
        description=(
            "NSR v0.4.0 neural talking-head engine "
            "(CPU-only, offline, LivePortrait ONNX pipeline)."
        ),
    )
    p.add_argument("--avatar", type=Path, default=None, help="RGBA PNG avatar.")
    p.add_argument("--webcam", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    args = _parse_args()
    apply_thread_env(args.threads)
    seed_all(args.seed)
    _setup_logging()

    from nsr_engine.config import ASSETS_DIR, EngineConfig
    from nsr_engine.engine import NSREngine

    logger = logging.getLogger("nsr.main")
    cfg = EngineConfig(
        avatar_path=args.avatar or ASSETS_DIR / "avatar.png",
        webcam_device=args.webcam,
        target_fps=args.fps,
        cpu_threads=args.threads,
        seed=args.seed,
        headless=args.headless,
        show_debug=args.debug,
    )
    logger.info("starting: %s", cfg)
    try:
        with NSREngine(cfg) as engine:
            engine.run()
        return 0
    except FileNotFoundError as exc:
        logger.error("missing file: %s", exc)
        return 2
    except ValueError as exc:
        logger.error("invalid input: %s", exc)
        return 3
    except RuntimeError as exc:
        logger.error("runtime failure: %s", exc)
        return 4
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
