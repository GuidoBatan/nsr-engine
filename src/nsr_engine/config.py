# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Immutable runtime configuration for NSR v1.0.0 — LivePortrait ONNX.

All paths resolved relative to the package root. No env-driven paths,
no runtime downloads. Frozen dataclasses — no mutation after construction.

Model files (§11 — DO NOT RENAME):

    models/face/appearance_feature_extractor.onnx    # §4.1 identity 3D feature
    models/face/motion_extractor.onnx                # §4.2 structured motion
    models/face/warping.onnx                         # §4.3 feature-volume warp
    models/face/spade_generator.onnx                 # §4.4 neural decoder
    models/face/stitching_retargeting.onnx           # §4.5 identity stitching

Face detection is handled by MediaPipe (bundled, no model file needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve()
ROOT: Path = _HERE.parents[2]
MODELS_DIR: Path = ROOT / "models" / "face"
ASSETS_DIR: Path = ROOT / "assets"


@dataclass(frozen=True)
class FacePipelineConfig:
    """MediaPipe face detector config. Cropping only — no landmark/ArcFace in v1.0.0."""

    det_score_threshold: float = 0.4
    det_nms_iou: float = 0.4

    # Crop padding as fraction of bbox max dimension. 0.25 = 25% padding
    # around tight bbox, roughly matching LivePortrait crop conventions.
    crop_padding_frac: float = 0.25

    # onnxruntime threading (shared across all neural model sessions).
    ort_intra_threads: int = 2
    ort_inter_threads: int = 1


@dataclass(frozen=True)
class NeuralModelConfig:
    """Paths for the five v1.0.0 LivePortrait ONNX models.

    All five files MUST exist on disk at engine startup or construction
    will raise FileNotFoundError. There is no runtime download path
    (§1 forbids external APIs).

    Filenames are pinned by §11 and MUST NOT be renamed.
    """

    appearance_encoder: Path = MODELS_DIR / "appearance_feature_extractor.onnx"
    motion_extractor: Path = MODELS_DIR / "motion_extractor.onnx"
    warping: Path = MODELS_DIR / "warping.onnx"
    spade_generator: Path = MODELS_DIR / "spade_generator.onnx"
    stitching: Path = MODELS_DIR / "stitching_retargeting.onnx"


@dataclass(frozen=True)
class AsyncConfig:
    """Thread + queue sizing for the 3-stage async pipeline (§5).

    last-state-wins: every queue has size 1 with drop-on-full. Stalls
    are impossible by construction; at worst a stage skips a frame.
    """

    capture_queue_size: int = 1
    motion_queue_size: int = 1
    render_queue_size: int = 1
    queue_get_timeout_s: float = 0.010


@dataclass(frozen=True)
class EngineConfig:
    """Top-level engine config. Frozen; no mutation after construction."""

    # I/O
    avatar_path: Path = ASSETS_DIR / "avatar.png"
    webcam_device: int = 0
    webcam_width: int = 640
    webcam_height: int = 480

    # Processing (see util/latents.py for single source of truth on
    # INTERNAL_RES / OUTPUT_RES / CROP_RES — these fields mirror those).
    proc_res: int = 256          # CROP_RES / INTERNAL_RES
    output_width: int = 512      # OUTPUT_RES (native from SPADE)
    output_height: int = 512
    target_fps: int = 10

    # Whether to run the stitching retargeting MLP to refine driving kp
    # for better identity preservation. Disabling saves ~sub-ms per frame
    # but regresses identity stability under large driving motion.
    use_stitching: bool = True

    # Face detection throttling: run detection every N frames, reuse bbox between.
    detect_every_n: int = 3

    # Threading
    cpu_threads: int = 4

    # Determinism
    seed: int = 0

    # Display
    show_debug: bool = False
    headless: bool = False

    # Sub-configs
    face: FacePipelineConfig = field(default_factory=FacePipelineConfig)
    neural: NeuralModelConfig = field(default_factory=NeuralModelConfig)
    async_: AsyncConfig = field(default_factory=AsyncConfig)

    @property
    def frame_budget_ms(self) -> float:
        return 1000.0 / float(self.target_fps)
