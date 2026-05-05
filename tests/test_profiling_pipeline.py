# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pytest

from nsr_engine.config import ASSETS_DIR, EngineConfig
from nsr_engine.engine import NSREngine
from nsr_engine.motion.keypoint_transform import transform_keypoints
from nsr_engine.util.latents import OUTPUT_RES, MotionParams
from tests._profiling.diagnostics import (
    assert_no_loco_face,
    audit_keypoints,
    audit_rgb,
    audit_warped_feature,
)
from tests._profiling.profiler import Profiler

REQUIRED_ONNX = [
    "models/face/appearance_feature_extractor.onnx",
    "models/face/motion_extractor.onnx",
    "models/face/warping.onnx",
    "models/face/spade_generator.onnx",
    "models/face/stitching_retargeting.onnx",
]

AVATAR_PATH = ASSETS_DIR / "avatar.png"

EYE_ROI = (
    int(OUTPUT_RES * 0.30),
    int(OUTPUT_RES * 0.30),
    int(OUTPUT_RES * 0.70),
    int(OUTPUT_RES * 0.50),
)
MOUTH_ROI = (
    int(OUTPUT_RES * 0.30),
    int(OUTPUT_RES * 0.60),
    int(OUTPUT_RES * 0.70),
    int(OUTPUT_RES * 0.85),
)

MICRO_FLOOR = 2.0
STATIC_CEILING = 6.0


def _all_models_present() -> bool:
    return all(os.path.exists(p) for p in REQUIRED_ONNX)


def _roi(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2].astype(np.float32)


def _non_roi_mean_abs(
    img_a: np.ndarray,
    img_b: np.ndarray,
    rois: list[tuple[int, int, int, int]],
) -> float:
    mask = np.ones(img_a.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in rois:
        mask[y1:y2, x1:x2] = False
    a = img_a.astype(np.float32)[mask]
    b = img_b.astype(np.float32)[mask]
    return float(np.mean(np.abs(a - b)))


def _perturb_eye_mouth(crop: np.ndarray) -> np.ndarray:
    """
    Simula micro-expresión noising eye + mouth en el crop 256x256.
    """
    h, w = crop.shape[:2]
    out = crop.copy()
    eye = (int(w * 0.30), int(h * 0.30), int(w * 0.70), int(h * 0.50))
    mouth = (int(w * 0.30), int(h * 0.60), int(w * 0.70), int(h * 0.85))
    rng = np.random.default_rng(42)

    for x1, y1, x2, y2 in (eye, mouth):
        patch = out[y1:y2, x1:x2].astype(np.int16)
        noise = rng.integers(-40, 41, size=patch.shape, dtype=np.int16)
        out[y1:y2, x1:x2] = np.clip(patch + noise, 0, 255).astype(np.uint8)

    return out


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "data"):
        return np.asarray(x.data)
    return np.asarray(x)


def _as_motion_params(out: Any) -> MotionParams:
    if isinstance(out, MotionParams):
        return out

    if isinstance(out, (tuple, list)) and len(out) >= 7:
        return MotionParams(
            pitch=np.asarray(out[0], dtype=np.float32),
            yaw=np.asarray(out[1], dtype=np.float32),
            roll=np.asarray(out[2], dtype=np.float32),
            t=np.asarray(out[3], dtype=np.float32),
            exp=np.asarray(out[4], dtype=np.float32),
            scale=np.asarray(out[5], dtype=np.float32),
            kp_canonical=np.asarray(out[6], dtype=np.float32),
        )

    raise TypeError(
        f"No pude convertir {type(out).__name__} a MotionParams. "
        "Esperaba MotionParams o tupla/lista de 7 elementos."
    )


def _ensure_uint8_rgb(rgb: Any) -> np.ndarray:
    """
    Normaliza salida RGB a uint8 HxWx3.
    """
    arr = np.asarray(rgb)

    if arr.ndim != 3:
        raise ValueError(f"RGB inválido: {arr.shape}")

    if arr.shape[2] == 4:
        arr = arr[..., :3]

    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.5:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)

    return np.clip(arr, 0, 255).astype(np.uint8)


def _extract_avatar_crop(engine: NSREngine) -> np.ndarray:
    """
    Obtiene el crop frozen del avatar desde el engine.
    """
    avatar_crop = getattr(engine, "_avatar_crop", None)
    if avatar_crop is None:
        raise AttributeError("NSREngine no expone _avatar_crop")

    if hasattr(avatar_crop, "bgr"):
        arr = avatar_crop.bgr
    elif hasattr(avatar_crop, "data"):
        arr = avatar_crop.data
    else:
        arr = avatar_crop

    return np.asarray(arr).copy()


def _run_pipeline(
    engine: NSREngine,
    driver_crop: np.ndarray,
    prof: Profiler | None = None,
):
    """
    Pipeline real respetando tipos internos.

    - MotionExtractor devuelve MotionParams (o algo convertible a eso)
    - transform_keypoints devuelve ImplicitKeypoints
    - stitcher.refine exige ImplicitKeypoints
    - warper y spade consumen los objetos del pipeline
    """
    if prof is None:
        t_motion = t_kp = t_stitch = t_warp = t_spade = None
    else:
        t_motion = prof.track("motion")
        t_kp = prof.track("kp_transform")
        t_stitch = prof.track("stitch")
        t_warp = prof.track("warp")
        t_spade = prof.track("spade")

    with (t_motion or _nullcontext()):
        motion = _as_motion_params(engine._motion_ext.extract(driver_crop))

    with (t_kp or _nullcontext()):
        kp_driving = transform_keypoints(motion)

    if engine._stitcher is not None:
        with (t_stitch or _nullcontext()):
            kp_driving = engine._stitcher.refine(engine._kp_source, kp_driving)

    with (t_warp or _nullcontext()):
        warped = engine._warper.warp(
            engine._avatar_feature,
            engine._kp_source,
            kp_driving,
        )

    with (t_spade or _nullcontext()):
        rgb = engine._spade.decode(warped)

    return motion, kp_driving, warped, _ensure_uint8_rgb(rgb)


def _nullcontext():
    class _N:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _N()


@pytest.mark.skipif(not _all_models_present(), reason="requires ONNX models")
class TestPipelineProfiling:
    def test_profile_and_audit_pipeline(self) -> None:
        """
        Profiling exacto del pipeline real + auditoría de micro-expresión.
        No modifica core, no inventa step(), no agrega hooks al engine.
        """
        cfg = EngineConfig(headless=True, avatar_path=AVATAR_PATH)

        t0 = time.perf_counter()
        engine = NSREngine(cfg)
        init_ms = (time.perf_counter() - t0) * 1000.0

        required = [
            "_motion_ext",
            "_warper",
            "_spade",
            "_avatar_feature",
            "_kp_source",
            "_avatar_crop",
        ]
        missing = [name for name in required if not hasattr(engine, name)]
        assert not missing, f"NSREngine no expone internals requeridos: {missing}"

        static_driver = _extract_avatar_crop(engine)
        perturbed_driver = _perturb_eye_mouth(static_driver)

        # Warmup no medido.
        _run_pipeline(engine, static_driver, None)
        _run_pipeline(engine, perturbed_driver, None)

        prof = Profiler()
        audit_acc: list[dict[str, float]] = []

        out_static = None
        out_perturbed = None

        static_repeats = 10
        perturbed_repeats = 10

        for _ in range(static_repeats):
            with prof.track("TOTAL_FRAME"):
                motion, kp_drv, warped, rgb = _run_pipeline(engine, static_driver, prof)
            out_static = rgb
            audit_acc.append(audit_warped_feature(_to_numpy(warped)))
            audit_acc.append(audit_rgb(rgb))
            audit_acc.append(
                audit_keypoints(
                    _to_numpy(engine._kp_source),
                    _to_numpy(kp_drv),
                    name="kp_static",
                )
            )

        for _ in range(perturbed_repeats):
            with prof.track("TOTAL_FRAME"):
                motion, kp_drv, warped, rgb = _run_pipeline(
                    engine, perturbed_driver, prof
                )
            out_perturbed = rgb
            audit_acc.append(audit_warped_feature(_to_numpy(warped)))
            audit_acc.append(audit_rgb(rgb))
            audit_acc.append(
                audit_keypoints(
                    _to_numpy(engine._kp_source),
                    _to_numpy(kp_drv),
                    name="kp_perturbed",
                )
            )

        assert out_static is not None
        assert out_perturbed is not None

        print("\n=== INIT ===")
        print(f"engine_init_ms={init_ms:.2f}")

        print("\n=== PROFILING ===")
        print(prof.dump())

        stats = prof.stats()
        assert "TOTAL_FRAME" in stats, "No se registró TOTAL_FRAME"
        assert "motion" in stats, "No se registró motion"
        assert "kp_transform" in stats, "No se registró kp_transform"
        assert "warp" in stats, "No se registró warp"
        assert "spade" in stats, "No se registró spade"

        total_avg = stats["TOTAL_FRAME"]["avg"]
        total_p95 = stats["TOTAL_FRAME"]["p95"]

        assert total_avg < 5000, f"Pipeline anormalmente lento: avg={total_avg:.2f}ms"
        assert total_p95 < 8000, f"Jitter excesivo: p95={total_p95:.2f}ms"

        heavy = max(
            (k for k in stats.keys() if k != "TOTAL_FRAME"),
            key=lambda k: stats[k]["avg"],
        )
        print(f"\nBOTTLENECK: {heavy} ({stats[heavy]['avg']:.2f} ms)")

        eye_delta = float(
            np.mean(np.abs(_roi(out_perturbed, EYE_ROI) - _roi(out_static, EYE_ROI)))
        )
        mouth_delta = float(
            np.mean(
                np.abs(_roi(out_perturbed, MOUTH_ROI) - _roi(out_static, MOUTH_ROI))
            )
        )
        non_roi_delta = _non_roi_mean_abs(
            out_perturbed,
            out_static,
            [EYE_ROI, MOUTH_ROI],
        )

        print("\n=== MICRO-EXPRESSION ===")
        print(f"eye_delta     = {eye_delta:.3f}")
        print(f"mouth_delta   = {mouth_delta:.3f}")
        print(f"non_roi_delta = {non_roi_delta:.3f}")

        assert eye_delta > MICRO_FLOOR, (
            f"eye ROI did not respond: mean |Δ| = {eye_delta:.3f}, floor = {MICRO_FLOOR:.3f}"
        )
        assert mouth_delta > MICRO_FLOOR, (
            f"mouth ROI did not respond: mean |Δ| = {mouth_delta:.3f}, floor = {MICRO_FLOOR:.3f}"
        )
        assert non_roi_delta < STATIC_CEILING, (
            f"non-ROI regions changed too much: mean |Δ| = {non_roi_delta:.3f}, "
            f"ceiling = {STATIC_CEILING:.3f}"
        )

        merged: dict[str, float] = {}
        for d in audit_acc:
            merged.update(d)

        print("\n=== AUDIT ===")
        for k, v in sorted(merged.items()):
            print(f"{k:35s} {v}")

        assert_no_loco_face(merged)
