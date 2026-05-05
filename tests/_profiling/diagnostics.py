# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

from __future__ import annotations

import numpy as np


def _finite_ok(arr: np.ndarray) -> bool:
    return np.all(np.isfinite(arr))


def _shape_ok(arr: np.ndarray, shape: tuple[int, ...]) -> bool:
    return arr.shape == shape


def _pairwise_distances(kp: np.ndarray) -> np.ndarray:
    """
    kp: (1, K, 3)
    retorna distancias (K, K)
    """
    pts = kp[0]  # (K,3)
    diffs = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diffs * diffs, axis=-1))
    return d


def audit_keypoints(
    kp_source: np.ndarray,
    kp_driving: np.ndarray,
    name: str = "kp",
) -> dict[str, float]:
    """
    Auditoría geométrica para detectar:
    - explosión de escala
    - traslación extrema
    - NaNs/inf
    - colapso (todos puntos iguales)
    - drift violento entre frames
    """

    assert kp_source.ndim == 3 and kp_source.shape[2] == 3
    assert kp_driving.ndim == 3 and kp_driving.shape[2] == 3

    out: dict[str, float] = {}

    # Finite
    out[f"{name}_finite_source"] = float(_finite_ok(kp_source))
    out[f"{name}_finite_driving"] = float(_finite_ok(kp_driving))

    # Centroid
    c_src = kp_source.mean(axis=1)  # (1,3)
    c_drv = kp_driving.mean(axis=1)

    out[f"{name}_centroid_shift_norm"] = float(
        np.linalg.norm((c_drv - c_src)[0])
    )

    # Spread (escala)
    d_src = _pairwise_distances(kp_source)
    d_drv = _pairwise_distances(kp_driving)

    # evitar diagonal cero
    mask = ~np.eye(d_src.shape[0], dtype=bool)

    mean_src = float(d_src[mask].mean())
    mean_drv = float(d_drv[mask].mean())

    out[f"{name}_mean_pair_dist_src"] = mean_src
    out[f"{name}_mean_pair_dist_drv"] = mean_drv

    if mean_src > 0:
        out[f"{name}_scale_ratio_drv/src"] = mean_drv / mean_src
    else:
        out[f"{name}_scale_ratio_drv/src"] = float("inf")

    # colapso
    out[f"{name}_collapse_src"] = float(mean_src < 1e-6)
    out[f"{name}_collapse_drv"] = float(mean_drv < 1e-6)

    return out


def audit_warped_feature(x: np.ndarray) -> dict[str, float]:
    """
    WarpedFeature3D: (1,256,64,64)
    """
    out: dict[str, float] = {}
    out["warp_finite"] = float(np.all(np.isfinite(x)))
    out["warp_min"] = float(np.min(x))
    out["warp_max"] = float(np.max(x))
    out["warp_mean"] = float(np.mean(x))
    out["warp_std"] = float(np.std(x))
    return out


def audit_rgb(rgb: np.ndarray) -> dict[str, float]:
    """
    RGB uint8 HxWx3
    """
    out: dict[str, float] = {}
    out["rgb_min"] = float(np.min(rgb))
    out["rgb_max"] = float(np.max(rgb))
    out["rgb_mean"] = float(np.mean(rgb))
    out["rgb_std"] = float(np.std(rgb))
    return out


def assert_no_loco_face(metrics: dict[str, float]) -> None:
    """
    Criterios duros para “cara de loco”.
    Ajustados para LivePortrait típico.
    """

    # Finite
    assert metrics.get("kp_finite_source", 1.0) == 1.0, "NaN en kp_source"
    assert metrics.get("kp_finite_driving", 1.0) == 1.0, "NaN en kp_driving"

    # Escala razonable (empírico)
    scale = metrics.get("kp_scale_ratio_drv/src", 1.0)
    assert 0.3 < scale < 3.0, f"Escala anómala: {scale}"

    # Traslación razonable
    shift = metrics.get("kp_centroid_shift_norm", 0.0)
    assert shift < 2.0, f"Traslación excesiva: {shift}"

    # No colapso
    assert metrics.get("kp_collapse_src", 0.0) == 0.0, "Colapso source"
    assert metrics.get("kp_collapse_drv", 0.0) == 0.0, "Colapso driving"

    # Warped feature rango
    if "warp_std" in metrics:
        assert metrics["warp_std"] > 1e-6, "Feature plano (sin señal)"
        assert abs(metrics["warp_mean"]) < 10.0, "Mean fuera de rango"

    # RGB sanity
    if "rgb_std" in metrics:
        assert metrics["rgb_std"] > 1.0, "Imagen casi constante"
