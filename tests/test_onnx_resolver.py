# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Tests for the shared ONNX init-time resolver.

Runs without any ONNX model files. Uses a FakeSession that mimics the
`get_inputs()` / `get_outputs()` surface ORT exposes.
"""

from __future__ import annotations

import pytest

from nsr_engine.onnx.resolver import (
    InputSignature,
    OutputSignature,
    resolve_inputs,
    resolve_outputs,
)


class _FakeIO:
    def __init__(self, name: str, shape: tuple) -> None:
        self.name = name
        self.shape = list(shape)


class _FakeSession:
    def __init__(self, inputs: list[_FakeIO], outputs: list[_FakeIO]) -> None:
        self._inputs = inputs
        self._outputs = outputs

    def get_inputs(self) -> list[_FakeIO]:
        return self._inputs

    def get_outputs(self) -> list[_FakeIO]:
        return self._outputs


# ---------------------------------------------------------------------------
# Warping: one unique-shape primary, two auxiliaries to ignore
# ---------------------------------------------------------------------------


class TestWarping:
    def _session(self) -> _FakeSession:
        return _FakeSession(
            inputs=[],
            outputs=[
                _FakeIO("output",      (1, 32, 16, 64, 64)),
                _FakeIO("deformation", (1, 16, 64, 64, 3)),
                _FakeIO("879",         (1, 21, 3)),
            ],
        )

    def test_unique_shape_resolves(self) -> None:
        sess = self._session()
        m = resolve_outputs(
            sess,
            [OutputSignature(role="warped_feature",
                             shape=(1, 32, 16, 64, 64))],
            context="warping",
        )
        assert m.roles_to_names == {"warped_feature": "output"}
        assert m.strategy == {"warped_feature": "shape"}


# ---------------------------------------------------------------------------
# SCRFD: 9 outputs, 2-D (N, D) layout, shape uniquely determines role+stride
# ---------------------------------------------------------------------------


class TestSCRFD:
    def _session(self) -> _FakeSession:
        # Real graph observed: raw numeric names, (N, D) shapes.
        return _FakeSession(
            inputs=[_FakeIO("input.1", (1, 3, 640, 640))],
            outputs=[
                _FakeIO("448", (12800, 1)),
                _FakeIO("471", (3200, 1)),
                _FakeIO("494", (800, 1)),
                _FakeIO("451", (12800, 4)),
                _FakeIO("474", (3200, 4)),
                _FakeIO("497", (800, 4)),
                _FakeIO("454", (12800, 10)),
                _FakeIO("477", (3200, 10)),
                _FakeIO("500", (800, 10)),
            ],
        )

    def test_shape_resolves_all_nine(self) -> None:
        sess = self._session()
        specs = [
            OutputSignature(role="score_8",  shape=(12800, 1)),
            OutputSignature(role="score_16", shape=(3200,  1)),
            OutputSignature(role="score_32", shape=(800,   1)),
            OutputSignature(role="bbox_8",   shape=(12800, 4)),
            OutputSignature(role="bbox_16", shape=(3200,  4)),
            OutputSignature(role="bbox_32", shape=(800,   4)),
            OutputSignature(role="kps_8",    shape=(12800, 10)),
            OutputSignature(role="kps_16",   shape=(3200,  10)),
            OutputSignature(role="kps_32",   shape=(800,   10)),
        ]
        m = resolve_outputs(sess, specs, context="scrfd")
        assert m.roles_to_names == {
            "score_8": "448", "score_16": "471", "score_32": "494",
            "bbox_8":  "451", "bbox_16":  "474", "bbox_32":  "497",
            "kps_8":   "454", "kps_16":   "477", "kps_32":   "500",
        }
        assert all(v == "shape" for v in m.strategy.values())

    def test_three_d_layout_also_resolves_via_accept_ranks(self) -> None:
        """Some SCRFD exports emit (1, N, D). The resolver must accept
        either via accept_ranks."""
        sess = _FakeSession(
            inputs=[],
            outputs=[
                _FakeIO("a", (1, 12800, 1)),
                _FakeIO("b", (1, 12800, 4)),
            ],
        )
        specs = [
            OutputSignature(role="score_8", shape=(12800, 1),
                            accept_ranks=(2, 3)),
            OutputSignature(role="bbox_8",  shape=(12800, 4),
                            accept_ranks=(2, 3)),
        ]
        m = resolve_outputs(sess, specs, context="scrfd")
        assert m.roles_to_names == {"score_8": "a", "bbox_8": "b"}


# ---------------------------------------------------------------------------
# MotionExtractor: 7 outputs; (1,66) fits 3 roles, (1,63) fits 2 roles.
# Requires stage-2 name-hint + positional fallback to disambiguate.
# ---------------------------------------------------------------------------


class TestMotionExtractor:
    def test_descriptive_names_resolve_via_name_hints(self) -> None:
        sess = _FakeSession(
            inputs=[],
            outputs=[
                _FakeIO("pitch_logits", (1, 66)),
                _FakeIO("yaw_logits",   (1, 66)),
                _FakeIO("roll_logits",  (1, 66)),
                _FakeIO("t",            (1, 3)),
                _FakeIO("exp",          (1, 63)),
                _FakeIO("scale",        (1, 1)),
                _FakeIO("kp",           (1, 63)),
            ],
        )
        specs = [
            OutputSignature(role="pitch", shape=(1, 66), name_hints=("pitch",)),
            OutputSignature(role="yaw",   shape=(1, 66), name_hints=("yaw",)),
            OutputSignature(role="roll",  shape=(1, 66), name_hints=("roll",)),
            OutputSignature(role="t",     shape=(1, 3)),
            OutputSignature(role="exp",   shape=(1, 63), name_hints=("exp",)),
            OutputSignature(role="scale", shape=(1, 1)),
            OutputSignature(role="kp",    shape=(1, 63), name_hints=("kp", "keypoint")),
        ]
        m = resolve_outputs(sess, specs, context="motion")
        assert m.roles_to_names["pitch"] == "pitch_logits"
        assert m.roles_to_names["yaw"] == "yaw_logits"
        assert m.roles_to_names["roll"] == "roll_logits"
        assert m.roles_to_names["exp"] == "exp"
        assert m.roles_to_names["kp"] == "kp"
        assert m.roles_to_names["t"] == "t"
        assert m.roles_to_names["scale"] == "scale"

    def test_numeric_names_resolve_via_positional_within_group(self) -> None:
        """If names are all raw IDs (no hints land), within-group
        positional fallback assigns in source order."""
        sess = _FakeSession(
            inputs=[],
            outputs=[
                _FakeIO("100", (1, 66)),   # pitch (by source order)
                _FakeIO("101", (1, 66)),   # yaw
                _FakeIO("102", (1, 66)),   # roll
                _FakeIO("103", (1, 3)),
                _FakeIO("104", (1, 63)),   # exp
                _FakeIO("105", (1, 1)),
                _FakeIO("106", (1, 63)),   # kp
            ],
        )
        specs = [
            OutputSignature(role="pitch", shape=(1, 66), name_hints=("pitch",)),
            OutputSignature(role="yaw",   shape=(1, 66), name_hints=("yaw",)),
            OutputSignature(role="roll",  shape=(1, 66), name_hints=("roll",)),
            OutputSignature(role="t",     shape=(1, 3)),
            OutputSignature(role="exp",   shape=(1, 63), name_hints=("exp",)),
            OutputSignature(role="scale", shape=(1, 1)),
            OutputSignature(role="kp",    shape=(1, 63), name_hints=("kp",)),
        ]
        m = resolve_outputs(sess, specs, context="motion")
        # Shape-unique roles still resolve via shape.
        assert m.strategy["t"] == "shape"
        assert m.strategy["scale"] == "shape"
        # Shape-ambiguous roles fall to positional within their groups.
        assert m.roles_to_names["pitch"] == "100"
        assert m.roles_to_names["yaw"] == "101"
        assert m.roles_to_names["roll"] == "102"
        assert m.strategy["pitch"] == "positional"
        assert m.roles_to_names["exp"] == "104"
        assert m.roles_to_names["kp"] == "106"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailures:
    def test_missing_role_raises_with_io_dump(self) -> None:
        sess = _FakeSession(
            inputs=[],
            outputs=[_FakeIO("a", (1, 2, 3))],
        )
        with pytest.raises(RuntimeError) as exc:
            resolve_outputs(
                sess,
                [OutputSignature(role="needed", shape=(99, 99))],
                context="test",
            )
        msg = str(exc.value)
        assert "needed" in msg
        assert "test" in msg
        assert "Session I/O" in msg

    def test_ambiguous_without_hints_resolves_positionally(self) -> None:
        # Two roles, two same-shape outputs, no hints -> positional.
        sess = _FakeSession(
            inputs=[],
            outputs=[_FakeIO("x", (1, 5)), _FakeIO("y", (1, 5))],
        )
        m = resolve_outputs(
            sess,
            [OutputSignature(role="a", shape=(1, 5)),
             OutputSignature(role="b", shape=(1, 5))],
            context="test",
        )
        assert m.roles_to_names == {"a": "x", "b": "y"}
        assert m.strategy == {"a": "positional", "b": "positional"}

    def test_fewer_outputs_than_same_shape_roles_raises(self) -> None:
        sess = _FakeSession(
            inputs=[],
            outputs=[_FakeIO("only", (1, 5))],
        )
        with pytest.raises(RuntimeError):
            resolve_outputs(
                sess,
                [OutputSignature(role="a", shape=(1, 5)),
                 OutputSignature(role="b", shape=(1, 5))],
                context="test",
            )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class TestInputResolution:
    def test_warping_inputs_by_shape_and_hint(self) -> None:
        sess = _FakeSession(
            inputs=[
                _FakeIO("feature_3d", (1, 32, 16, 64, 64)),
                _FakeIO("kp_src",     (1, 21, 3)),
                _FakeIO("kp_drv",     (1, 21, 3)),
            ],
            outputs=[],
        )
        specs = [
            InputSignature(role="feature", shape=(1, 32, 16, 64, 64)),
            InputSignature(role="kp_source",  shape=(1, 21, 3),
                           name_hints=("src", "source")),
            InputSignature(role="kp_driving", shape=(1, 21, 3),
                           name_hints=("drv", "driving")),
        ]
        m = resolve_inputs(sess, specs, context="warping")
        assert m.roles_to_names == {
            "feature":    "feature_3d",
            "kp_source":  "kp_src",
            "kp_driving": "kp_drv",
        }
