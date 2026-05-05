# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""§14: "no landmark or TPS dependency remains" — regression guard.

Scans the installed v0.1.0 package source for any import / symbol
reintroducing a removed subsystem. If anything smuggles TPS or the
landmark stack back in, this test fails loudly before integration.
"""

from __future__ import annotations

import re
from pathlib import Path

import nsr_engine

PACKAGE_ROOT = Path(nsr_engine.__file__).resolve().parent

FORBIDDEN_MODULES = (
    "nsr_engine.face.landmark",
    "nsr_engine.face.landmark_schema",
    "nsr_engine.face.identity",
    "nsr_engine.face.blendshapes",
    "nsr_engine.face.rigid_nonrigid",
    "nsr_engine.face.observation",
    "nsr_engine.face.pipeline",
    "nsr_engine.motion.tps",
    "nsr_engine.motion.regions",
    "nsr_engine.motion.field",
    "nsr_engine.motion.pose_retargeting",
    "nsr_engine.render.avatar",
    "nsr_engine.util.one_euro",
)

# Substrings that should not appear in v0.1.0 source.
FORBIDDEN_SUBSTRINGS = (
    "TPS",
    "landmark_106",
    "arcface",
    "blendshape",
    "rigid_nonrigid",
    "pose_retargeting",
    "OneEuro",
    "one_euro",
)

# Substrings allowed despite the filters — contextual mentions in
# docstrings that describe what was removed are fine.
_WHITELIST_PATHS: tuple[str, ...] = ()


def _py_sources() -> list[Path]:
    return [p for p in PACKAGE_ROOT.rglob("*.py")]


class TestNoLegacyDependencies:
    def test_no_forbidden_modules_physically_present(self) -> None:
        for mod in FORBIDDEN_MODULES:
            rel = mod.replace("nsr_engine.", "").replace(".", "/") + ".py"
            path = PACKAGE_ROOT / rel
            assert not path.exists(), f"forbidden module still present: {path}"

    def test_no_forbidden_imports(self) -> None:
        import_re = re.compile(
            r"^\s*(?:from|import)\s+([a-zA-Z0-9_.]+)", re.MULTILINE
        )
        for src in _py_sources():
            text = src.read_text(encoding="utf-8")
            for match in import_re.finditer(text):
                name = match.group(1)
                for forbidden in FORBIDDEN_MODULES:
                    assert not name.startswith(forbidden), (
                        f"{src.relative_to(PACKAGE_ROOT)} still imports "
                        f"forbidden module: {name}"
                    )

    def test_no_forbidden_substrings_in_code(self) -> None:
        for src in _py_sources():
            if any(w in str(src) for w in _WHITELIST_PATHS):
                continue
            # Strip docstrings / comments — the lexical scan targets
            # code tokens, not prose mentioning removed features.
            text = src.read_text(encoding="utf-8")
            code_only = _strip_strings_and_comments(text)
            for needle in FORBIDDEN_SUBSTRINGS:
                assert needle.lower() not in code_only.lower(), (
                    f"{src.relative_to(PACKAGE_ROOT)} contains forbidden "
                    f"token in code: {needle!r}"
                )


def _strip_strings_and_comments(src: str) -> str:
    """Remove triple-quoted strings, single-line '#' comments, and string
    literals. Conservative: we keep code tokens only."""
    # Triple-quoted strings (greedy-safe with re.DOTALL).
    src = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
    src = re.sub(r"'''.*?'''", "", src, flags=re.DOTALL)
    # Line comments.
    src = re.sub(r"#[^\n]*", "", src)
    # Single-line strings. Crude but sufficient for this check.
    src = re.sub(r'"[^"\n]*"', "", src)
    src = re.sub(r"'[^'\n]*'", "", src)
    return src
