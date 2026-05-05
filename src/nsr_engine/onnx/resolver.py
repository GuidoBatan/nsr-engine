# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""Shared ONNX init-time name resolver.

Single entry point used by all NSR ONNX wrappers to map semantic roles
(like "pitch_logits", "score_8", "warped_feature") to the actual output
tensor names in a loaded session. Returns a `{role: name}` dict; does
not read, allocate, or run tensors.

Resolution strategy (strict priority)
-------------------------------------
For each declared role, the resolver tries methods in this order:

  1. SHAPE SIGNATURE — match the role's expected shape against the
     session's declared output shapes. If exactly one session output
     matches, use it.

  2. NAME HINTS — if multiple session outputs have the same shape (so
     step 1 is ambiguous), attempt substring matching on lowercased
     output names against the role's `name_hints`.

  3. POSITIONAL (within a same-shape group) — if roles share an
     expected shape AND name hints still don't disambiguate, assign
     roles to same-shape session outputs in the source order declared
     by the caller. This is the only positional fallback.

At every level, the resolver fails loudly with the full session I/O
signature on the error if any role cannot be uniquely resolved.

What this resolver DOES NOT do
------------------------------
- It does not run inference. It operates only on declared shapes and
  names from `session.get_outputs()` / `session.get_inputs()`.
- It does not return tensors. Callers run their own `session.run(...)`
  with the names returned here.
- It does not define a per-model contract type. Each caller declares
  its own role specs inline as a list of `OutputSignature` / `InputSignature`.
- It does not enforce `len(outputs) == 1` or any output-arity rule.
  Arity is implicit: each declared role must resolve, extra outputs
  are ignored.

Contract tolerance
------------------
Role shapes accept `None` entries as wildcards for dynamic axes. This
is how ONNX exports typically report batch/symbolic dims. The
resolver normalizes rank mismatches via explicit `accept_ranks` so a
caller can say "accept either (N, D) or (1, N, D)" without the
resolver guessing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from nsr_engine.util.onnx_compat import ort

# An ONNX tensor shape: each axis is a positive int (concrete) or a string
# (named symbolic dim) or None (unknown / dynamic).
OnnxShape: TypeAlias = tuple[int | str | None, ...]

# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputSignature:
    """Declarative spec for one expected output of an ONNX graph.

    `shape` entries may be positive ints (exact match required),
    `None` (wildcard for that axis), or `-1` (wildcard, alias of None
    for ergonomics when callers want to use that convention).

    `accept_ranks`, when non-empty, allows shape matching at multiple
    ranks — the resolver tries `shape` at every listed rank via
    leading-axis stripping or prepending. Use this for exports where
    the batch dim may or may not be present (e.g. SCRFD's (N, D) vs
    (1, N, D)).

    `name_hints` is a tuple of lowercased substrings; an output name
    matches a hint if it contains the substring as a substring of
    its lowercased form. Hints are used only to disambiguate when
    shape matching produces multiple candidates.
    """

    role: str
    shape: tuple[int | None, ...]
    name_hints: tuple[str, ...] = ()
    accept_ranks: tuple[int, ...] = ()


@dataclass(frozen=True)
class InputSignature:
    """Declarative spec for one expected input of an ONNX graph.

    Same fields as OutputSignature; separate type so static callers
    don't accidentally pass an OutputSignature to `resolve_inputs`.
    """

    role: str
    shape: tuple[int | None, ...]
    name_hints: tuple[str, ...] = ()
    accept_ranks: tuple[int, ...] = ()


@dataclass(frozen=True)
class ResolvedMap:
    """Result of a resolution. Immutable.

    `roles_to_names` is the primary output: `{role → declared name}`.
    `strategy` reports which method resolved each role, for logs/CI.
    `io_dump` is the full session signature, for error display and for
    `validate_onnx_resolution` to print on success.
    """

    roles_to_names: dict[str, str]
    strategy: dict[str, str]  # role -> "shape" | "name" | "positional"
    io_dump: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_outputs(
    session: ort.InferenceSession,
    specs: Sequence[OutputSignature],
    *,
    context: str = "onnx",
) -> ResolvedMap:
    """Resolve `specs` to output names in `session`. See module docstring."""
    outputs = session.get_outputs()
    return _resolve(
        [(o.name, tuple(o.shape)) for o in outputs],
        specs,
        context=context,
        which="output",
        io_dump=_describe_session(session),
    )


def resolve_inputs(
    session: ort.InferenceSession,
    specs: Sequence[InputSignature],
    *,
    context: str = "onnx",
) -> ResolvedMap:
    """Resolve `specs` to input names in `session`. See module docstring."""
    inputs = session.get_inputs()
    return _resolve(
        [(i.name, tuple(i.shape)) for i in inputs],
        specs,
        context=context,
        which="input",
        io_dump=_describe_session(session),
    )


def validate_onnx_resolution(
    session: ort.InferenceSession,
    output_specs: Sequence[OutputSignature] = (),
    input_specs: Sequence[InputSignature] = (),
    *,
    context: str = "onnx",
    print_result: bool = True,
) -> tuple[ResolvedMap | None, ResolvedMap | None]:
    """CLI/CI helper: attempt resolution and report the outcome.

    Returns (output_map, input_map). Either may be None if no specs
    were provided for that side. Raises RuntimeError on any resolution
    failure, with the full session I/O dump attached.
    """
    out_map = resolve_outputs(session, output_specs, context=context) if output_specs else None
    in_map = resolve_inputs(session, input_specs, context=context) if input_specs else None

    if print_result:
        print(f"[validate_onnx_resolution] {context}")
        print(f"  session I/O: {_describe_session(session)}")
        if out_map is not None:
            for role in (s.role for s in output_specs):
                print(
                    f"  output role={role!r} -> name={out_map.roles_to_names[role]!r} "
                    f"(via {out_map.strategy[role]})"
                )
        if in_map is not None:
            for role in (s.role for s in input_specs):
                print(
                    f"  input  role={role!r} -> name={in_map.roles_to_names[role]!r} "
                    f"(via {in_map.strategy[role]})"
                )
        print("  PASS")

    return out_map, in_map


# ---------------------------------------------------------------------------
# Core resolution
# ---------------------------------------------------------------------------


@dataclass
class _Bucket:
    """Mutable working state while resolving roles that share a shape."""

    session_names: list[str] = field(default_factory=list)
    role_specs: list[OutputSignature | InputSignature] = field(default_factory=list)


def _resolve(
    session_entries: list[tuple[str, OnnxShape]],
    specs: Sequence[OutputSignature | InputSignature],
    *,
    context: str,
    which: str,
    io_dump: str,
) -> ResolvedMap:
    """Run the three-stage resolution. See module docstring."""
    # Stage 0: partition session entries into candidate pools per spec.
    # Every spec gets its list of candidates (session entries whose shape
    # matches the spec at at least one of its accepted ranks).
    candidates: dict[str, list[str]] = {}
    for spec in specs:
        candidates[spec.role] = [
            name for name, shape in session_entries
            if _shape_fits(shape, spec)
        ]

    roles_to_names: dict[str, str] = {}
    strategy: dict[str, str] = {}
    used_names: set[str] = set()

    # Stage 1: unique-shape resolution.
    # If a role has exactly one candidate AND that candidate is unique to
    # this role (no other spec claims it), accept.
    for spec in specs:
        cands = [n for n in candidates[spec.role] if n not in used_names]
        if len(cands) == 1:
            # Only accept if no later spec also matches exclusively to this name.
            # (If multiple roles match the same candidate set, we defer to stage 2.)
            exclusive = all(
                cands[0] not in candidates[s.role]
                for s in specs
                if s.role != spec.role
            )
            if exclusive:
                roles_to_names[spec.role] = cands[0]
                strategy[spec.role] = "shape"
                used_names.add(cands[0])

    # Stage 2: among remaining specs, group by shape-candidate-set and
    # apply name hints, then positional fallback within each group.
    remaining = [s for s in specs if s.role not in roles_to_names]
    # Group remaining specs by the tuple of their still-eligible candidates.
    groups: dict[tuple[str, ...], _Bucket] = {}
    for spec in remaining:
        cands_t = tuple(n for n in candidates[spec.role] if n not in used_names)
        bucket = groups.setdefault(cands_t, _Bucket())
        bucket.role_specs.append(spec)
        for n in cands_t:
            if n not in bucket.session_names:
                bucket.session_names.append(n)

    for cand_tuple, bucket in groups.items():
        roles_in_bucket = bucket.role_specs
        names_in_bucket = list(cand_tuple)

        # Sanity: there must be at least as many session outputs as roles.
        if len(names_in_bucket) < len(roles_in_bucket):
            raise _fail(
                context, which, io_dump,
                f"insufficient {which}s for roles "
                f"{[s.role for s in roles_in_bucket]}: "
                f"only {len(names_in_bucket)} candidates {names_in_bucket}"
            )

        # 2a: name-hint matching within the bucket.
        hint_claimed: dict[str, str] = {}  # role -> name
        for spec in roles_in_bucket:
            hint_matches = [
                n for n in names_in_bucket
                if n not in hint_claimed.values()
                and any(h in n.lower() for h in spec.name_hints)
            ]
            if len(hint_matches) == 1:
                hint_claimed[spec.role] = hint_matches[0]

        for role, name in hint_claimed.items():
            roles_to_names[role] = name
            strategy[role] = "name"
            used_names.add(name)

        # 2b: positional fallback for any role in this bucket still unresolved.
        # Assign in the source-declared order of the role_specs, pulling from
        # the session_names in their original declaration order, skipping any
        # already claimed above.
        unresolved = [s for s in roles_in_bucket if s.role not in roles_to_names]
        pool = [n for n in names_in_bucket if n not in used_names]
        if len(pool) < len(unresolved):
            raise _fail(
                context, which, io_dump,
                f"name hints left ambiguity for roles "
                f"{[s.role for s in unresolved]}: only {len(pool)} "
                f"candidates remain in {names_in_bucket}"
            )
        # zip is intentionally short: pool may exceed unresolved (extra
        # outputs ignored); the length guard above rules out the other
        # direction. strict=False makes this explicit.
        for spec, name in zip(unresolved, pool, strict=False):
            roles_to_names[spec.role] = name
            strategy[spec.role] = "positional"
            used_names.add(name)

    # Final check: every declared role is resolved.
    missing = [s.role for s in specs if s.role not in roles_to_names]
    if missing:
        raise _fail(
            context, which, io_dump,
            f"could not resolve role(s) {missing}"
        )

    return ResolvedMap(
        roles_to_names=roles_to_names,
        strategy=strategy,
        io_dump=io_dump,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shape_fits(actual: OnnxShape, spec: OutputSignature | InputSignature) -> bool:
    """Return True if `actual` matches `spec.shape` at any accepted rank.

    Rank rule:
      - If ranks are equal, match per-axis.
      - If ranks differ, reject UNLESS `actual` contains at least one
        symbolic (non-positive-int) dim; in that case, attempt padding
        `actual` with leading wildcards to the spec's rank, or stripping
        leading axes off `actual` down to the spec's rank, and retry.

    Type rule (per axis):
      - int vs int   : must match exactly
      - str vs int   : wildcard (session symbolic dim matches any spec int)
      - None / str / non-positive-int on either side: wildcard
    """
    accepted = spec.accept_ranks if spec.accept_ranks else (len(spec.shape),)
    for rank in accepted:
        if rank == len(spec.shape):
            if _shape_match(actual, spec.shape):
                return True
        elif rank == len(spec.shape) + 1:
            if _shape_match(actual, (1,) + spec.shape):
                return True
            if _shape_match(actual, (None,) + spec.shape):
                return True
        elif rank == len(spec.shape) - 1 and len(spec.shape) >= 1:
            stripped = spec.shape[1:]
            if _shape_match(actual, stripped):
                return True

    # Rank-mismatch fallback: only if `actual` has symbolic dims.
    # Some exporters (e.g. LivePortrait warping) report shapes like
    # ('batch_size', 1, 'Sigmoidoutput_dim_2', 'Sigmoidoutput_dim_3')
    # — rank 4 — when the runtime tensor is actually rank 5. We accept
    # these by padding `actual` with leading wildcards, or stripping
    # leading axes, to match the spec's rank, then per-axis matching.
    if not _has_symbolic(actual):
        return False
    spec_rank = len(spec.shape)
    actual_rank = len(actual)
    if actual_rank < spec_rank:
        pad = (None,) * (spec_rank - actual_rank)
        if _shape_match(pad + tuple(actual), spec.shape):
            return True
    elif actual_rank > spec_rank:
        stripped_actual = tuple(actual)[actual_rank - spec_rank:]
        if _shape_match(stripped_actual, spec.shape):
            return True
    return False


def _has_symbolic(shape: OnnxShape) -> bool:
    """True if any axis is not a positive int (i.e. None, str, or <=0)."""
    return any(not (isinstance(d, int) and d > 0) for d in shape)


def _shape_match(actual: OnnxShape, expected: tuple[int | None, ...]) -> bool:
    """Strict per-axis match: None/str/negative in either side is wildcard."""
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected, strict=True):
        a_wild = not (isinstance(a, int) and a > 0)
        e_wild = e is None or (isinstance(e, int) and e <= 0)
        if a_wild or e_wild:
            continue
        if a != e:
            return False
    return True


def _describe_session(session: ort.InferenceSession) -> str:
    parts: list[str] = [
        f"in[{i.name!r}]={tuple(i.shape)}" for i in session.get_inputs()
    ]
    parts.extend(
        f"out[{o.name!r}]={tuple(o.shape)}" for o in session.get_outputs()
    )
    return " ; ".join(parts)


def _fail(context: str, which: str, io_dump: str, detail: str) -> RuntimeError:
    return RuntimeError(
        f"{context}: {which} resolution failed — {detail}. "
        f"Session I/O: {io_dump}"
    )


__all__ = [
    "InputSignature",
    "OutputSignature",
    "ResolvedMap",
    "resolve_inputs",
    "resolve_outputs",
    "validate_onnx_resolution",
]
