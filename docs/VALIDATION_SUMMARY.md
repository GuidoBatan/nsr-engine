# NSR Engine v1.0.0 (LivePortrait ONNX + MediaPipe) — Validation & Contract Specification

## Purpose

This document defines the validation surface and ABI guarantees for NSR Engine v1.0.0.  
It separates:

- Invariants enforced by automated tests
- Behaviors dependent on ONNX model presence
- Behaviors outside the automated validation boundary

Additionally, it formalizes **ABI stability via deterministic fingerprinting**.

---

## Execution modes

The suite auto-detects environment state via:

    tests/conftest.py::_all_models_present

and runs in one of two modes:

### Minimal mode (default)
Triggered when at least one required `.onnx` file is missing.

- Integration tests are skipped
- Contract, math, and system invariants are still fully enforced

### Integrated mode
Triggered when all required ONNX assets are present.

- Full pipeline is exercised end-to-end
- Includes identity, temporal, and micro-expression validation

---

## Required ONNX assets (§11)

    models/face/appearance_feature_extractor.onnx
    models/face/motion_extractor.onnx
    models/face/warping.onnx
    models/face/spade_generator.onnx
    models/face/stitching_retargeting.onnx

Face detection uses MediaPipe (bundled via pip). No model file in `models/face/` is required for detection.

---

## Contract versioning

The latent-space contract is explicitly versioned:

    NSR_LATENT_CONTRACT_VERSION == "0.1.0-liveportrait"

Rules:

- Semantic version (0.1.0) tracks engine ABI
- Suffix ("liveportrait") binds contract to model topology
- Any structural change requires version bump + fingerprint update

---

## ABI stability (fingerprint system)

The contract is protected by a deterministic fingerprint:

    src/nsr_engine/contract/fingerprint.lock

Generated via:

    uv run python scripts/generate_contract_fingerprint.py

### What the fingerprint encodes

- Tensor shapes
- Tensor dtypes
- Structural contract definitions
- Critical constants (channels, keypoints, resolutions)

### What it does NOT encode

- Runtime values
- ONNX weights
- Performance characteristics

### Enforcement

CI recomputes the fingerprint and compares it to the locked value:

    diff current.txt fingerprint.lock

Any mismatch:

→ CI FAILURE

---

## ABI change policy

### Non-breaking change (forbidden)

- Modifying shapes
- Changing dtypes
- Altering tensor semantics

Result:
    CI fails → must revert

### Breaking change (explicit)

Required steps:

1. Update contract version
2. Regenerate fingerprint
3. Commit new fingerprint.lock

This guarantees traceability and reproducibility.

---

## What is validated unconditionally

The following assertions are enforced in BOTH execution modes.

---

### 1) Data-contract integrity — tests/test_contracts.py

- Contract version is pinned:
      "0.1.0-liveportrait"

- Constants:
      INTERNAL_RES == CROP_RES
      OUTPUT_RES == 512
      NUM_KEYPOINTS == 21
      KEYPOINT_DIMS == 3
      POSE_BINS == 66

- AppearanceFeature3D:
      rejects invalid dtype, rank, channels, depth, spatial, non-finite

- MotionParams:
      rejects invalid shapes and non-finite values

- ImplicitKeypoints:
      rejects invalid count and dtype

- WarpedFeature3D:
      rejects shape mismatch

- CropResult / RenderResult:
      enforce dtype and dimensional correctness

---

### 2) Keypoint and pose math — tests/test_keypoint_transform.py

- decode_pose_bins:
      uniform → 0°
      bin 0 → -97.5°
      bin 65 → +97.5°

- rotation_matrix:
      identity at zero
      orthogonal, det = +1
      correct ZYX convention

- transform_keypoints:
      scale → rotation → expression → translation
      tz forced to zero

- apply_stitching_delta:
      correct delta application
      ignores trailing values
      non-mutating

---

### 3) Async slot semantics — tests/test_async_slot.py

- timeout returns (None, False)
- put/get consistency
- last-value retention
- drop-on-full enforced
- non-blocking put (10k ops < 1s)
- close unblocks consumer

---

### 4) Face-crop behavior — tests/test_face_crop.py

- correct shape and dtype
- center pixel preserved
- out-of-frame → black padding
- zero padding → tight crop

---

### 5) Removal of legacy dependencies — tests/test_no_legacy_deps.py

- forbidden modules absent
- no legacy imports
- forbidden tokens not present in code (excluding comments/docs)

---

### 6) Removal of inference stubs — tests/test_no_inference_stubs.py

- no NotImplementedError remains
- all wrappers importable:
      AppearanceEncoder
      MotionExtractor
      LatentFlowWarper
      SpadeGenerator
      StitchingRetargeting

---

## What is validated only in integrated mode

Requires all ONNX assets.

---

### 1) Identity preservation — tests/test_identity_preservation.py

Protocol:
- 50 frames with small noise
- re-encode + cosine similarity

Assertions:

    mean similarity ≥ 0.85
    min within 0.05 of mean

---

### 2) Micro-expression responsiveness — tests/test_micro_expression.py

Protocol:
- perturb eye/mouth regions

Assertions:

    ROI Δ > 2.0
    non-ROI Δ < 3.0

---

### 3) Temporal stability — tests/test_temporal_stability.py

Static driver:
    Δ < 1.5

Slow drift:
    coefficient of variation < 0.5

---

## Test execution summary

Execution results depend on environment:

Minimal mode:
    integration tests skipped

Integrated mode:
    full suite executed

Exact counts are not fixed and evolve over time.

Authoritative results:
    CI (GitHub Actions)

---

## Validation gaps

1. Motion extractor output mapping:
   relies on name heuristics + fallback

2. Warping input mapping:
   kp_source vs kp_driving ambiguity

3. Stitching MLP packaging:
   assumes (1, 126) input

4. Performance:
   no strict timing SLA enforced

---

## Contract boundary definition

The ABI guarantees ONLY:

- Tensor shapes
- Tensor dtypes
- Structural semantics

The ABI explicitly excludes:

- Numerical equivalence across models
- Performance guarantees
- Visual quality guarantees

---

## System integrity model

The engine enforces correctness at three levels:

1. Static contract validation (latents.py)
2. Behavioral validation (tests/)
3. Structural identity (fingerprint.lock)

This combination ensures:

- Determinism
- Reproducibility
- Safe evolution of the system

---

## Operational rule

Any modification to:

- latents.py
- tensor shapes
- structural constants

WITHOUT updating fingerprint.lock:

→ CI failure

This is intentional and required.

---

## Summary

NSR Engine v1.0.0 provides:

- Strict latent-space ABI
- Deterministic contract fingerprinting
- Dual-mode validation (minimal / integrated)
- CI-enforced structural integrity

This defines a reproducible and enforceable contract surface suitable for
production-grade evolution.