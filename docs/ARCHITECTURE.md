# NSR Engine v1.0.0 — Architecture (LivePortrait ONNX + MediaPipe)

## 1. Invariants

- CPU-only. Every ONNX session is opened with `CPUExecutionProvider` explicitly (`face/onnx_util.py::make_session`). No auto-resolve.
- Runtime dependencies: `numpy==1.26.4`, `opencv-python==4.8.1.78`, `onnxruntime==1.18.0`, `mediapipe==0.10.14`. No others admitted (§7).
- No network access at runtime. All model files must be present on disk before construction; see `scripts/fetch_models.py`.
- Appearance is encoded once per session and frozen (§4.1). Motion is per-frame (§4.2).
- No 2D landmarks, no TPS, no cv2-based geometric deformation pipeline (§1). The LivePortrait pipeline uses implicit 3D keypoints internal to the motion model; these are NOT landmarks and are consumed only by the warping + stitching ONNX graphs.

## 2. Data flow (five-stage LivePortrait pipeline, §11)

```
 avatar (init only)                       webcam BGR (per frame)
      │                                              │
      ▼                                              ▼
 FaceCropper (MediaPipe) ─► CROP_RES BGR          FaceCropper (MediaPipe) ─► CROP_RES BGR
      │                                              │
      ▼                                              ▼
 AppearanceEncoder       MotionExtractor (source init)      MotionExtractor (per frame)
 (ONNX, F)                (ONNX, M)                          (ONNX, M)
      │                        │                              │
      ▼                        ▼                              ▼
 AppearanceFeature3D      MotionParams                   MotionParams
 (1,32,16,64,64) f32      (pitch,yaw,roll,t,exp,         (pitch,yaw,roll,t,exp,
 [FROZEN]                 scale,kp_canonical)            scale,kp_canonical)
                              │                              │
                              ▼                              ▼
                         transform_keypoints          transform_keypoints
                              │                              │
                              ▼                              ▼
                         kp_source                      kp_driving (raw)
                         (1,21,3) [FROZEN]                     │
                                                             │
                                                  StitchingRetargeting (ONNX, S, optional)
                                                             │
                                                             ▼
                                                      kp_driving (refined)
                                                             │
 AppearanceFeature3D ──┐                                      │
 kp_source ────────────┼──────────────────────────────────────┤
                        ▼                                      ▼
                   WarpingNetwork (ONNX, W) ◄──────────────────┘
                        │
                        ▼
                   WarpedFeature3D (1,32,16,64,64) f32
                        │
                        ▼
                   SpadeGenerator (ONNX, G)
                        │
                        ▼
                   RGB uint8 (OUTPUT_RES, OUTPUT_RES, 3) — native 512×512, no cv2 upscale
```

Every arrow above carries a validated dataclass from `util/latents.py`. Shape, dtype, and finiteness are checked at each boundary. ONNX loaders validate input/output arity and shape against the expected contracts at `__init__` and fail loudly on mismatch.

## 3. Init-time freezes

Three tensors are computed once in `NSREngine.__init__` and reused every frame:

| Tensor                    | Shape                    | Source                                                            |
|--------------------------|--------------------------|-------------------------------------------------------------------|
| `_avatar_feature`        | `(1, 32, 16, 64, 64)`    | `AppearanceEncoder.encode(avatar_crop)`                            |
| `_avatar_motion`         | structured MotionParams  | `MotionExtractor.extract(avatar_crop)`                             |
| `_kp_source`             | `(1, 21, 3)`             | `transform_keypoints(_avatar_motion)`                              |

No recomputation during `run()`.

## 4. Concurrency model (§5)

Three worker threads + main display loop:

```
┌──────────────┐    cap_slot     ┌─────────────┐    motion_slot     ┌──────────────┐    render_slot   ┌────────────┐
│ capture thr  │ ──────────────► │ motion thr  │ ────────────────► │ render thr   │ ───────────────► │ main loop  │
└──────────────┘                  └─────────────┘                   └──────────────┘                  └────────────┘
 webcam.read               MediaPipe crop + motion                transform_keypoints →
                           extract                           stitching → warping →
                                                              spade decode
```

Every slot has capacity 1 with **drop-on-full** semantics (`_LastWinsSlot`). Producers never block; consumers either get the freshest value or, if nothing new has been produced, the last-consumed value with `fresh=False`. No stage can stall another.

## 5. Keypoint transform

`motion/keypoint_transform.py` implements the canonical→implicit keypoint transform in pure numpy. Formula (matching upstream `KwaiVGI/LivePortrait::transform_keypoint`):

```
R = R_z(roll) · R_y(yaw) · R_x(pitch)        # ZYX intrinsic, degrees
kp = kp_canonical @ Rᵀ + exp                 # (1, 21, 3)
kp = scale · kp
kp = kp + t_xy0                              # only (x, y) translated; t_z zeroed
```

Pose bin decoding (motion extractor emits softmax over 66 bins for each Euler angle):

```
degree = Σ( softmax(logits)[i] · i ) · 3 − 97.5
```

Both are covered by `tests/test_keypoint_transform.py`.

## 6. ONNX I/O contracts (validated at load)

| Module                         | Filename (§11, do not rename)          | Input(s)                                     | Output(s)                                 |
|--------------------------------|-----------------------------------------|----------------------------------------------|-------------------------------------------|
| `AppearanceEncoder`            | `appearance_feature_extractor.onnx`      | `(1, 3, 256, 256)` f32 RGB [0,1]             | `(1, 32, 16, 64, 64)` f32                 |
| `MotionExtractor`              | `motion_extractor.onnx`                  | `(1, 3, 256, 256)` f32 RGB [0,1]             | 7 outputs: 3×`(1,66)`, `(1,3)`, `(1,1)`, 2×`(1,63)` |
| `LatentFlowWarper`             | `warping.onnx`                           | `(1,32,16,64,64)` + `(1,21,3)` + `(1,21,3)`   | `(1, 32, 16, 64, 64)` f32                 |
| `SpadeGenerator`               | `spade_generator.onnx`                   | `(1, 32, 16, 64, 64)` f32                    | `(1, 3, 512, 512)` f32 RGB [0,1]          |
| `StitchingRetargeting`         | `stitching_retargeting.onnx`             | `(1, 126)` f32                               | `(1, N≥63)` f32                            |

Input/output **names** are discovered at runtime from `session.get_inputs()`/`get_outputs()` — never hardcoded. Shape compatibility is checked at init; runtime shape is re-checked on first inference.

### Motion extractor output resolution

`motion_extractor.onnx` exports vary in:
- output **names** (may be descriptive or `output_0`…`output_6`),
- output **ordering**.

The loader resolves the seven outputs in two passes:
1. **Name-based**: substring match on lowercased names against the role hints (`pitch`, `yaw`, `roll`, `t`/`translation`, `exp`/`expression`, `scale`, `kp`/`keypoint`/`canonical`). If every output maps to exactly one role and all seven roles are covered, this mapping is used.
2. **Shape-based fallback**: group by shape signature (3×`(1,66)`, 1×`(1,3)`, 1×`(1,1)`, 2×`(1,63)`). Within the pose group, source order is taken as `(pitch, yaw, roll)` (upstream convention). Within the exp/kp group, source order is `(exp, kp)`.

If the fallback is engaged and your export uses a different ordering, the loader will not detect it. Verification: check that the first 66-tensor maps to pitch by feeding a driver with known tilted pose.

### Warping input resolution

Same pattern. The 5-D tensor is unambiguously the feature volume; the two 3-D `(1, 21, 3)` tensors are disambiguated by name hint (`source`/`src` vs `driving`/`drv`/`target`), falling back to source order (source first, driving second).

## 7. Performance note (§10 relaxed per contract override)

The original v0.1.0 target of ≤25 ms/frame on Ryzen 3 CPU is **not achievable** with the LivePortrait FP32 export. The SPADE generator alone is the dominant cost on CPU. Community CPU benchmarks for the full LivePortrait pipeline are in the 150–300 ms/frame range. Per the contract override ("relax real-time constraint, prioritize correctness and stability"), the engine no longer enforces a per-frame budget; the `last-state-wins` slot topology ensures the display loop remains responsive regardless of render thread latency.

## 8. Failure modes

| Condition                                 | Behavior                                           |
|-------------------------------------------|----------------------------------------------------|
| Any ONNX file missing                    | `FileNotFoundError` at `__init__`                  |
| ONNX shape mismatch vs contract          | `RuntimeError` at `__init__` with exact detail     |
| Motion extractor output count ≠ 7        | `RuntimeError` at `__init__`                       |
| MediaPipe returns no face on avatar      | `RuntimeError` at `__init__`                       |
| MediaPipe returns no face on webcam frame| Render thread reuses last MotionParams             |
| Webcam returns None                      | Capture thread skips; pipeline idles               |
| Runtime tensor shape mismatch            | `RuntimeError` from the specific module            |
| Runtime non-finite values                | `ValueError` from dataclass validate()             |

There is **no silent fallback path**. The engine either runs with real inference or fails loudly at construction.
