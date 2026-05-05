# Changelog

## [1.0.0] — 2026-04-27

### Changed
- **BREAKING:** Replaced InsightFace SCRFD (det_10g.onnx) with MediaPipe FaceDetector
- License: Non-commercial (CC-BY-NC) → Apache 2.0 (fully commercial)
- Removed onnxruntime dependency for face detection (MediaPipe bundles TFLite inference)
- Updated all Python files with SPDX license headers

### Added
- MediaPipe integration (google-ai-edge/mediapipe ≥0.10.9)
- New test: test_face_detector_mediapipe
- SPDX-License-Identifier headers on all Python source files
- Author metadata in pyproject.toml

### Performance
- Precision: ~2-3% lower than SCRFD (acceptable for portfolio demo)
- Latency: Similar (CPU-first, edge-device optimized)
- Reproducibility: Models auto-bundled (no external downloads for face detection)

### Migration for users
Old: `from nsr_engine.face.detector import SCRFD`
New: `from nsr_engine.face.detector import FaceDetector`
Output format: Identical (backward compatible at module level).

## [0.4.0] — 2026-04-27

### Added
- Initial public release
- Five-stage LivePortrait ONNX pipeline (appearance encoder → motion extractor → warping → SPADE generator → stitching)
- RGL-based latent handling and contract-integrity guard (`fingerprint.lock`)
- CPU-only execution via `onnxruntime` CPUExecutionProvider
- Webcam capture module with async scheduling
- ONNX model validation with strict shape/dtype contracts
- Comprehensive test suite (14 test modules)

### Removed
- `engine_bkp.py` development artifact

### Notes
- Derivative of KwaiVGI/LivePortrait (MIT license for code; see NOTICE.md)
- `det_10g.onnx` (InsightFace buffalo_l) restricts this build to non-commercial use
- Models fetched at runtime via `scripts/fetch_models.py`
