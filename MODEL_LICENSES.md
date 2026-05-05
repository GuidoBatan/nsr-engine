# Model Licenses

NSR Engine does NOT distribute model weights.
All model files must be obtained by the user from third-party sources prior to use.
The licenses below govern the weights as published by their respective authors.

---

## LivePortrait ONNX weights

- **Files:**
  - `appearance_feature_extractor.onnx`
  - `motion_extractor.onnx`
  - `warping.onnx`
  - `spade_generator.onnx`
  - `stitching_retargeting.onnx`
- **Source:** https://huggingface.co/myn0908/Live-Portrait-ONNX
- **Upstream project:** https://github.com/KwaiVGI/LivePortrait
- **Upstream license:** MIT
- **Status:** Weights are derived from the KwaiVGI/LivePortrait project.
  The ONNX exports at the source above are third-party conversions.
  NSR Engine makes no representation regarding redistribution rights of these
  specific exports. The user is responsible for verifying license terms at the
  source repository before use or redistribution.
- **Redistribution:** NOT guaranteed. Obtain directly from source.

---

## MediaPipe face detection model

- **File:** `face_detection_short_range.tflite` (bundled inside mediapipe pip package)
- **Source:** https://github.com/google-ai-edge/mediapipe
- **License:** Apache-2.0
- **Status:** Bundled and distributed by Google as part of the mediapipe package.
  No separate download required. Redistribution permitted under Apache-2.0.

---

## Disclaimer

The NSR Engine source code is licensed under Apache-2.0.
This license does NOT extend to model weights obtained from third-party sources.
Each model is subject to its own license terms.
The user assumes full responsibility for compliance with applicable model licenses.
