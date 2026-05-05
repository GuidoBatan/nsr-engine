# Notices

NSR Engine v1.0.0
Copyright 2026 Guido Batan
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0. You may not use this project
except in compliance with the License. You may obtain a copy at:
https://www.apache.org/licenses/LICENSE-2.0

---

## Upstream Attribution

### KwaiVGI/LivePortrait
- **License:** MIT
- **URL:** https://github.com/KwaiVGI/LivePortrait
- **Usage:** Architecture, motion keypoint formulas, and ONNX model weights (externally sourced).
  Model weights are NOT distributed with this repository. They must be downloaded
  separately from https://huggingface.co/myn0908/Live-Portrait-ONNX/tree/main.
  The license governing those weights is determined solely by their upstream source.

### Google MediaPipe
- **Project:** google-ai-edge/mediapipe
- **License:** Apache-2.0
- **URL:** https://github.com/google-ai-edge/mediapipe
- **Usage:** Face detection (FaceDetector Tasks API). TFLite model bundled within
  the mediapipe pip package; no separate download required.
