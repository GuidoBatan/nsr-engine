# Third-Party Licenses

This project (NSR Engine) is licensed under Apache-2.0.
The following third-party packages are used at runtime and are NOT modified or redistributed.
Each package is obtained from PyPI at installation time.

---

## numpy — BSD-3-Clause

- Version: 1.26.4
- SPDX: BSD-3-Clause
- URL: https://numpy.org
- Source: https://github.com/numpy/numpy
- License text: https://github.com/numpy/numpy/blob/main/LICENSE.txt

---

## opencv-python — Apache-2.0

- Version: 4.8.1.78
- SPDX: Apache-2.0
- URL: https://opencv.org
- Source: https://github.com/opencv/opencv-python
- License text: https://github.com/opencv/opencv/blob/master/LICENSE

---

## onnxruntime — MIT

- Version: 1.18.0
- SPDX: MIT
- URL: https://onnxruntime.ai
- Source: https://github.com/microsoft/onnxruntime
- License text: https://github.com/microsoft/onnxruntime/blob/main/LICENSE

---

## mediapipe — Apache-2.0

- Version: 0.10.14
- SPDX: Apache-2.0
- URL: https://ai.google.dev/edge/mediapipe
- Source: https://github.com/google-ai-edge/mediapipe
- License text: https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE

Bundled TFLite model (face_detection_short_range.tflite) is included in the
mediapipe pip package and is subject to the same Apache-2.0 license.

---

## Dev dependencies (not included in distribution)

| Package         | Version  | SPDX     |
|-----------------|----------|----------|
| pytest          | 8.2.2    | MIT      |
| ruff            | 0.5.0    | MIT      |
| mypy            | 1.10.0   | MIT      |

---

_This file was generated from `pyproject.toml` pinned dependencies.
Licenses verified against official project repositories._
