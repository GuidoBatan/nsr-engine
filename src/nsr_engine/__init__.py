# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Guido Batan
# Licensed under Apache License, Version 2.0
# See LICENSE file for full terms

"""NSR Engine — neural talking-head synthesis (CPU, offline).

v0.4.0-liveportrait: five-stage LivePortrait ONNX pipeline
(appearance_feature_extractor / motion_extractor / warping /
spade_generator / stitching_retargeting). No 2D landmarks, no TPS,
no cv2-based geometric deformation.
"""

__version__ = "0.4.0+liveportrait"
