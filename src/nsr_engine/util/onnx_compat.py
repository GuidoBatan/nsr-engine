"""Centralized onnxruntime import shim.

Provides a single :data:`ort` symbol that resolves to the real
:mod:`onnxruntime` module when installed, or to a minimal stub that allows
import-time access in skeleton-mode test environments where onnxruntime is
intentionally absent.

The stub only implements the surface needed for graceful degradation; any
attempt to actually run inference against it will raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import onnxruntime as ort
else:
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover — exercised only without onnxruntime
        import numpy as np

        class _FakeSession:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def run(self, *args: Any, **kwargs: Any) -> list[Any]:
                return [np.zeros((1, 256), dtype=np.float32)]

        class _FakeOrtModule:
            InferenceSession = _FakeSession

        ort = _FakeOrtModule()  # type: ignore[assignment]


__all__ = ["ort"]
