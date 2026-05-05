# NSR Engine v1.0.0

CPU neural talking-head synthesis (offline). LivePortrait architecture with MediaPipe face detection.
[![Tests](https://github.com/GuidoBatan/nsr-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/GuidoBatan/nsr-engine/actions)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-green)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows-lightgrey.svg)
![Architecture](https://img.shields.io/badge/arch-LivePortrait--ONNX-orange.svg)

## Author

**Guido Batan** | Software Systems Architect (AI/ML)

This project is part of a portfolio demonstrating software architecture and AI/ML systems design.
See [guidobatan.com](https://guidobatan.com) for more.

## Features

- **Apache 2.0 Licensed** — commercial-friendly, redistribution permitted
- **CPU-First Design** — runs on CPU; GPU optional for acceleration
- **No External Model Downloads for Face Detection** — MediaPipe models bundled via pip
- Five-stage LivePortrait pipeline (F → M → W → G → S) fully integrated
- Strict ONNX shape/dtype contract validation at startup
- 3-thread async pipeline (Capture → Motion → Render) with last-state-wins scheduling

## Architecture

**Face Detection:** MediaPipe FaceDetector (Apache 2.0)
**Appearance Encoder:** LivePortrait appearance_feature_extractor.onnx
**Motion Pipeline:** LivePortrait motion_extractor.onnx + warping.onnx + spade_generator.onnx + stitching_retargeting.onnx

Derived from KwaiVGI/LivePortrait (MIT).
See NOTICE.md for full attribution.

## Installation

The project is dependency-locked via `requirements.lock` (generated with
`uv pip compile pyproject.toml`). Install from the lockfile for byte-for-byte
reproducibility, then add the package itself in editable mode.

### With `uv` (recommended)

```bash
uv venv --python 3.11
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
uv pip sync requirements.lock
uv pip install -e ".[dev]" --no-deps
```

### With `pip`

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.lock
pip install -e ".[dev]" --no-deps
```

The `--no-deps` flag is intentional: dependency resolution belongs to the
lockfile, not to `pyproject.toml`. This is the same flow used in CI
(`.github/workflows/tests.yml`).

> **Note on `uv run`.** Avoid `uv run pytest` for this project. `uv run`
> resolves an isolated environment from `pyproject.toml` (without the
> lockfile), which can pull different transitive versions and trigger
> re-downloads of large packages such as `mediapipe`. Activate `.venv`
> and invoke tools directly.

## Model Acquisition (Neural Pipeline)

Source: [LivePortrait ONNX (HuggingFace)](https://huggingface.co/myn0908/Live-Portrait-ONNX/tree/main)

Download the following models into `models/face/`:

- `appearance_feature_extractor.onnx`
- `motion_extractor.onnx`
- `warping.onnx`
- `spade_generator.onnx`
- `stitching_retargeting.onnx`

Do not rename files. The engine expects these exact filenames for session resolution.

| Pipeline stage | File |
|----------------|------|
| F - appearance | appearance_feature_extractor.onnx |
| M - motion     | motion_extractor.onnx |
| W - warping    | warping.onnx |
| G - generator  | spade_generator.onnx |
| S - stitching  | stitching_retargeting.onnx |

Full ONNX I/O contracts:
```bash
python scripts/print_model_contracts.py   # prints to stdout; no download
```

## Execution

```bash
python -m nsr_engine --avatar assets/avatar.png --debug
```

The engine workflow:
1. Initializes MediaPipe face detector and five ONNX sessions.
2. Validates session arity and shape contracts against specifications.
3. Computes and freezes avatar appearance features and source latents.
4. Orchestrates the 3-thread asynchronous pipeline (Capture → Motion → Render).

## Testing

After installation (with `.venv` active):

```bash
ruff check .
mypy src/
pytest tests/ -v
```

Two execution modes:

- **Skeleton mode** (default): one or more of the five LivePortrait ONNX
  files are missing from `models/face/`. Tests marked `requires_onnx`
  are skipped; data-contract, wiring, and shape-assertion tests run
  unconditionally.
- **Integrated mode**: all five ONNX files present. Regression tests
  (identity preservation, temporal stability, responsiveness) activate.

Detection is conservative — partial presence fails closed (skeleton).

## Documentation

- `docs/ARCHITECTURE.md`: Data flow, concurrency model, and I/O contracts.
- `docs/VALIDATION_SUMMARY.md`: Test suite coverage and inference verification.
- `CHANGELOG.md`: Version history.

## License

Licensed under Apache License 2.0. See [LICENSE](./LICENSE) for details.

This project is a derivative of [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) (MIT License).
See [NOTICE.md](NOTICE.md) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for full attribution.
