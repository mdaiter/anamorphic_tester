# anamorph_fit

Optical aberration fitting utilities for anamorphic lenses. The toolkit evaluates the
20-term third-order anamorphic basis from *Primary Aberration Theory for General
Anamorphic Systems* (§8.4, Eqs. 8‑32 → 8‑51) and solves least-squares fits against
Quadoa wavefront data (local exports or the live REST API).

## Prerequisites

- Python 3.11+
- Poetry (for dependency management)

## Installation

```bash
poetry install
poetry install --extras csv          # enable pandas-based CSV parsing
poetry install --extras plot         # enable matplotlib plotting
poetry install --extras "csv plot"   # install both extras
```

## Usage

### Command line

The CLI provides three subcommands. All logging honours `--log-level` or `-v`.

```bash
# Analyze a live/exported Quadoa wavefront (fits Eq. 8-32 – 8-51 coefficients)
poetry run anamorph-fit analyze --lens L001 --field 30 --base-url exports/ --export-json result.json

# Compute metrics (RMS_x/RMS_y, skew strength, anamorphic ratio) for a saved export
poetry run anamorph-fit metrics --input wavefront.json --json

# Inspect the schema parsed from docs/quadoa_api/*
poetry run anamorph-fit dump-schema
```

Options:

- `--base-url` points to a REST endpoint (https) or directory with Quadoa exports.
- `--export-json` writes coefficients, RMS, fit metrics, and metadata to disk.
- `--plot` / `--save-plot` render residual heatmaps (requires `--extras plot`).
- `--basis heatmap` additionally emits a basis-correlation heatmap.

> The orthonormal basis requires at least 16 pupil samples (≥4×4 grid) to remain well-conditioned.

Example data: see `examples/sample_anamorphic.csv` and `examples/sample_quadoa.json`.
For a short walkthrough, open `examples/quickstart.md`.

### Quadoa schema extraction

Run the helper script to regenerate the schema used by `dump-schema`:

```bash
poetry run python tools/extract_quadoa_schema.py
cat generated/quadoa_schema.json
```

The script parses every `.html` / `.rst` file under `docs/quadoa_api/`, recording
field names, formats, and documentation line numbers for traceability.

### Python API

```python
import numpy as np
from anamorph_fit.fit import fit_aberrations_full
from anamorph_fit.io import parse_quadoa_export

# Synthetic wavefront
X, Y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64), indexing="xy")
W = 0.05 * X**3 - 0.03 * Y**2
synthetic = fit_aberrations_full(X, Y, W)

# Live/recorded Quadoa export (Eq. 8-32 – 8-51 mapping)
pupil = parse_quadoa_export(lens_id="L001", field_angle=30.0, base_url="https://quadoa/api/")
X, Y = np.meshgrid(pupil.x, pupil.y, indexing="xy")
result = fit_aberrations_full(X, Y, pupil.wavefront)
print(result.metrics["rms_ratio_xy"], result.metrics["skew_relative_strength"])
```

The helper :class:`quadoa.client.QuadoaClient` normalises wavefront units, resamples scattered data with `scipy.interpolate.griddata`, and retrieves system metadata.

## Testing

```bash
POETRY_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest
```

The integration tests cover documentation parsing, Quadoa payload normalisation, CLI flows, and the Eq. (8-32 – 8-51) basis. Install extras to enable the optional cases.
