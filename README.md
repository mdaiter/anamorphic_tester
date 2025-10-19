# anamorph_fit

Optical aberration fitting utilities for anamorphic lenses. This package evaluates a
16-term aberration basis and solves least-squares fits against wavefront data exported
from Quadoa (CSV or JSON).

## Prerequisites

- Python 3.11+
- Poetry (for dependency management)

## Installation

```bash
poetry install
```

This pulls runtime dependencies (NumPy, SciPy, Pydantic). To opt into CSV parsing or
plotting support, install the corresponding extras:

```bash
poetry install --extras csv
poetry install --extras plot
poetry install --extras csv --extras plot  # install both
```

## Usage

### Command line

```bash
poetry run anamorph-fit path/to/export.csv --json
poetry run anamorph-fit path/to/export.csv --plot
poetry run anamorph-fit path/to/export.csv --export-json fit.json
poetry run anamorph-fit path/to/export.csv --save-plot residual.png --basis heatmap
```

> The orthonormal basis requires at least 16 pupil samples (e.g., a 4×4 grid or denser).

Supported options:

- `--json` prints coefficients as JSON (names, values, RMS).
- `--export-json PATH` writes the same payload to disk.
- `--plot` displays a residual heatmap (requires matplotlib).
- `--save-plot PATH` saves the residual heatmap to an image file; combine with `--basis heatmap` to also export a correlation heatmap.
- `--basis heatmap` visualizes basis correlations to diagnose conditioning.
- `--log-level LEVEL` or `-v/--verbose` adjust CLI logging verbosity.

Example data: see `examples/sample_anamorphic.csv` and `examples/sample_quadoa.json`.
For a short walkthrough, open `examples/quickstart.md`.

### Python API

```python
import numpy as np
from anamorph_fit.fit import fit_aberrations

X, Y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64), indexing="xy")
W = 0.05 * X**3 - 0.03 * Y**2

result = fit_aberrations(X, Y, W)
print(dict(zip(result.names, result.values)))
print("RMS:", result.rms_error)
```

To load directly from Quadoa exports:

```python
from anamorph_fit.fit import fit_from_file

result = fit_from_file("path/to/export.json")
```

## Testing

```bash
POETRY_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest
```

CSV- and CLI-related tests are skipped if `pandas` is absent. Install it to run the
complete suite:

```bash
poetry add pandas
```
