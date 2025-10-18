# Quickstart: anamorph_fit

This walkthrough demonstrates how to fit aberration coefficients from synthetic data
and from Quadoa-style exports.

## 1. Synthetic wavefront

```python
import numpy as np
from anamorph_fit import fit_aberrations

x = np.linspace(-1.0, 1.0, 64)
y = np.linspace(-1.0, 1.0, 64)
X, Y = np.meshgrid(x, y, indexing="xy")
W = 0.05 * X**3 - 0.02 * Y**2 + 0.01 * X * Y

result = fit_aberrations(X, Y, W)
print("RMS error:", result.rms_error)
for name, value in zip(result.names, result.values):
    print(f"{name:>15}: {value:+.4e}")
```

## 2. CSV export

```bash
poetry run anamorph-fit examples/sample_anamorphic.csv --json
poetry run anamorph-fit examples/sample_anamorphic.csv --save-plot residual.png --basis heatmap
```

This writes `residual.png` and `residual_basis.png` alongside the sample data.

## 3. JSON export

```bash
poetry run anamorph-fit examples/sample_quadoa.json --export-json fit.json
```

The `examples/sample_quadoa.json` file mimics a minimal Quadoa export with three
sample points.
