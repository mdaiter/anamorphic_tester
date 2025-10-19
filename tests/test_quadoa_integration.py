"""Integration tests for the Quadoa client and anamorphic fitter.

References
----------
- docs/quadoa_api/wavefront_export.html :: *Anamorphic Wavefront Data*
- Eqs. (8-32) – (8-51) for the A/B/C basis definitions
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anamorph_fit.fit import fit_aberrations_full
from anamorph_fit.io import parse_quadoa_export
from quadoa.client import QuadoaClient
from tools import extract_quadoa_schema


def _make_docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs" / "quadoa_api"
    docs.mkdir(parents=True, exist_ok=True)

    (docs / "wavefront_export.html").write_text(
        """
        <html><body>
        <h1>Anamorphic Wavefront Data</h1>
        <p>/optical/wavefront/export</p>
        <pre>{"x": [], "y": [], "opd": [], "wavelength_nm": 550.0}</pre>
        <ul>
          <li>opd (float)</li>
          <li>wavelength_nm (float)</li>
          <li>units (string)</li>
        </ul>
        </body></html>
        """,
        encoding="utf-8",
    )

    (docs / "system_export.rst").write_text(
        """
        /optical/system/{lens_id}

        * lens_id (string)
        * surface_count (int)
        * anamorphic_ratio (float)
        """,
        encoding="utf-8",
    )
    return docs


def test_schema_extraction(tmp_path, monkeypatch):
    docs = _make_docs(tmp_path)
    schema_output = tmp_path / "schema.json"

    monkeypatch.setattr(extract_quadoa_schema, "DOC_ROOT", docs)
    monkeypatch.setattr(extract_quadoa_schema, "OUTPUT_PATH", schema_output)

    schema = extract_quadoa_schema.extract_schema()
    assert "/optical/wavefront/export" in schema
    assert schema["/optical/wavefront/export"]["fields"]
    extract_quadoa_schema.main()
    assert schema_output.exists()


def _write_quadoa_payload(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-1.2, 1.2, 7).tolist()
    y = np.linspace(-0.8, 0.8, 5).tolist()
    mesh = np.outer(np.linspace(0.08, -0.04, 5), np.linspace(0.12, -0.1, 7))

    wavefront = {
        "metadata": {
            "wavelength_nm": 550.0,
            "units": "micrometers",
        },
        "wavefront": {
            "x": x,
            "y": y,
            "opd": mesh.tolist(),
            "units": "micrometers",
        },
    }
    (base / "L100_field25.00.json").write_text(json.dumps(wavefront), encoding="utf-8")

    system = {
        "system": {
            "lens_id": "L100",
            "surface_count": 14,
            "anamorphic_ratio": 1.4,
        }
    }
    (base / "L100_system.json").write_text(json.dumps(system), encoding="utf-8")


def test_quadoa_client_normalization(tmp_path):
    base = tmp_path / "exports"
    _write_quadoa_payload(base)

    client = QuadoaClient(str(base))
    wavefront = client.fetch_wavefront("L100", 25)
    assert wavefront["units"] == "waves"
    assert wavefront["wavefront"].ndim == 2

    system = client.fetch_system("L100")
    assert system["anamorphic_ratio"] == 1.4

    pupil = parse_quadoa_export(wavefront)
    assert pupil.metadata["anamorphic_metrics"]["anamorphic_ratio"] > 1

    X, Y = np.meshgrid(pupil.x, pupil.y, indexing="xy")
    result = fit_aberrations_full(X, Y, pupil.wavefront)
    assert result.metrics["rms_ratio_xy"] >= 0
    assert result.metrics["skew_relative_strength"] >= 0
