"""CLI smoke tests for the anamorphic analyzer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_wavefront_export(tmp_path: Path) -> Path:
    x = np.linspace(-1.0, 1.0, 5).tolist()
    y = np.linspace(-0.5, 0.5, 4).tolist()
    wavefront = (np.outer(np.linspace(0.05, -0.05, 4), np.linspace(0.1, -0.1, 5))).tolist()

    payload = {
        "metadata": {
            "wavelength_nm": 550.0,
            "units": "micrometers",
            "export_type": "wavefront",
        },
        "wavefront": {
            "x": x,
            "y": y,
            "opd": wavefront,
            "units": "micrometers",
        },
    }

    path = tmp_path / "L001_field30.00.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    system = {
        "system": {
            "lens_id": "L001",
            "surface_count": 12,
            "anamorphic_ratio": 1.2,
        }
    }
    (tmp_path / "L001_system.json").write_text(json.dumps(system), encoding="utf-8")
    return path


def test_cli_analyze_local(tmp_path):
    _write_wavefront_export(tmp_path)
    out = tmp_path / "result.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "anamorph_fit",
            "--log-level",
            "warning",
            "analyze",
            "--lens",
            "L001",
            "--field",
            "30",
            "--base-url",
            str(tmp_path),
            "--export-json",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert "metrics" in payload
    assert payload["metrics"]["rms_ratio_xy"] >= 0


def test_cli_metrics(tmp_path):
    wf_path = _write_wavefront_export(tmp_path)

    subprocess_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "anamorph_fit",
            "metrics",
            "--input",
            str(wf_path),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(subprocess_result.stdout)
    assert metrics["anamorphic_ratio"] > 1
    assert "skew_relative_strength" in metrics


def test_cli_dump_schema(tmp_path, monkeypatch):
    schema_dir = Path("generated")
    schema_dir.mkdir(exist_ok=True)
    schema_file = schema_dir / "quadoa_schema.json"
    schema_file.write_text(json.dumps({"wavefront": {"fields": []}}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "anamorph_fit", "dump-schema"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wavefront" in result.stdout
