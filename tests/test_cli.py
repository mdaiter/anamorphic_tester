"""CLI smoke tests."""

import json
import subprocess
import sys

import numpy as np
import pytest
from pathlib import Path


def _write_flat_csv(path: Path) -> None:
    xs = np.linspace(-1.0, 1.0, 5)
    ys = np.linspace(-1.0, 1.0, 5)
    rows = ["x,y,wavefront"]
    for y in ys:
        for x in xs:
            rows.append(f"{x},{y},0.0")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_cli_json(tmp_path):
    pytest.importorskip("pandas")

    path = tmp_path / "d.csv"
    _write_flat_csv(path)

    result = subprocess.run(
        [sys.executable, "-m", "anamorph_fit", str(path), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert "names" in payload
    assert "values" in payload
    assert "rms_error" in payload


def test_cli_export_json(tmp_path):
    pytest.importorskip("pandas")

    path = tmp_path / "d.csv"
    _write_flat_csv(path)
    out = tmp_path / "result.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "anamorph_fit",
            str(path),
            "--export-json",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(out.read_text())
    assert "names" in data
    assert "values" in data


def test_cli_basis_save(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib.pyplot")

    path = tmp_path / "d.csv"
    _write_flat_csv(path)
    image = tmp_path / "residual.png"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "anamorph_fit",
            str(path),
            "--save-plot",
            str(image),
            "--basis",
            "heatmap",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert image.exists()
    basis_image = image.with_name(f"{image.stem}_basis{image.suffix}")
    assert basis_image.exists()
