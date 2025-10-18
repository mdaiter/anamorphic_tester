"""CLI smoke tests."""

import json
import subprocess
import sys

import pytest


def test_cli_json(tmp_path):
    pytest.importorskip("pandas")

    path = tmp_path / "d.csv"
    path.write_text("x,y,wavefront\n0,0,0\n", encoding="utf-8")

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
    assert "column_norms" in payload
    assert "condition_number" in payload


def test_cli_export_json(tmp_path):
    pytest.importorskip("pandas")

    path = tmp_path / "d.csv"
    path.write_text("x,y,wavefront\n0,0,0\n", encoding="utf-8")
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
    path.write_text("x,y,wavefront\n0,0,0\n", encoding="utf-8")
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
