"""Tests for I/O helpers."""

import json

import numpy as np
import pytest

from anamorph_fit.io import PupilData, parse_quadoa_export


def test_parse_json(tmp_path):
    sample = {
        "pupil": {
            "x": [-1, 0, 1],
            "y": [-1, 0, 1],
            "wavefront": [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        },
        "meta": {"wavelength_nm": 550, "units": "waves"},
    }
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(sample))

    data = parse_quadoa_export(path)

    assert isinstance(data, PupilData)
    assert np.all((data.x >= -1) & (data.x <= 1))
    assert np.all((data.y >= -1) & (data.y <= 1))
    assert data.wavefront.shape == (3, 3)
    np.testing.assert_allclose(data.wavefront, 0.0)
    assert data.wavelength_nm == 550
    assert data.units == "waves"


def test_parse_csv(tmp_path):
    pd = pytest.importorskip("pandas")
    xs = np.array([0.0, 5.0, 10.0])
    ys = np.array([0.0, 5.0, 10.0])
    X, Y = np.meshgrid(xs, ys)
    wavefront = X + Y

    df = pd.DataFrame(
        {
            "x": X.ravel(),
            "y": Y.ravel(),
            "wavefront": wavefront.ravel(),
        }
    )
    path = tmp_path / "wf.csv"
    df.to_csv(path, index=False)

    data = parse_quadoa_export(path)

    np.testing.assert_allclose(data.x, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(data.y, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(data.wavefront, wavefront)
    assert data.units == "waves"
    assert data.wavelength_nm == 550.0


def test_parse_json_defaults(tmp_path):
    sample = {
        "pupil": {
            "x": [-1, 0, 1],
            "y": [-1, 0, 2],
            "wavefront": [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
        }
    }
    path = tmp_path / "wf_defaults.json"
    path.write_text(json.dumps(sample))

    data = parse_quadoa_export(path)

    assert data.wavelength_nm == 550.0
    assert data.units == "waves"
    assert data.wavefront.shape == (3, 3)


def test_micrometer_conversion(tmp_path):
    sample = {
        "pupil": {
            "x": [0, 1],
            "y": [0, 1],
            "wavefront": [
                [0.55, 0.0],
                [0.0, 0.55],
            ],
        },
        "meta": {"units": "micrometers"},
    }
    path = tmp_path / "wf_microns.json"
    path.write_text(json.dumps(sample))

    data = parse_quadoa_export(path)

    np.testing.assert_allclose(data.wavefront, [[1.0, 0.0], [0.0, 1.0]])
    assert data.units == "waves"
    assert data.wavelength_nm == 550.0
