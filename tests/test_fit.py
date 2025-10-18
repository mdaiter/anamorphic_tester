"""Validation tests for aberration fitting routines."""

import numpy as np
import pytest

from anamorph_fit.basis import _basis_terms
from anamorph_fit.fit import (
    ABERRATION_NAMES,
    fit_aberrations,
    fit_aberrations_numpy,
    fit_from_file,
)


@pytest.fixture(scope="module")
def grid():
    coords = np.linspace(-1.0, 1.0, 64)
    return np.meshgrid(coords, coords, indexing="xy")


@pytest.mark.parametrize("name", ABERRATION_NAMES)
def test_single_term(name, grid):
    X, Y = grid
    coeff_true = 0.123
    basis = _basis_terms(X, Y)[name]
    W = coeff_true * basis

    result = fit_aberrations(X, Y, W)
    coeffs = dict(zip(result.names, result.values))

    assert abs(coeffs[name] - coeff_true) < 1e-3
    for other, value in coeffs.items():
        if other != name:
            assert abs(value) < 1e-3


def test_random_combo(grid):
    X, Y = grid
    rng = np.random.default_rng(0)
    coeffs_true = rng.uniform(-0.1, 0.1, len(ABERRATION_NAMES))
    terms = _basis_terms(X, Y)
    stack = np.stack([terms[name] for name in ABERRATION_NAMES], axis=0)
    W = np.tensordot(coeffs_true, stack, axes=1)

    result = fit_aberrations(X, Y, W)

    assert np.allclose(result.values, coeffs_true, atol=1e-3)
    assert result.rms_error < 1e-6
    assert result.values.shape == (len(ABERRATION_NAMES),)
    assert result.column_norms.shape == (len(ABERRATION_NAMES),)
    assert result.condition_number > 0

    design = np.column_stack([terms[name].ravel() for name in ABERRATION_NAMES])
    residual = W.ravel() - design @ result.values
    reproj_rms = np.sqrt(np.mean(residual**2))
    assert reproj_rms < 1e-6


def test_end_to_end_csv(tmp_path):
    pytest.importorskip("pandas")

    x = np.linspace(-1.0, 1.0, 32)
    y = np.linspace(-1.0, 1.0, 32)
    X, Y = np.meshgrid(x, y, indexing="xy")
    W = 0.05 * X**3 - 0.03 * Y**2

    path = tmp_path / "wf.csv"
    np.savetxt(
        path,
        np.c_[X.ravel(), Y.ravel(), W.ravel()],
        delimiter=",",
        header="x,y,wavefront",
        comments="",
    )

    result = fit_from_file(path)
    coeffs = dict(zip(result.names, result.values))

    assert abs(coeffs["x-coma"] - 0.05) < 1e-3
    assert abs(coeffs["y-defocus"] + 0.03) < 1e-3
    assert result.rms_error < 1e-3


def test_backend_equivalence(grid):
    X, Y = grid
    W = 0.04 * X**4 - 0.02 * Y**3

    scipy_res = fit_aberrations(X, Y, W, backend="scipy")
    numpy_res = fit_aberrations(X, Y, W, backend="numpy")
    direct_numpy = fit_aberrations_numpy(X, Y, W)

    np.testing.assert_allclose(scipy_res.values, numpy_res.values, atol=1e-9)
    np.testing.assert_allclose(scipy_res.values, direct_numpy.values, atol=1e-9)
    assert scipy_res.condition_number == pytest.approx(numpy_res.condition_number)


def test_rectangular_grid():
    x = np.linspace(-1.0, 1.0, 128)
    y = np.linspace(-1.0, 1.0, 64)
    X, Y = np.meshgrid(x, y, indexing="xy")
    W = 0.02 * X**2 + 0.01 * Y

    result = fit_aberrations(X, Y, W)
    coeffs = dict(zip(result.names, result.values))

    assert coeffs["x-defocus"] == pytest.approx(0.02, abs=1e-3)
    assert coeffs["y-tilt"] == pytest.approx(0.01, abs=1e-3)


def test_ignore_nan_samples():
    coords = np.linspace(-1.0, 1.0, 32)
    X, Y = np.meshgrid(coords, coords, indexing="xy")
    W = 0.05 * X - 0.02 * Y**2
    W[10:14, 10:14] = np.nan

    result = fit_aberrations(X, Y, W)
    coeffs = dict(zip(result.names, result.values))

    assert abs(coeffs["x-tilt"] - 0.05) < 5e-3
    assert abs(coeffs["y-defocus"] + 0.02) < 5e-3


def test_basis_orthogonality():
    coords = np.linspace(-1.0, 1.0, 64)
    X, Y = np.meshgrid(coords, coords, indexing="xy")
    basis = _basis_terms(X, Y)
    matrix = np.column_stack([basis[name].ravel() for name in ABERRATION_NAMES])
    gram = matrix.T @ matrix
    diag = np.diag(gram)
    assert np.all(diag > 0)
    norm = np.sqrt(np.outer(diag, diag))
    corr = gram / norm
    off_mean = np.mean(np.abs(corr - np.eye(len(ABERRATION_NAMES))))
    assert off_mean < 0.2
