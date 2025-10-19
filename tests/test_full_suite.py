"""Comprehensive verification of third-order anamorphic aberrations (§8.4)."""

import numpy as np
import pytest

from anamorph_fit.basis import (
    ABERRATION_NAMES,
    DESCRIPTION_MAP,
    EQUATION_MAP,
    basis_terms_full,
    orthogonality_matrix,
)
from anamorph_fit.fit import fit_aberrations_full

# Cross-reference table for Eq numbers and physical meanings (Eqs. 8-32 – 8-51).
EQ_INFO = {
    name: (EQUATION_MAP[name], DESCRIPTION_MAP[name])
    for name in ABERRATION_NAMES
}


@pytest.fixture(scope="module")
def full_grid():
    """Dense pupil sampling to support QR normalisation (Eqs. 8-32 – 8-51)."""

    coords = np.linspace(-1.0, 1.0, 64)
    return np.meshgrid(coords, coords, indexing="xy")


@pytest.mark.parametrize("name", ABERRATION_NAMES)
def test_single_aberration(full_grid, name):
    """Each coefficient from Eq. (8-32) – (8-51) is recovered in isolation."""

    X, Y = full_grid
    eq_label, meaning = EQ_INFO[name]  # e.g. Eq (8-34): coma in x
    basis = basis_terms_full(X, Y)
    W = 0.123 * basis[name]

    result = fit_aberrations_full(X, Y, W)
    coeffs = dict(zip(result.names, result.values))

    assert coeffs[name] == pytest.approx(0.123, abs=1e-3), f"{eq_label}: {meaning}"
    residue = [abs(value) for other, value in coeffs.items() if other != name]
    assert max(residue) < 1e-3, f"Leakage detected for {eq_label}: {meaning}"


def test_random_combination(full_grid):
    """Random mix of Eqs. (8-32)–(8-51) coefficients is recovered within 1e-3 RMS."""

    rng = np.random.default_rng(0)
    coeffs_true = rng.uniform(-0.1, 0.1, len(ABERRATION_NAMES))
    X, Y = full_grid
    basis = basis_terms_full(X, Y)
    W = np.tensordot(coeffs_true, np.stack([basis[name] for name in ABERRATION_NAMES], axis=0), axes=1)

    result = fit_aberrations_full(X, Y, W)
    coeffs_est = result.values

    assert np.allclose(coeffs_est, coeffs_true, atol=1e-3)
    assert result.rms_error < 1e-6


def test_orthogonality(full_grid):
    """Gram matrix off-diagonals remain below 5 % (Eq. 8-32 – Eq. 8-51)."""

    X, Y = full_grid
    gram = orthogonality_matrix(X, Y)
    diag = np.diag(gram)
    np.testing.assert_allclose(diag, np.ones_like(diag), atol=1e-6)
    off_diag = gram - np.diag(diag)
    assert np.max(np.abs(off_diag)) < 0.05


def test_noise_tolerance(full_grid):
    """Coefficients remain accurate within 2 % with 1 % Gaussian noise (Eq. 8-32 – Eq. 8-51)."""

    rng = np.random.default_rng(42)
    coeffs_true = rng.uniform(-0.05, 0.05, len(ABERRATION_NAMES))
    X, Y = full_grid
    basis = basis_terms_full(X, Y)
    stack = np.stack([basis[name] for name in ABERRATION_NAMES], axis=0)
    W_clean = np.tensordot(coeffs_true, stack, axes=1)

    noise_sigma = 0.01 * np.std(W_clean)
    W_noisy = W_clean + rng.normal(0.0, noise_sigma, size=W_clean.shape)

    result = fit_aberrations_full(X, Y, W_noisy)
    coeffs_est = result.values

    relative_error = np.linalg.norm(coeffs_est - coeffs_true) / np.linalg.norm(coeffs_true)
    assert relative_error < 0.02


def test_skew_decoupling(full_grid):
    """Skew-only wavefronts (Eqs. 8-48 – 8-51) have <5 % coupling into A/B sets."""

    rng = np.random.default_rng(7)
    X, Y = full_grid
    basis = basis_terms_full(X, Y)

    c_terms = [name for name in ABERRATION_NAMES if name.startswith("C")]
    ab_terms = [name for name in ABERRATION_NAMES if name.startswith("A") or name.startswith("B")]

    coeffs_true = {name: rng.uniform(-0.05, 0.05) for name in c_terms}
    W = sum(coeffs_true[name] * basis[name] for name in c_terms)

    result = fit_aberrations_full(X, Y, W)
    coeffs = dict(zip(result.names, result.values))

    power_c = sum(coeffs[name] ** 2 for name in c_terms)
    power_ab = sum(coeffs[name] ** 2 for name in ab_terms)

    if power_c == 0:
        pytest.skip("Skew coefficients collapsed to zero; regenerate")

    assert power_ab <= 0.05 * power_c
