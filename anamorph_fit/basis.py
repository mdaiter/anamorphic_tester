"""Third-order anamorphic aberration basis (Eqs. 8-32 – 8-51)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Tuple

import numpy as np

# Basis definitions: (name, equation reference, physical meaning, polynomial lambda).
# Each comment cites the originating equation and aberration interpretation.
BASIS_DEFINITIONS: list[Tuple[str, str, str, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = [
    # Eq (8-32): D_{Iₓ} piston – constant offset for the x-relative skew optical system.
    ("A1_piston_x", "Eq (8-32)", "piston (x-RSOS)", lambda x, y: np.ones_like(x)),
    # Eq (8-32): D_{Iₓ} tiltₓ – linear x-tilt from the entrance pupil.
    ("A2_tilt_x", "Eq (8-32)", "tilt in x", lambda x, y: x),
    # Eq (8-33): D_{IIₓ} defocusₓ – parabolic x-defocus.
    ("A3_defocus_x", "Eq (8-33)", "defocus in x", lambda x, y: x**2),
    # Eq (8-34): D_{IIIₓ} comaₓ – third-order x-coma.
    ("A4_coma_x", "Eq (8-34)", "coma in x", lambda x, y: x**3),
    # Eq (8-35): D_{IVₓ} cross-coma x²y – asymmetric coma coupling.
    ("A5_cross_coma_xxy", "Eq (8-35)", "cross-coma x²y", lambda x, y: x**2 * y),
    # Eq (8-35): D_{IVₓ} astigmatism x³y – higher-order x-astigmatism.
    ("A6_astig_x3y", "Eq (8-35)", "astigmatism x³y", lambda x, y: x**3 * y),
    # Eq (8-36): D_{Vₓ} sphericalₓ – fourth-order x-spherical aberration.
    ("A7_spherical_x", "Eq (8-36)", "spherical in x", lambda x, y: x**4),
    # Eq (8-36): D_{Vₓ} field curvatureₓ – fourth-order field curvature coupling.
    ("A8_field_curvature_x", "Eq (8-36)", "field curvature x", lambda x, y: x**4 * y),
    # Eq (8-40): D_{Iᵧ} tiltᵧ – linear y-tilt of the y-RSOS.
    ("B1_tilt_y", "Eq (8-40)", "tilt in y", lambda x, y: y),
    # Eq (8-41): D_{IIᵧ} defocusᵧ – parabolic y-defocus.
    ("B2_defocus_y", "Eq (8-41)", "defocus in y", lambda x, y: y**2),
    # Eq (8-42): D_{IIIᵧ} comaᵧ – third-order y-coma.
    ("B3_coma_y", "Eq (8-42)", "coma in y", lambda x, y: y**3),
    # Eq (8-43): D_{IVᵧ} cross-coma xy² – asymmetric coma coupling.
    ("B4_cross_coma_xyy", "Eq (8-43)", "cross-coma xy²", lambda x, y: x * y**2),
    # Eq (8-43): D_{IVᵧ} astigmatism xy³ – higher-order y-astigmatism.
    ("B5_astig_xy3", "Eq (8-43)", "astigmatism xy³", lambda x, y: x * y**3),
    # Eq (8-44): D_{Vᵧ} sphericalᵧ – fourth-order y-spherical aberration.
    ("B6_spherical_y", "Eq (8-44)", "spherical in y", lambda x, y: y**4),
    # Eq (8-44): D_{Vᵧ} field curvatureᵧ – fourth-order field curvature coupling.
    ("B7_field_curvature_y", "Eq (8-44)", "field curvature y", lambda x, y: x * y**4),
    # Eq (8-44): D_{Vᵧ} toroidal coupling – balanced x²y² term.
    ("B8_toroidal_xy", "Eq (8-44)", "toroidal coupling", lambda x, y: x**2 * y**2),
    # Eq (8-48): skew oblique astigmatism – xy(x² + y²).
    ("C1_skew_oblique_astig", "Eq (8-48)", "skew oblique astigmatism", lambda x, y: x * y * (x**2 + y**2)),
    # Eq (8-49): skew distortion – x³y − xy³.
    ("C2_skew_distortion", "Eq (8-49)", "skew distortion", lambda x, y: x**3 * y - x * y**3),
    # Eq (8-50): cross-field skew – x²y³ + x³y².
    ("C3_cross_field_skew", "Eq (8-50)", "cross-field skew", lambda x, y: x**2 * y**3 + x**3 * y**2),
    # Eq (8-51): higher-order skew – (x² + y²)²xy.
    ("C4_higher_order_skew", "Eq (8-51)", "higher-order skew", lambda x, y: (x**2 + y**2)**2 * x * y),
]

ABERRATION_NAMES: list[str] = [name for name, *_ in BASIS_DEFINITIONS]
EQUATION_MAP: Dict[str, str] = {name: eq for name, eq, *_ in BASIS_DEFINITIONS}
DESCRIPTION_MAP: Dict[str, str] = {name: desc for name, _, desc, _ in BASIS_DEFINITIONS}

__all__ = [
    "ABERRATION_NAMES",
    "EQUATION_MAP",
    "DESCRIPTION_MAP",
    "basis_terms_full",
    "basis_terms",
    "orthogonality_matrix",
]


def basis_terms_full(X: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray]:
    """Return the orthonormal 20-term basis derived from Eqs. (8-32)–(8-51)."""

    x, y = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(Y, dtype=float))
    if x.shape != y.shape:
        raise ValueError("X and Y must broadcast to the same shape")
    if x.size < len(ABERRATION_NAMES):
        raise ValueError(
            f"At least {len(ABERRATION_NAMES)} samples are required to orthogonalise "
            "the anamorphic basis."
        )

    raw = _raw_polynomials(x, y)
    ortho = _orthonormalise(raw)

    ordered = OrderedDict()
    for idx, (name, _, _, _) in enumerate(BASIS_DEFINITIONS):
        ordered[name] = ortho[idx].reshape(x.shape)
    return ordered


# Backwards-compatible alias for legacy imports.
basis_terms = basis_terms_full


def orthogonality_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute the Gram matrix of the orthonormal basis (Eqs. 8-32 – 8-51)."""

    terms = basis_terms_full(X, Y)
    matrix = np.column_stack([term.ravel() for term in terms.values()])
    n = matrix.shape[0]
    return (matrix.T @ matrix) / n


def _raw_polynomials(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the un-normalised polynomial fields referenced by §8.4."""

    return np.stack([expr(x, y) for _, _, _, expr in BASIS_DEFINITIONS], axis=0)


def _orthonormalise(terms: np.ndarray) -> np.ndarray:
    """Orthonormalise the raw monomials via QR decomposition."""

    n_terms = terms.shape[0]
    samples = terms.reshape(n_terms, -1).T  # (N, 20)
    q, _ = np.linalg.qr(samples)

    n = samples.shape[0]
    q_scaled = q * np.sqrt(n)

    raw_cols = samples.T
    ortho_cols = q_scaled.T
    signs = np.sign(np.sum(ortho_cols * raw_cols, axis=1))
    signs[signs == 0] = 1.0
    ortho_cols *= signs[:, None]

    return ortho_cols.reshape((n_terms,) + terms.shape[1:])
