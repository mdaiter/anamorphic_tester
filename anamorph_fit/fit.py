"""Least-squares fitting for anamorphic aberration coefficients."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
from scipy.linalg import lstsq

from .basis import ABERRATION_NAMES, _basis_terms
from .io import parse_quadoa_export

__all__ = ["FitResult", "fit_aberrations", "fit_from_file", "fit_aberrations_numpy"]

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FitResult:
    """Container for aberration fit results."""

    names: list[str]
    values: np.ndarray
    rms_error: float
    column_norms: np.ndarray
    condition_number: float


def fit_aberrations(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    *,
    backend: str = "scipy",
) -> FitResult:
    """Solve for aberration coefficients that best fit the provided wavefront.

    Parameters
    ----------
    X, Y
        Coordinate grids describing the sample locations.
    W
        Observed wavefront (or measurement) values.
    """

    x, y, w = _prepare_grids(X, Y, W)

    terms = _basis_terms(x, y)
    design = np.column_stack([terms[name].ravel() for name in ABERRATION_NAMES])
    target = w.ravel()

    finite_rows = np.all(np.isfinite(design), axis=1)
    finite_target = np.isfinite(target)
    valid = finite_rows & finite_target
    if not np.all(valid):
        dropped = int((~valid).sum())
        warnings.warn(f"Dropping {dropped} non-finite samples before fitting")
        LOGGER.warning("Dropping %d non-finite samples before fitting", dropped)
        design = design[valid, :]
        target = target[valid]
    if design.size == 0 or target.size == 0:
        raise ValueError("No finite samples available for fitting")

    LOGGER.debug(
        "Fitting aberrations using %s backend on %d samples", backend, design.shape[0]
    )

    column_norms = np.linalg.norm(design, axis=0)
    zero_mask = column_norms == 0.0
    if zero_mask.any():
        warnings.warn("Zero-norm basis column detected; results may be unstable")
        LOGGER.warning("Zero-norm basis columns detected at indices %s", np.where(zero_mask)[0])
        column_norms[zero_mask] = 1.0

    design_normalized = design / column_norms
    condition_number = float(np.linalg.cond(design_normalized))
    if condition_number > 1e5:
        warnings.warn(f"Ill-conditioned basis (cond={condition_number:.2e})")
        LOGGER.warning("Ill-conditioned basis detected (cond=%s)", condition_number)
    else:
        LOGGER.debug("Basis condition number: %.3e", condition_number)

    coeffs_scaled = _solve_coefficients(design_normalized, target, backend=backend)
    coeffs = coeffs_scaled / column_norms

    residual = target - design @ coeffs
    rms = float(np.linalg.norm(residual) / np.sqrt(residual.size))
    LOGGER.debug("Fit RMS error: %.6e", rms)

    return FitResult(
        names=list(ABERRATION_NAMES),
        values=coeffs,
        rms_error=rms,
        column_norms=column_norms,
        condition_number=condition_number,
    )


def fit_from_file(path: str | Path, *, backend: str = "scipy") -> FitResult:
    """Convenience wrapper that parses a Quadoa export and fits coefficients."""

    pupil = parse_quadoa_export(path)
    result = fit_aberrations(pupil.x, pupil.y, pupil.wavefront, backend=backend)
    return result


def fit_aberrations_numpy(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> FitResult:
    """Fit aberrations using the NumPy backend."""

    return fit_aberrations(X, Y, W, backend="numpy")


def _prepare_grids(
    X: np.ndarray | Sequence[float],
    Y: np.ndarray | Sequence[float],
    W: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize coordinate inputs to broadcastable grids."""

    x_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(Y, dtype=float)
    w_arr = np.asarray(W, dtype=float)

    if x_arr.ndim == 1 and y_arr.ndim == 1 and w_arr.ndim == 2:
        expected = (y_arr.size, x_arr.size)
        if w_arr.shape != expected:
            raise ValueError(
                f"wavefront shape {w_arr.shape} is not compatible with "
                f"x/y axes {(y_arr.size, x_arr.size)}"
            )
        x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing="xy")
        return x_grid, y_grid, w_arr

    x_grid, y_grid = np.broadcast_arrays(x_arr, y_arr)
    w_grid = np.broadcast_to(w_arr, x_grid.shape)
    return x_grid, y_grid, w_grid


def _solve_coefficients(design: np.ndarray, target: np.ndarray, *, backend: str) -> np.ndarray:
    """Dispatch to the requested least-squares backend."""

    if backend == "scipy":
        coeffs, *_ = lstsq(design, target, cond=None, check_finite=True, lapack_driver="gelsd")
    elif backend == "numpy":
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    else:
        raise ValueError("backend must be 'scipy' or 'numpy'")
    return coeffs
