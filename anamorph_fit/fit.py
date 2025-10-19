"""Least-squares fitting for the third-order anamorphic aberrations (Eqs. 8-32–8-51)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Sequence

import math

import numpy as np
from scipy.linalg import lstsq

from .basis import ABERRATION_NAMES, basis_terms_full
from .io import parse_quadoa_export

__all__ = ["FitResult", "fit_aberrations_full", "fit_aberrations", "fit_from_file"]

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FitResult:
    """Container for aberration fit results."""

    names: list[str]
    values: np.ndarray
    rms_error: float
    metrics: dict[str, float]


def fit_aberrations_full(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
) -> FitResult:
    """Solve the 20-term aberration system defined by Eqs. (8-32)–(8-51)."""

    x, y, w = _prepare_grids(X, Y, W)

    terms = basis_terms_full(x, y)
    design = np.column_stack([term.ravel() for term in terms.values()])
    target = w.ravel()

    finite_rows = np.all(np.isfinite(design), axis=1)
    finite_target = np.isfinite(target)
    valid = finite_rows & finite_target
    if not np.all(valid):
        dropped = int((~valid).sum())
        LOGGER.warning("Dropping %d non-finite samples before fitting", dropped)
        design = design[valid, :]
        target = target[valid]

    if design.size == 0 or target.size == 0:
        raise ValueError("No finite samples available for fitting")

    # Eq (8-32)–(8-51): coefficients recovered via linear least squares
    coeffs, *_ = lstsq(design, target, cond=None, check_finite=True, lapack_driver="gelsd")
    residual = target - design @ coeffs
    rms = float(np.sqrt(np.mean(residual**2)))

    metrics = _compute_fit_metrics(terms, coeffs, rms)

    return FitResult(
        names=list(ABERRATION_NAMES),
        values=coeffs,
        rms_error=rms,
        metrics=metrics,
    )


# Backwards-compatible alias for previous API name.
def fit_aberrations(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
) -> FitResult:
    """Alias of :func:`fit_aberrations_full` for legacy imports."""

    return fit_aberrations_full(X, Y, W)


def fit_from_file(path: str | Path) -> FitResult:
    """Fit the Eq. (8-32)–(8-51) coefficients from a Quadoa export."""

    pupil = parse_quadoa_export(path)
    X, Y = np.meshgrid(pupil.x, pupil.y, indexing="xy")
    return fit_aberrations_full(X, Y, pupil.wavefront)


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


def _compute_fit_metrics(terms: dict[str, np.ndarray], coeffs: np.ndarray, rms: float) -> dict[str, float]:
    """Derive anisotropy metrics from the fitted coefficients.

    The groupings follow Eq. (8-32) – (8-44) for x/y terms and Eq. (8-48) – (8-51)
    for skew-ray aberrations.
    """

    names = list(terms.keys())
    a_terms = [(name, coeffs[idx]) for idx, name in enumerate(names) if name.startswith("A")]
    b_terms = [(name, coeffs[idx]) for idx, name in enumerate(names) if name.startswith("B")]
    c_terms = [(name, coeffs[idx]) for idx, name in enumerate(names) if name.startswith("C")]

    def _rms_for(group: list[tuple[str, float]]) -> float:
        if not group:
            return 0.0
        wave = sum(coeff * terms[name] for name, coeff in group)
        return float(np.sqrt(np.mean(wave**2)))

    rms_x = _rms_for(a_terms)
    rms_y = _rms_for(b_terms)
    rms_ratio = rms_x / rms_y if rms_y else float("inf")

    idx_map = {name: coeffs[idx] for idx, name in enumerate(names)}
    coma_ratio = float(abs(idx_map.get("A4_coma_x", 0.0)) / abs(idx_map.get("B3_coma_y", 1e-12)))

    skew_strength = math.sqrt(sum(value**2 for _, value in c_terms))
    skew_ratio = skew_strength / rms if rms else 0.0

    return {
        "rms_x": rms_x,
        "rms_y": rms_y,
        "rms_ratio_xy": rms_ratio,
        "coma_ratio_ax_bx": coma_ratio,
        "skew_relative_strength": skew_ratio,
    }
