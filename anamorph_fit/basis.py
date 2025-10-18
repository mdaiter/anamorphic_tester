"""Aberration basis utilities for anamorphic optics fitting."""

from __future__ import annotations

import numpy as np

try:  # Optional acceleration hook
    from numba import njit as _njit  # pragma: no cover
except ImportError:  # pragma: no cover
    def _njit(*_args, **_kwargs):
        def decorator(func):
            return func

        if _args and callable(_args[0]) and not _kwargs:
            return _args[0]
        return decorator

ABERRATION_NAMES: list[str] = [
    "piston",
    "x-tilt",
    "y-tilt",
    "x-defocus",
    "y-defocus",
    "x-spherical",
    "y-spherical",
    "x-coma",
    "y-coma",
    "astig_x",
    "astig_y",
    "oblique_astig",
    "x-field",
    "y-field",
    "x-distortion",
    "y-distortion",
]

__all__ = ["ABERRATION_NAMES", "_basis_terms"]


def _basis_terms(X: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate the aberration basis terms on the provided grid.

    Parameters
    ----------
    X, Y
        Coordinate arrays. They are broadcast together to support meshgrid
        outputs as well as vector inputs.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of aberration name to its evaluated term with the same shape
        as the broadcasted input.
    """

    x, y = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(Y, dtype=float))

    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    x4 = x2 * x2
    y4 = y2 * y2
    xy = x * y
    x2y = x2 * y
    xy2 = x * y2
    x3y = x3 * y
    xy3 = x * y3
    x3y2 = x3 * y2
    x2y3 = x2 * y3
    x2y2 = x2 * y2

    return {
        "piston": np.ones_like(x, dtype=float),  # uniform phase shift
        "x-tilt": x,  # linear tilt about y axis
        "y-tilt": y,  # linear tilt about x axis
        "x-defocus": x2,  # quadratic curvature aligned with x
        "y-defocus": y2,  # quadratic curvature aligned with y
        "x-spherical": x4,  # fourth-order spherical component in x
        "y-spherical": y4,  # fourth-order spherical component in y
        "x-coma": x3,  # cubic coma tail in x
        "y-coma": y3,  # cubic coma tail in y
        "astig_x": x2y,  # astigmatism varying with y while bending in x
        "astig_y": xy2,  # astigmatism varying with x while bending in y
        "oblique_astig": 2.0 * xy,  # 45-degree astigmatic saddle
        "x-field": x2y2,  # symmetric fourth-order field curvature
        "y-field": x3y,  # higher-order field variation dominated by x
        "x-distortion": x2y3,  # fifth-order distortion weighted toward y
        "y-distortion": x3y2,  # fifth-order distortion weighted toward x
    }
