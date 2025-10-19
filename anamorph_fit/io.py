"""Input/output utilities for anamorph_fit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

__all__ = ["PupilData", "parse_quadoa_export"]


class PupilData(BaseModel):
    """Normalized pupil description for fitting routines."""

    x: np.ndarray = Field(...)
    y: np.ndarray = Field(...)
    wavefront: np.ndarray = Field(...)
    wavelength_nm: float = 550.0
    units: str = "waves"

    model_config = dict(arbitrary_types_allowed=True)

    @field_validator("x", "y", "wavefront", mode="before")
    @classmethod
    def _coerce_array(cls, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value, dtype=float)


def parse_quadoa_export(path: str | Path) -> PupilData:
    """Parse a Quadoa export file (JSON or CSV) into normalized pupil data."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        return _parse_json(path)
    return _parse_csv(path)


def _parse_json(path: Path) -> PupilData:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pupil = payload.get("pupil", {})
    meta = payload.get("meta") or payload.get("metadata", {})

    x = np.asarray(pupil.get("x", []), dtype=float)
    y = np.asarray(pupil.get("y", []), dtype=float)
    wavefront = np.asarray(pupil.get("wavefront", []), dtype=float)
    if wavefront.ndim != 2:
        raise ValueError("wavefront must be a 2D array in JSON exports")

    ny, nx = wavefront.shape

    if x.size != nx:
        if x.size > 1:
            x = np.linspace(x.min(), x.max(), nx)
        else:
            x = np.linspace(-1.0, 1.0, nx)
    if y.size != ny:
        if y.size > 1:
            y = np.linspace(y.min(), y.max(), ny)
        else:
            y = np.linspace(-1.0, 1.0, ny)

    x_norm = _normalize_axis(x)
    y_norm = _normalize_axis(y)

    units_raw = str(meta.get("units", "waves")).lower()
    wavelength_nm = float(meta.get("wavelength_nm", 550.0))
    if units_raw in {"micrometer", "micrometers", "micron", "microns"}:
        wavefront = wavefront / (wavelength_nm * 1e-3)
        units = "waves"
    else:
        units = units_raw or "waves"

    return PupilData(
        x=x_norm,
        y=y_norm,
        wavefront=wavefront,
        wavelength_nm=wavelength_nm,
        units=units,
    )


def _parse_csv(path: Path) -> PupilData:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pandas is required to parse CSV exports; install with `poetry install --extras csv`."
        ) from exc

    df = pd.read_csv(path)
    required = {"x", "y", "wavefront"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"CSV missing required columns: {missing}")

    pivot = (
        df.pivot_table(values="wavefront", index="y", columns="x", sort=False)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    wavefront = pivot.to_numpy(dtype=float)

    x_norm = _normalize_axis(x)
    y_norm = _normalize_axis(y)

    return PupilData(
        x=x_norm,
        y=y_norm,
        wavefront=wavefront,
        wavelength_nm=550.0,
        units="waves",
    )


def _normalize_axis(values: np.ndarray) -> np.ndarray:
    """Scale a 1-D coordinate array into [-1, 1]."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    vmin = values.min()
    vmax = values.max()
    if np.isclose(vmax, vmin):
        return np.zeros_like(values)
    scale = 2.0 / (vmax - vmin)
    return (values - vmin) * scale - 1.0
