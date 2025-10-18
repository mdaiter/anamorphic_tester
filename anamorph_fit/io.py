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

    @field_validator("wavefront")
    @classmethod
    def _validate_shape(cls, wavefront: np.ndarray, info: Any) -> np.ndarray:
        x = info.data.get("x")
        y = info.data.get("y")
        if x is None or y is None:
            return wavefront
        expected = (np.size(np.unique(y)), np.size(np.unique(x)))
        if wavefront.shape != expected:
            raise ValueError(f"wavefront shape {wavefront.shape} expected {expected}")
        return wavefront


def parse_quadoa_export(path: str | Path) -> PupilData:
    """Parse a Quadoa export file (JSON or CSV) into normalized pupil data."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        pupil = payload.get("pupil", {})
        meta = payload.get("meta", {})
        x = np.asarray(pupil.get("x", []), dtype=float)
        y = np.asarray(pupil.get("y", []), dtype=float)
        wavefront = np.asarray(pupil.get("wavefront", []), dtype=float)
        wavelength = float(meta.get("wavelength_nm", 550.0))
        units = str(meta.get("units", "waves"))
    else:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - exercised in runtime usage
            raise ImportError(
                "pandas is required to parse CSV exports; please install pandas"
            ) from exc

        df = pd.read_csv(path)
        required = {"x", "y", "wavefront"}
        if not required.issubset(df.columns):
            missing = ", ".join(sorted(required - set(df.columns)))
            raise ValueError(f"CSV missing required columns: {missing}")
        pivot = df.pivot_table(
            values="wavefront", index="y", columns="x", sort=False
        ).sort_index(axis=0).sort_index(axis=1)
        x = pivot.columns.to_numpy(dtype=float)
        y = pivot.index.to_numpy(dtype=float)
        wavefront = pivot.to_numpy(dtype=float)
        wavelength = 550.0
        units = "waves"

    x_norm = _normalize_axis(x)
    y_norm = _normalize_axis(y)
    wavefront = np.asarray(wavefront, dtype=float)

    return PupilData(
        x=x_norm,
        y=y_norm,
        wavefront=wavefront,
        wavelength_nm=wavelength,
        units=units,
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
