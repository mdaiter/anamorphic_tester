"""Input/output utilities for anamorph_fit."""

from __future__ import annotations

import json
import logging
import math
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
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = dict(arbitrary_types_allowed=True)

    @field_validator("x", "y", "wavefront", mode="before")
    @classmethod
    def _coerce_array(cls, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value, dtype=float)


def parse_quadoa_export(
    source: str | Path | dict | None = None,
    *,
    lens_id: str | None = None,
    field_angle: float | None = None,
    base_url: str | None = None,
) -> PupilData:
    """Return a :class:`PupilData` instance sourced from Quadoa exports or API.

    The logic follows the wavefront interface described in
    ``docs/quadoa_api/wavefront_export.html``.  When ``lens_id`` and
    ``field_angle`` are provided the :class:`~quadoa.client.QuadoaClient`
    is used to request the live endpoint; otherwise ``source`` may point to a
    JSON/CSV export on disk.
    """

    from quadoa.client import QuadoaClient

    if lens_id is not None:
        if field_angle is None:
            raise ValueError("`field_angle` must be provided when `lens_id` is used")
        client = QuadoaClient(base_url)
        data = client.fetch_wavefront(lens_id, float(field_angle))
    elif isinstance(source, dict):
        data = QuadoaClient._normalise_wavefront_payload(source)
    elif source is not None:
        path = Path(source)
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            data = QuadoaClient._normalise_wavefront_payload(raw)
        else:
            pupil = _parse_csv(path)
            data = {
                "x": pupil.x,
                "y": pupil.y,
                "wavefront": pupil.wavefront,
                "wavelength_nm": pupil.wavelength_nm,
                "units": pupil.units,
                "metadata": pupil.metadata,
            }
    else:
        raise ValueError("Either `source` or (`lens_id`, `field_angle`) must be provided")

    metrics = _compute_anamorphic_metrics(data["x"], data["y"], data["wavefront"])
    metadata = dict(data.get("metadata", {}))
    metadata["anamorphic_metrics"] = metrics

    LOGGER = logging.getLogger(__name__)
    LOGGER.info(
        "Computed anamorphic metrics: ratio=%.3f skew=%.3f deg",
        metrics["anamorphic_ratio"],
        math.degrees(metrics["principal_axis_skew"]),
    )

    return PupilData(
        x=np.asarray(data["x"], dtype=float),
        y=np.asarray(data["y"], dtype=float),
        wavefront=np.asarray(data["wavefront"], dtype=float),
        wavelength_nm=float(data["wavelength_nm"]),
        units=str(data["units"]),
        metadata=metadata,
    )


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


def _compute_anamorphic_metrics(x: np.ndarray, y: np.ndarray, wavefront: np.ndarray) -> dict[str, float]:
    """Compute simple anisotropy metrics used by the CLI."""

    x_range = float(np.max(x) - np.min(x))
    y_range = float(np.max(y) - np.min(y))
    anamorphic_ratio = x_range / y_range if y_range else float("inf")

    if min(wavefront.shape) < 2:
        mean_dx = mean_dy = 0.0
    else:
        edge_order = 1 if min(wavefront.shape) <= 2 else 2
        grad_y, grad_x = np.gradient(wavefront, y, x, edge_order=edge_order)
        mean_dx = float(np.mean(grad_x))
        mean_dy = float(np.mean(grad_y))
    principal_axis_skew = math.atan2(mean_dy, mean_dx)

    rms = float(np.sqrt(np.mean(wavefront**2)))

    return {
        "anamorphic_ratio": anamorphic_ratio,
        "principal_axis_skew": principal_axis_skew,
        "wavefront_rms": rms,
    }
