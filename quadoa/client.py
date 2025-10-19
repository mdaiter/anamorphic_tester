"""Quadoa client utilities.

This module implements a lightweight interface for the Quadoa optical API as
described in ``docs/quadoa_api/wavefront_export.html`` (section *Anamorphic
Wavefront Data*) and related REST documentation.  The extracted basis functions
and coefficients correspond to the third-order anamorphic aberrations of
§8.4 (Eqs. 8-32 – 8-51).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

import numpy as np
from scipy.interpolate import griddata

LOGGER = logging.getLogger(__name__)


class QuadoaClient:
    """Client for reading Quadoa optical exports or live REST API.

    Parameters
    ----------
    base_url:
        Base endpoint documented in ``docs/quadoa_api/rest_api_overview.rst``.
        If `base_url` is ``None`` or points to a local directory, the client
        loads JSON exports directly from disk (as produced by
        ``wavefront_export.html``).  When `base_url` starts with ``http://`` or
        ``https://`` the client performs REST calls to the live API.

    Notes
    -----
    The resulting dictionaries expose the canonical fields
    ``x``, ``y``, ``opd`` (wavefront optical path difference) and
    ``wavelength_nm`` which map directly onto the Eq. (8-32) – (8-51) basis.
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url
        if base_url is None:
            self._mode = "local"
            self._base_path = Path(".")
        elif base_url.startswith(("http://", "https://")):
            self._mode = "remote"
            self._base_path = None
        else:
            self._mode = "local"
            self._base_path = Path(base_url)

    # --------------------------------------------------------------------- API
    def fetch_wavefront(self, lens_id: str, field_angle: float) -> Dict[str, Any]:
        """Return normalized wavefront map (x, y, opd, wavelength).

        - OPD is normalised to *waves* per Eq. (8-32) before fitting.
        - Scattered data is resampled to a rectangular grid using
          ``scipy.interpolate.griddata``.

        Returns a dictionary with keys: ``x``, ``y``, ``wavefront``, ``units``,
        ``wavelength_nm`` and optional ``metadata``.
        """

        payload = self._load_wavefront_payload(lens_id, field_angle)
        normalised = self._normalise_wavefront_payload(payload)
        LOGGER.info(
            "Wavefront source=%s lens=%s field=%.1f shape=%s",
            self.base_url or "local",
            lens_id,
            field_angle,
            normalised["wavefront"].shape,
        )
        return normalised

    def fetch_system(self, lens_id: str) -> Dict[str, Any]:
        """Return optical system metadata (anamorphic ratio, surfaces, etc.)."""

        payload = self._load_system_payload(lens_id)
        metadata = payload.get("system", payload)
        LOGGER.debug("System metadata retrieved for %s: keys=%s", lens_id, list(metadata))
        return metadata

    # ----------------------------------------------------------------- helpers
    def _load_wavefront_payload(self, lens_id: str, field_angle: float) -> Dict[str, Any]:
        endpoint = f"/optical/wavefront/export"
        params = {"lens_id": lens_id, "field": f"{field_angle:.2f}"}

        if self._mode == "remote":
            url = urljoin(self.base_url.rstrip("/") + "/", endpoint.lstrip("/"))
            req = Request(f"{url}?{urlencode(params)}", headers={"Accept": "application/json"})
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

        base = self._base_path or Path(".")
        filename = f"{lens_id}_field{field_angle:.2f}.json"
        path = base / filename
        if not path.exists():
            # Fallback to generic naming convention
            path = base / "wavefront" / lens_id / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("metadata", {})["source"] = str(path)
        return data

    def _load_system_payload(self, lens_id: str) -> Dict[str, Any]:
        endpoint = f"/optical/system/{lens_id}"

        if self._mode == "remote":
            url = urljoin(self.base_url.rstrip("/") + "/", endpoint.lstrip("/"))
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

        base = self._base_path or Path(".")
        path = base / f"{lens_id}_system.json"
        if not path.exists():
            path = base / "system" / f"{lens_id}.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.setdefault("metadata", {})["source"] = str(path)
        return metadata

    # ------------------------------------------------------------ normalisers
    @staticmethod
    def _normalise_wavefront_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        meta = payload.get("metadata") or payload.get("meta", {})
        wavefront = payload.get("wavefront")
        if isinstance(wavefront, np.ndarray):
            return {
                "x": np.asarray(payload.get("x"), dtype=float),
                "y": np.asarray(payload.get("y"), dtype=float),
                "wavefront": np.asarray(wavefront, dtype=float),
                "wavelength_nm": float(meta.get("wavelength_nm", payload.get("wavelength_nm", 550.0))),
                "units": str(meta.get("units", payload.get("units", "waves"))),
                "metadata": dict(meta),
            }
        if wavefront is None and "pupil" in payload:
            wavefront = payload["pupil"]
        if wavefront is None:
            wavefront = payload

        x = np.asarray(wavefront.get("x", []), dtype=float)
        y = np.asarray(wavefront.get("y", []), dtype=float)
        opd = np.asarray(wavefront.get("opd") or wavefront.get("wavefront"), dtype=float)
        wavelength_nm = float(
            wavefront.get("wavelength_nm")
            or meta.get("wavelength_nm", 550.0)
        )
        units = str(
            wavefront.get("units")
            or meta.get("units", "waves")
        ).lower()

        if opd.size == 0:
            raise ValueError("Wavefront payload missing `opd`/`wavefront` data")

        if units in {"micrometer", "micrometers", "micron", "microns", "µm", "um"}:
            opd = opd / (wavelength_nm * 1e-3)
            units = "waves"

        if opd.ndim == 2:
            ny, nx = opd.shape
            if x.size == nx and y.size == ny:
                pass
            else:
                if x.size == 0:
                    x = np.linspace(-1.0, 1.0, nx)
                if y.size == 0:
                    y = np.linspace(-1.0, 1.0, ny)

        x_grid, y_grid, opd_grid = QuadoaClient._resample_grid(x, y, opd)

        metadata = dict(meta)
        metadata.setdefault("units", units)
        metadata.setdefault("wavelength_nm", wavelength_nm)

        return {
            "x": x_grid,
            "y": y_grid,
            "wavefront": opd_grid,
            "wavelength_nm": wavelength_nm,
            "units": units,
            "metadata": metadata,
        }

    # -------------------------------------------------------------- resampling
    @staticmethod
    def _resample_grid(x: np.ndarray, y: np.ndarray, opd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ensure the wavefront map is on a rectangular grid."""

        if opd.ndim == 2 and x.ndim == 1 and y.ndim == 1 and opd.shape == (y.size, x.size):
            return x, y, opd

        flat_x = np.asarray(x).ravel()
        flat_y = np.asarray(y).ravel()
        flat_opd = opd.reshape(-1)

        grid_x = np.linspace(flat_x.min(), flat_x.max(), int(math.sqrt(flat_x.size)) or 32)
        grid_y = np.linspace(flat_y.min(), flat_y.max(), int(math.sqrt(flat_y.size)) or 32)
        gx, gy = np.meshgrid(grid_x, grid_y, indexing="xy")
        gz = griddata(
            points=np.column_stack([flat_x, flat_y]),
            values=flat_opd,
            xi=(gx, gy),
            method="cubic",
            fill_value=np.nan,
        )

        nan_mask = np.isnan(gz)
        if nan_mask.any():
            gz[nan_mask] = griddata(
                np.column_stack([flat_x, flat_y]),
                flat_opd,
                (gx[nan_mask], gy[nan_mask]),
                method="nearest",
            )

        return grid_x, grid_y, gz
