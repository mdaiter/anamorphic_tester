"""Command-line interface entry points for anamorph_fit."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from .basis import ABERRATION_NAMES, _basis_terms
from .fit import FitResult, fit_from_file
from .io import parse_quadoa_export

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
    elif args.log_level:
        logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    try:
        result = fit_from_file(args.file, backend=args.backend)
    except Exception as exc:  # pragma: no cover - defensive
        parser.exit(2, f"error: failed to fit '{args.file}': {exc}\n")

    payload = _result_to_json_ready(result)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(result, args.file))

    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    show_plots = args.plot
    if args.plot or args.save_plot:
        _plot_residual(
            args.file,
            result,
            show=show_plots,
            save_path=args.save_plot,
        )

    if args.basis == "heatmap":
        basis_save = _basis_save_path(args.save_plot)
        basis_show = args.plot or (args.save_plot is None)
        _plot_basis_heatmap(
            args.file,
            result,
            show=basis_show,
            save_path=basis_save,
        )

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anamorph_fit",
        description="Fit anamorphic aberration coefficients from Quadoa exports.",
    )
    parser.add_argument("file", type=Path, help="Path to Quadoa CSV or JSON export.")
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit fit results as JSON.",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Write fit results JSON to the specified file.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Display residual heatmap.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        help="Save the residual heatmap to an image file.",
    )
    parser.add_argument(
        "--basis",
        choices=("none", "heatmap"),
        default="none",
        help="Generate additional basis diagnostics (e.g. 'heatmap').",
    )
    parser.add_argument(
        "--backend",
        choices=("scipy", "numpy"),
        default="scipy",
        help="Least-squares backend to use.",
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error", "critical"),
        help="Set logging level (overrides default INFO).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser


def _result_to_json_ready(result: FitResult) -> dict[str, object]:
    return {
        "names": list(result.names),
        "values": np.asarray(result.values, dtype=float).tolist(),
        "rms_error": float(result.rms_error),
        "column_norms": np.asarray(result.column_norms, dtype=float).tolist(),
        "condition_number": float(result.condition_number),
    }


def _format_report(result: FitResult, source: Path) -> str:
    header = f"{'Aberration':<20}{'Coefficient':>14}"
    divider = "-" * len(header)
    lines = [
        f"Source: {source}",
        header,
        divider,
    ]
    for name, value in zip(result.names, result.values):
        lines.append(f"{name:<20}{value:>14.6e}")
    lines.extend(
        [
            divider,
            f"{'RMS error':<20}{result.rms_error:>14.6e}",
        ]
    )
    return "\n".join(lines)


def _plot_residual(
    path: Path,
    result: FitResult,
    *,
    show: bool,
    save_path: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit("matplotlib is required for plotting") from exc

    pupil = parse_quadoa_export(path)
    X, Y = np.meshgrid(pupil.x, pupil.y, indexing="xy")
    basis = _basis_terms(X, Y)
    stack = np.stack([basis[name] for name in result.names], axis=0)
    modeled = np.tensordot(result.values, stack, axes=1)
    residual = pupil.wavefront - modeled

    fig, ax = plt.subplots()
    extent = [pupil.x.min(), pupil.x.max(), pupil.y.min(), pupil.y.max()]
    im = ax.imshow(
        residual,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        aspect="equal",
    )
    ax.set_title(f"Residual ({pupil.units})")
    ax.set_xlabel("X (normalized)")
    ax.set_ylabel("Y (normalized)")
    fig.colorbar(im, ax=ax, label=f"Residual ({pupil.units})")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_basis_heatmap(
    path: Path,
    result: FitResult,
    *,
    show: bool,
    save_path: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit("matplotlib is required for plotting") from exc

    pupil = parse_quadoa_export(path)
    X, Y = np.meshgrid(pupil.x, pupil.y, indexing="xy")
    basis = _basis_terms(X, Y)
    matrix = np.column_stack([basis[name].ravel() for name in ABERRATION_NAMES])
    gram = matrix.T @ matrix
    diag = np.sqrt(np.clip(np.diag(gram), a_min=1e-12, a_max=None))
    normalized = (matrix / diag).T @ (matrix / diag)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(normalized, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Basis Correlation Heatmap")
    ax.set_xticks(range(len(result.names)))
    ax.set_yticks(range(len(result.names)))
    ax.set_xticklabels(result.names, rotation=90)
    ax.set_yticklabels(result.names)
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _basis_save_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    suffix = path.suffix or ".png"
    stem = f"{path.stem}_basis"
    return path.with_name(stem + suffix)
