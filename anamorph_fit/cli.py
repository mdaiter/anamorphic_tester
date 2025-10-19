"""Command-line interface entry points for anamorph_fit."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from .basis import ABERRATION_NAMES, basis_terms_full, orthogonality_matrix
from .fit import FitResult, fit_aberrations_full
from .io import PupilData, parse_quadoa_export
from quadoa.client import QuadoaClient

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")

    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "metrics":
        return _cmd_metrics(args)
    if args.command == "dump-schema":
        return _cmd_dump_schema(args)
    parser.error("Unknown command")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anamorph_fit",
        description="Analyze Quadoa wavefront exports using Eq. (8-32)–(8-51) basis.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Set logging level (debug, info, warning, error, critical).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Fetch a Quadoa wavefront and fit Eq. (8-32)–(8-51) coefficients",
    )
    analyze.add_argument("--lens", required=True, help="Lens identifier (Quadoa API)")
    analyze.add_argument("--field", type=float, required=True, help="Field angle in degrees")
    analyze.add_argument("--base-url", help="REST endpoint or local directory")
    analyze.add_argument("--export-json", type=Path, help="Write fit results to JSON file")
    analyze.add_argument("--plot", action="store_true", help="Display residual heatmap")
    analyze.add_argument("--save-plot", type=Path, help="Save residual heatmap image")
    analyze.add_argument(
        "--basis",
        choices=("none", "heatmap"),
        default="none",
        help="Render basis correlation heatmap",
    )

    metrics = subparsers.add_parser(
        "metrics",
        help="Compute anamorphic metrics for an exported wavefront",
    )
    metrics.add_argument("--input", type=Path, required=True, help="Path to Quadoa wavefront JSON/CSV")
    metrics.add_argument("--json", action="store_true", help="Print metrics as JSON")

    dump = subparsers.add_parser(
        "dump-schema",
        help="Print schema parsed from docs/quadoa_api",
    )

    return parser


def _result_to_json_ready(result: FitResult) -> dict[str, object]:
    return {
        "names": list(result.names),
        "values": np.asarray(result.values, dtype=float).tolist(),
        "rms_error": float(result.rms_error),
        "metrics": result.metrics,
    }


def _format_report(result: FitResult, source: str) -> str:
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
    lines.append(divider)
    lines.append("Metrics:")
    for key, value in result.metrics.items():
        lines.append(f"  {key}: {value:.6e}" if isinstance(value, float) else f"  {key}: {value}")
    return "\n".join(lines)


def _plot_residual(
    pupil: PupilData,
    result: FitResult,
    *,
    show: bool,
    save_path: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit("matplotlib is required for plotting") from exc

    X, Y = np.meshgrid(pupil.x, pupil.y, indexing="xy")
    basis = basis_terms_full(X, Y)
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
    pupil: PupilData,
    *,
    show: bool,
    save_path: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit("matplotlib is required for plotting") from exc

    X, Y = np.meshgrid(pupil.x, pupil.y, indexing="xy")
    normalized = orthogonality_matrix(X, Y)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(normalized, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Basis Correlation Heatmap")
    ax.set_xticks(range(len(ABERRATION_NAMES)))
    ax.set_yticks(range(len(ABERRATION_NAMES)))
    ax.set_xticklabels(ABERRATION_NAMES, rotation=90)
    ax.set_yticklabels(ABERRATION_NAMES)
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


def _cmd_analyze(args: argparse.Namespace) -> int:
    pupil = parse_quadoa_export(
        lens_id=args.lens,
        field_angle=args.field,
        base_url=args.base_url,
    )

    X, Y = np.meshgrid(pupil.x, pupil.y, indexing="xy")
    result = fit_aberrations_full(X, Y, pupil.wavefront)

    payload = _result_to_json_ready(result)
    payload["metadata"] = pupil.metadata

    print(_format_report(result, f"lens={args.lens} field={args.field:.2f}"))

    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.plot or args.save_plot:
        _plot_residual(pupil, result, show=args.plot, save_path=args.save_plot)

    if args.basis == "heatmap":
        basis_path = _basis_save_path(args.save_plot)
        _plot_basis_heatmap(pupil, show=args.plot or basis_path is None, save_path=basis_path)

    system = QuadoaClient(args.base_url).fetch_system(args.lens)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("System metadata surfaces=%s anamorphic=%s", system.get("surface_count"), system.get("anamorphic_ratio"))
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    pupil = parse_quadoa_export(args.input)
    X, Y = np.meshgrid(pupil.x, pupil.y, indexing="xy")
    result = fit_aberrations_full(X, Y, pupil.wavefront)

    metrics = dict(pupil.metadata.get("anamorphic_metrics", {}))
    metrics.update(result.metrics)

    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print("Anamorphic metrics (Eq. 8-32–8-51):")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6e}" if isinstance(value, float) else f"  {key}: {value}")
    return 0


def _cmd_dump_schema(args: argparse.Namespace) -> int:
    schema_path = Path(__file__).resolve().parents[1] / "generated" / "quadoa_schema.json"
    if not schema_path.exists():
        raise SystemExit("Schema not found; run tools/extract_quadoa_schema.py")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    print(json.dumps(schema, indent=2, sort_keys=True))
    return 0
