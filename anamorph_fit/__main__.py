"""Module entry point for ``python -m anamorph_fit``."""

from .cli import main


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover - exercised via module execution
    run()

