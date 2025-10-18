"""Public interface for the anamorph_fit package."""

from .fit import fit_aberrations, fit_from_file
from .io import parse_quadoa_export

__all__ = ["fit_aberrations", "fit_from_file", "parse_quadoa_export"]
__version__ = "0.1.0"


def hello_world() -> str:
    """Return a simple readiness indicator."""
    return "anamorph_fit ready"
