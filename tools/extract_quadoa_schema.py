
#!/usr/bin/env python3
"""Extract field metadata from Quadoa API documentation (HTML/RST)."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable

DOC_ROOT = Path(__file__).resolve().parents[1] / "docs" / "quadoa_api"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "generated" / "quadoa_schema.json"

# Regex patterns
ENDPOINT_PATTERN = re.compile(r"/(?:optical|wavefront|system|export|api)[\w/-]*")
JSON_KEY_PATTERN = re.compile(r'"([A-Za-z0-9_]+)"\s*[:=]')
BULLET_FIELD_PATTERN = re.compile(r"\s*[-*]\s*([A-Za-z0-9_]+)\s*\(([^)]+)\)")
FORMAT_PATTERN = re.compile(r"(json|xml|csv|yaml)", re.IGNORECASE)

def _iter_doc_files(root: Path) -> Iterable[Path]:
    for ext in ("*.html", "*.rst"):
        yield from root.rglob(ext)

def _normalise_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().strip("\"\'.,;()[]{}")
    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint
    return endpoint

def extract_schema() -> Dict[str, dict]:
    if not DOC_ROOT.exists():
        raise FileNotFoundError(f"Documentation directory not found: {DOC_ROOT}")

    schema: Dict[str, dict] = {}
    global_key = "__global__"

    for doc_path in _iter_doc_files(DOC_ROOT):
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        current_endpoint = None

        for idx, line in enumerate(lines, start=1):
            # Detect endpoints (e.g., /optical/wavefront/export)
            for match in ENDPOINT_PATTERN.findall(line):
                endpoint = _normalise_endpoint(match)
                entry = schema.setdefault(
                    endpoint,
                    {
                        "fields": defaultdict(lambda: {"types": set(), "sources": set()}),
                        "formats": set(),
                        "sources": set(),
                    },
                )
                entry["sources"].add(f"{doc_path.relative_to(DOC_ROOT)}:L{idx} # extracted from {doc_path.name}:L{idx}")
                current_endpoint = endpoint

            # Capture explicit field definitions in bullet lists, e.g. "- opd (float)"
            bullet = BULLET_FIELD_PATTERN.match(line)
            if bullet:
                field_name, field_type = bullet.groups()
                target = current_endpoint or global_key
                entry = schema.setdefault(
                    target,
                    {
                        "fields": defaultdict(lambda: {"types": set(), "sources": set()}),
                        "formats": set(),
                        "sources": set(),
                    },
                )
                field_info = entry["fields"][field_name]
                field_info["types"].add(field_type.strip())
                field_info["sources"].add(f"{doc_path.relative_to(DOC_ROOT)}:L{idx} # extracted from {doc_path.name}:L{idx}")

            # Capture JSON-style fields
            for json_match in JSON_KEY_PATTERN.finditer(line):
                field_name = json_match.group(1)
                target = current_endpoint or global_key
                entry = schema.setdefault(
                    target,
                    {
                        "fields": defaultdict(lambda: {"types": set(), "sources": set()}),
                        "formats": set(),
                        "sources": set(),
                    },
                )
                field_info = entry["fields"][field_name]
                field_info["sources"].add(f"{doc_path.relative_to(DOC_ROOT)}:L{idx} # extracted from {doc_path.name}:L{idx}")

            # Capture format hints (json/xml/etc.)
            for format_match in FORMAT_PATTERN.findall(line):
                fmt = format_match.lower()
                target = current_endpoint or global_key
                entry = schema.setdefault(
                    target,
                    {
                        "fields": defaultdict(lambda: {"types": set(), "sources": set()}),
                        "formats": set(),
                        "sources": set(),
                    },
                )
                entry["formats"].add(fmt)
                entry["sources"].add(f"{doc_path.relative_to(DOC_ROOT)}:L{idx} # extracted from {doc_path.name}:L{idx}")

    # Convert sets to sorted lists for JSON serialisation
    output: Dict[str, dict] = {}
    for endpoint, data in schema.items():
        fields_payload = []
        for field_name, info in sorted(data["fields"].items()):
            fields_payload.append(
                {
                    "name": field_name,
                    "types": sorted(info["types"]),
                    "sources": sorted(info["sources"]),
                }
            )

        formats = sorted(data["formats"])
        output[endpoint] = {
            "fields": fields_payload,
            "formats": formats,
            "sources": sorted(data["sources"]),
        }

    return output

def main() -> None:
    schema = extract_schema()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Schema extracted to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
