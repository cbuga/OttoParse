#!/usr/bin/env python
"""Manual test runner for OttoParse Phase 1.

Point it at a single file or a folder of PDFs/images and it prints a per-page
report and writes cleaned images next to the results.

Examples
--------
    # one file
    python run_ottoparse.py samples/scan.pdf

    # a whole folder
    python run_ottoparse.py samples/

    # accept any-script text layers (e.g. Latin test PDFs), custom output dir
    python run_ottoparse.py samples/ --any-script --out results/

    # also dump each record as JSON
    python run_ottoparse.py samples/doc.pdf --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ottoparse import Pipeline
from ottoparse.intake import DefaultRouter
from ottoparse.io import IMAGE_EXTENSIONS, PDF_EXTENSIONS

SUPPORTED = IMAGE_EXTENSIONS | PDF_EXTENSIONS


def collect_inputs(target: Path) -> list[Path]:
    """Return the list of supported files under a file or directory target."""
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(
            p for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED
        )
    raise FileNotFoundError(target)


def report_record(rec, dump_json: bool) -> None:
    """Print a human-readable summary of one page record."""
    p = rec.provenance
    sev = rec.max_severity.value if rec.max_severity else "clean"
    print(f"  page {p.page_index}: {p.source_path.value}  [{sev}]")
    print(f"    routing : {p.path_reason}")
    if rec.history:
        steps = ", ".join(f"{s.name}({s.strategy})" for s in rec.history)
        print(f"    steps   : {steps}")
    if rec.flags:
        for f in rec.flags:
            meas = f" = {f.measure}" if f.measure is not None else ""
            print(f"    flag    : {f.code} [{f.severity.value}]{meas} — {f.detail or ''}")
    if rec.text is not None:
        preview = rec.text[:120].replace("\n", " ")
        print(f"    text    : {preview}{'...' if len(rec.text) > 120 else ''}")
    if rec.image_path is not None:
        print(f"    cleaned : {rec.image_path}")
    if dump_json:
        print("    json    :", json.dumps(rec.to_dict(), ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual OttoParse Phase 1 runner.")
    parser.add_argument("target", type=Path, help="A file or a folder of files.")
    parser.add_argument("--out", type=Path, default=Path("results"),
                        help="Directory for cleaned images (default: results/).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Rasterization DPI for image-only PDF pages.")
    parser.add_argument("--any-script", action="store_true",
                        help="Accept text layers of any script (sets arabic ratio to 0).")
    parser.add_argument("--json", action="store_true",
                        help="Also print each record as JSON.")
    args = parser.parse_args(argv)

    try:
        inputs = collect_inputs(args.target)
    except FileNotFoundError:
        print(f"error: no such file or directory: {args.target}", file=sys.stderr)
        return 2

    if not inputs:
        print(f"no supported files found in {args.target}", file=sys.stderr)
        print(f"supported extensions: {sorted(SUPPORTED)}", file=sys.stderr)
        return 1

    router = DefaultRouter(min_arabic_ratio=0.0) if args.any_script else DefaultRouter()
    pipe = Pipeline(router=router, rasterize_dpi=args.dpi, write_images_to=args.out)

    total_pages = 0
    total_errors = 0
    for path in inputs:
        print(f"\n=== {path.name} ===")
        try:
            records = pipe.process(path)
        except Exception as exc:  # noqa: BLE001 - surface any hard failure per-file
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        for rec in records:
            report_record(rec, args.json)
            total_pages += 1
            total_errors += int(rec.has_errors)

    print(f"\nprocessed {len(inputs)} file(s), {total_pages} page(s), "
          f"{total_errors} page(s) with errors.")
    print(f"cleaned images written to: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
