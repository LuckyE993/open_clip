#!/usr/bin/env python3
"""Convert JSONL captions to OpenCLIP-compatible TSV.

Usage:
  python scripts/jsonl_to_tsv_clip.py --input outputs/hicervix_5cls_train_captions.jsonl
  python scripts/jsonl_to_tsv_clip.py --input outputs/hicervix_5cls_val_captions.jsonl
  python scripts/jsonl_to_tsv_clip.py --input INPUT.jsonl --output OUTPUT.tsv --skip-invalid

Expected JSONL fields (defaults):
- image_path -> filepath
- description -> title
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSONL captions to OpenCLIP-compatible TSV (filepath/title)."
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output TSV file (default: input path with .tsv extension)",
    )
    parser.add_argument(
        "--img-key",
        default="image_path",
        help="JSON key for image path (default: image_path)",
    )
    parser.add_argument(
        "--caption-key",
        default="description",
        help="JSON key for caption text (default: description)",
    )
    parser.add_argument(
        "--sep",
        default="\t",
        help="Column separator (default: tab)",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid rows instead of failing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on first invalid row (default behavior)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    return parser.parse_args()


def _sanitize_field(value: Any) -> str:
    text = str(value)
    # Replace tabs/newlines to keep TSV integrity
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _derive_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_suffix(".tsv")
    return input_path.with_name(input_path.name + ".tsv")


def _validate_flags(args: argparse.Namespace) -> None:
    if args.skip_invalid and args.strict:
        raise SystemExit("--skip-invalid and --strict are mutually exclusive")


def _extract_row(
    obj: dict, img_key: str, caption_key: str
) -> Tuple[str, str]:
    if img_key not in obj or caption_key not in obj:
        raise KeyError("Missing required keys")
    img_val = obj.get(img_key)
    cap_val = obj.get(caption_key)
    if img_val is None or cap_val is None:
        raise ValueError("Null value for required keys")
    img_text = _sanitize_field(img_val).strip()
    cap_text = _sanitize_field(cap_val).strip()
    if not img_text or not cap_text:
        raise ValueError("Empty value for required keys")
    return img_text, cap_text


def main() -> int:
    args = _parse_args()
    _validate_flags(args)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    output_path = Path(args.output) if args.output else _derive_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(
            f"Output file already exists: {output_path}. Use --overwrite to replace.",
            file=sys.stderr,
        )
        return 2

    tmp_path = Path(str(output_path) + ".tmp")

    total = 0
    written = 0
    skipped = 0

    try:
        with input_path.open("r", encoding="utf-8") as src, tmp_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as dst:
            header = f"filepath{args.sep}title\n"
            dst.write(header)

            for line_no, line in enumerate(src, start=1):
                if not line.strip():
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                    img_text, cap_text = _extract_row(
                        obj, args.img_key, args.caption_key
                    )
                    dst.write(f"{img_text}{args.sep}{cap_text}\n")
                    written += 1
                except Exception as exc:  # noqa: BLE001 - provide a clear message
                    if args.skip_invalid:
                        skipped += 1
                        print(
                            f"Skipping invalid row {line_no}: {exc}",
                            file=sys.stderr,
                        )
                        continue
                    # default/strict: fail fast
                    raise RuntimeError(f"Invalid row {line_no}: {exc}") from exc

        # Atomic replace to avoid partial outputs
        os.replace(tmp_path, output_path)
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Total rows: {total}")
    print(f"Written rows: {written}")
    print(f"Skipped rows: {skipped}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
