# Hicervix CLIP Training Pipeline (JSONL to TSV + Config-Driven Train)

## Summary
Build a small, config-driven training setup around the existing OpenCLIP training CLI. The work adds:
- A JSONL -> TSV conversion script for the Hicervix dataset.
- A Python training launcher that reads YAML/JSON config and calls `python -m open_clip_train.main`.
- A project doc `tarin_hicervix.md` (spelling per user request) describing conversion, config, and training usage.

This keeps OpenCLIP's core training code unchanged and focuses on reproducible configuration and dataset adaptation.

## Goals
- Convert `outputs/hicervix_5cls_{train,val}_captions.jsonl` into OpenCLIP-compatible TSV with columns `filepath` and `title`.
- Use `description` from JSONL as the caption text.
- Provide a Python training script that loads YAML/JSON config and invokes the existing OpenCLIP CLI with adjustable parameters.
- Provide clear documentation for running conversion and training in a conda env named `re0312`.

## Non-Goals
- No changes to OpenCLIP core training code or dataset loaders.
- No new model architectures.
- No distributed training orchestration beyond what OpenCLIP already supports.

## Assumptions
- JSONL rows include `image_path` and `description` (verified).
- `image_path` values are directly accessible (no extra prefix required).
- Relative `image_path` values are resolved relative to the current working directory when training.
- Training uses model `ViT-L-14` and pretrained weights `openai` by default.

## Data Mapping
Input JSONL fields:
- `image_path` -> output column `filepath`
- `description` -> output column `title`

Output format:
- TSV (tab-separated), header: `filepath\ttitle`
- Default output path: if `--output` not provided, replace the input file extension with `.tsv` in the same directory.
  - Example: `outputs/hicervix_5cls_train_captions.jsonl` -> `outputs/hicervix_5cls_train_captions.tsv`
  - The doc will also show how to explicitly write to `outputs/hicervix_5cls_{train,val}.tsv`.

OpenCLIP training flags (CSV dataset):
- `--dataset-type csv`
- `--csv-separator "\t"`
- `--csv-img-key filepath`
- `--csv-caption-key title`

## Components

### 1) Dataset Conversion Script
**File:** `scripts/jsonl_to_tsv_clip.py` (name reflects TSV output)

Responsibilities:
- Read input JSONL file line-by-line.
- Extract `image_path` and `description`.
- Write TSV with header `filepath` and `title`.
- Support CLI args:
  - `--input` (required)
  - `--output` (optional; default derived from input)
  - `--img-key` (default `image_path`)
  - `--caption-key` (default `description`)
  - `--sep` (default `\t`)
  - `--skip-invalid` (default true)
  - `--fail-on-error` (default false)

Error handling:
- If a row is invalid JSON or missing required fields:
  - Default: skip and count it (`--skip-invalid`).
  - If `--fail-on-error` is set, abort on the first invalid row (overrides `--skip-invalid`).
  - If `--skip-invalid=false` (and `--fail-on-error` is not set), abort on the first invalid row.
- If `--output` points to a non-existent directory, create parent directories.
- Print a summary of total rows, written rows, and skipped rows.

### 2) Config-Driven Training Script
**File:** `scripts/train_hicervix.py`

Responsibilities:
- Load a YAML/JSON config file.
- Convert config keys into CLI flags for `open_clip_train.main`.
- Run training via `subprocess` and pass through exit codes.

YAML dependency:
- Use PyYAML if available.
- If PyYAML is not installed and a `.yaml/.yml` config is provided, print a clear error and exit non-zero.
- JSON configs always work with the standard library.

Config format:
- Flat key/value map only (no nested sections).
- Keys may be written with underscores or dashes; underscores are converted to dashes.
  - Example: `batch_size` -> `--batch-size`
  - Example: `csv_separator` -> `--csv-separator`
- Boolean values:
  - `true` -> include `--flag`
  - `false` -> omit `--flag`

Minimum required config keys:
- `train_data`
- `val_data`

Defaulted keys (optional in config):
- `model` (default `ViT-L-14`)
- `pretrained` (default `openai`)

Common optional keys:
- `batch_size`, `epochs`, `lr`, `wd`, `precision`, `workers`, `seed`
- `logs`, `name`, `save_frequency`, `save_most_recent`
- CSV-related flags (`dataset_type`, `csv_separator`, `csv_img_key`, `csv_caption_key`)

Separator handling:
- In config, `csv_separator` should be the literal tab character. The documentation will show both:
  - YAML: `csv_separator: "\t"` (parsed as a tab)
  - JSON: `"csv_separator": "\t"` (escaped to a tab)

Behavior:
- Print the final command before execution for reproducibility.
- Validate that required fields exist.
- Support `--dry-run` to print the command and exit 0 without executing.

### 3) User Documentation
**File:** `tarin_hicervix.md` (repo root)

Contents:
- Conda env activation (`re0312`).
- JSONL -> TSV conversion commands (with default output behavior and explicit outputs).
- Config example (YAML and JSON snippets).
- Training invocation using `scripts/train_hicervix.py --config ...`.
- Notes on adjusting parameters (batch size, epochs, LR, etc.).

## Interfaces & Integration Points
- Conversion script produces TSV files consumed by OpenCLIP CSV dataset loader.
- Training script only calls `python -m open_clip_train.main` with CLI flags.

## Testing & Verification
- Smoke test conversion script on first 1-2 lines and confirm output schema and row counts.
- Use `--dry-run` for the training script to verify the generated command.

## Open Questions
None.

## Risks
- If JSONL contains unexpected nulls or missing fields, rows are skipped or cause a hard fail depending on flags. This is acceptable and reported.
- File paths must be correct; no additional prefix is applied.
