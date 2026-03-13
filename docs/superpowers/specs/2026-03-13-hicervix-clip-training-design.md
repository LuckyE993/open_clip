# Hicervix CLIP Training Pipeline (JSONL to CSV + Config-Driven Train)

## Summary
Build a small, config-driven training setup around the existing OpenCLIP training CLI. The work adds:
- A JSONL -> TSV conversion script for the Hicervix dataset.
- A Python training launcher that reads YAML/JSON config and calls `python -m open_clip_train.main`.
- A project doc `tarin_hicervix.md` describing conversion, config, and training usage.

This keeps OpenCLIP's core training code unchanged and focuses on reproducible configuration and dataset adaptation.

## Goals
- Convert `outputs/hicervix_5cls_{train,val}_captions.jsonl` into OpenCLIP-compatible CSV/TSV with columns `filepath` and `title`.
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
- Training is done with model `ViT-L-14` and pretrained weights `openai`.

## Data Mapping
Input JSONL fields:
- `image_path` -> output column `filepath`
- `description` -> output column `title`

Output format:
- TSV (tab-separated), header: `filepath\ttitle`
- Default output locations:
  - `outputs/hicervix_5cls_train.tsv`
  - `outputs/hicervix_5cls_val.tsv`

OpenCLIP training flags (CSV dataset):
- `--dataset-type csv`
- `--csv-separator "\t"`
- `--csv-img-key filepath`
- `--csv-caption-key title`

## Components

### 1) Dataset Conversion Script
**File:** `scripts/jsonl_to_csv_clip.py`

Responsibilities:
- Read input JSONL file line-by-line.
- Extract `image_path` and `description`.
- Write TSV with header `filepath` and `title`.
- Support CLI args:
  - `--input` (required)
  - `--output` (required)
  - `--img-key` (default `image_path`)
  - `--caption-key` (default `description`)
  - `--sep` (default `\t`)
  - `--skip-invalid` (default true)

Error handling:
- If a row is missing a required field or is invalid JSON:
  - By default: skip and count it.
  - Optional: `--fail-on-error` to abort.
- Print a summary of total rows, written rows, and skipped rows.

### 2) Config-Driven Training Script
**File:** `scripts/train_hicervix.py`

Responsibilities:
- Load a YAML/JSON config file.
- Convert config keys into CLI flags for `open_clip_train.main`.
- Run training via `subprocess` and pass through exit codes.

Config format:
- Top-level object with keys mapping to CLI arguments.
- Supports nested sections for readability (example in doc), but flattened to CLI flags.

Minimum required config keys:
- `train_data`
- `val_data`
- `model` (default `ViT-L-14`)
- `pretrained` (default `openai`)

Common optional keys:
- `batch_size`, `epochs`, `lr`, `wd`, `precision`, `workers`, `seed`
- `logs`, `name`, `save_frequency`, `save_most_recent`
- CSV-related flags (`dataset_type`, `csv_separator`, `csv_img_key`, `csv_caption_key`)

Behavior:
- Print the final command before execution for reproducibility.
- Validate that required fields exist.
- Allow overriding config values via CLI (if requested later).

### 3) User Documentation
**File:** `tarin_hicervix.md`

Contents:
- Conda env activation (`re0312`).
- JSONL -> TSV conversion commands.
- Config example (YAML and JSON snippets).
- Training invocation using `scripts/train_hicervix.py --config ...`.
- Notes on adjusting parameters (batch size, epochs, LR, etc.).

## Interfaces & Integration Points
- Conversion script produces TSV files consumed by OpenCLIP CSV dataset loader.
- Training script only calls `python -m open_clip_train.main` with CLI flags.

## Testing & Verification
- Smoke test conversion script on first 1-2 lines and confirm output schema.
- Run training script with `--dry-run` (if implemented) or observe printed command.

## Open Questions
None.

## Risks
- If JSONL contains unexpected nulls or missing fields, rows are skipped. This is acceptable and reported.
- File paths must be correct; no additional prefix is applied.
