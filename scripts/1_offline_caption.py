from __future__ import annotations

"""
Offline caption generator for HiCervix images (ModelScope OpenAI-compatible API).

Quick start:
  1) Create .env (see .env.example) with MODELSCOPE_API_KEY
  2) Edit configs/offline_caption.yaml for CSV/image paths and model params
  3) Run:
       python scripts/offline_caption.py --config configs/offline_caption.yaml

Dry run (no API calls, writes JSONL with status=dry_run):
  python scripts/offline_caption.py --config configs/offline_caption.yaml --limit 5 --dry-run

Notes:
  - image_mode:
      base64  -> embed local image as data URL (default)
      file_url -> use file:// URL
      path     -> pass local path string
  - image_url_prefix:
      If set, overrides local loading and builds URL as
      {image_url_prefix}/{image_name}
  - deterministic=true forces temperature=0, top_p=1, top_k=0
  - run.show_progress=true enables tqdm progress bar with ETA
"""

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from tqdm import tqdm

REQUIRED_COLS = {"image_name", "class_name", "level_1"}
DEFAULT_BASE_URL = "https://api-inference.modelscope.cn/v1"
STANDARD_PARAM_KEYS = {
    "temperature",
    "top_p",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "n",
    "logit_bias",
    "user",
    "response_format",
    "seed",
    "tools",
    "tool_choice",
    "logprobs",
    "top_logprobs",
}


def load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Config root must be a mapping")
    return data


def build_prompt(template: str, class_name: str) -> str:
    return template.format(class_name=class_name)


def normalize_params(params: Dict[str, Any], deterministic: bool) -> Dict[str, Any]:
    out = dict(params or {})
    if deterministic:
        out["temperature"] = 0.0
        out["top_p"] = 1.0
        out["top_k"] = 0
    return out


def build_image_url(image_path: Path, image_name: str, data_cfg: Dict[str, Any]) -> str:
    prefix = str(data_cfg.get("image_url_prefix") or "").strip()
    if prefix:
        return f"{prefix.rstrip('/')}/{image_name}"

    mode = str(data_cfg.get("image_mode") or "base64").lower()
    if mode == "file_url":
        return image_path.as_uri()
    if mode == "path":
        return str(image_path)

    payload = image_path.read_bytes()
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    b64 = base64.b64encode(payload).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_record(
    row: Dict[str, Any],
    image_path: str,
    prompt: str,
    description: Optional[str],
    model_name: str,
    params: Dict[str, Any],
    seed: int,
    source_csv: str,
    row_index: int,
    status: str,
    error: Optional[str],
) -> Dict[str, Any]:
    base = f"{row['image_name']}|{row_index}|{seed}"
    record_id = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    return {
        "id": record_id,
        "image_name": row["image_name"],
        "image_path": image_path,
        "class_name": row["class_name"],
        "level_1": row.get("level_1"),
        "prompt": prompt,
        "description": description,
        "model": model_name,
        "params": params,
        "seed": seed,
        "source_csv": source_csv,
        "row_index": row_index,
        "status": status,
        "error": error,
    }


def count_csv_rows(csv_path: Path, done: set[str]) -> tuple[int, int]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLS - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns in {csv_path}: {sorted(missing)}")

        total = 0
        remaining = 0
        for row in reader:
            total += 1
            if not done or row["image_name"] not in done:
                remaining += 1
    return total, remaining


def call_model(
    client: Any,
    model_name: str,
    prompt: str,
    image_url: str,
    params: Dict[str, Any],
    stream: bool,
    timeout: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]
    std_params: Dict[str, Any] = {}
    extra_body: Dict[str, Any] = {}
    for key, value in (params or {}).items():
        if key in STANDARD_PARAM_KEYS:
            std_params[key] = value
        else:
            extra_body[key] = value

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=stream,
        timeout=timeout,
        extra_body=extra_body or None,
        **std_params,
    )

    if not stream:
        return response.choices[0].message.content or ""

    parts = []
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            parts.append(content)
    return "".join(parts)


def run(config: Dict[str, Any], *, dry_run: bool = False, limit: Optional[int] = None) -> int:
    data_cfg = config["data"]
    output_cfg = config["output"]
    prompt_cfg = config["prompt"]
    model_cfg = config["model"]
    params_cfg = config.get("params", {})
    run_cfg = config.get("run", {})

    csv_path = Path(data_cfg["csv_path"])
    image_root = Path(data_cfg["image_root"])
    jsonl_path = Path(output_cfg["jsonl_path"])
    prompt_template = prompt_cfg["template"]

    deterministic = bool(config.get("deterministic", False))
    seed = int(config.get("seed", 42))
    params = normalize_params(params_cfg, deterministic=deterministic)

    stream = bool(run_cfg.get("stream", False))
    timeout = int(run_cfg.get("timeout", 60))
    max_retries = int(run_cfg.get("max_retries", 2))
    backoff = float(run_cfg.get("backoff", 1.5))
    resume = bool(run_cfg.get("resume", True))
    use_seed = bool(run_cfg.get("use_seed", True))
    show_progress = bool(run_cfg.get("show_progress", True))

    if use_seed and "seed" not in params:
        params["seed"] = seed

    client = None
    if not dry_run:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        base_url = os.environ.get("MODELSCOPE_BASE_URL", DEFAULT_BASE_URL)
        api_key = os.environ.get("MODELSCOPE_API_KEY")
        if not api_key:
            raise SystemExit("MODELSCOPE_API_KEY missing in environment/.env")
        client = OpenAI(base_url=base_url, api_key=api_key)

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if resume and jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = obj.get("image_name")
                if name:
                    done.add(name)

    processed = 0
    image_url_prefix = str(data_cfg.get("image_url_prefix") or "").strip()
    use_local = not image_url_prefix

    progress = None
    if show_progress:
        _, remaining = count_csv_rows(csv_path, done if resume else set())
        progress = tqdm(total=remaining, desc="caption", unit="img")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLS - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns in {csv_path}: {sorted(missing)}")

        for row_index, row in enumerate(reader):
            if limit is not None and processed >= limit:
                break

            image_name = row["image_name"]
            if resume and image_name in done:
                continue

            image_path = image_root / image_name
            prompt = build_prompt(prompt_template, row["class_name"])

            if use_local and not image_path.exists():
                record = build_record(
                    row=row,
                    image_path=str(image_path),
                    prompt=prompt,
                    description=None,
                    model_name=model_cfg["name"],
                    params=params,
                    seed=seed,
                    source_csv=str(csv_path),
                    row_index=row_index,
                    status="missing_file",
                    error="image not found",
                )
                with jsonl_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
                if progress:
                    progress.update(1)
                continue

            if dry_run:
                record = build_record(
                    row=row,
                    image_path=str(image_path),
                    prompt=prompt,
                    description=None,
                    model_name=model_cfg["name"],
                    params=params,
                    seed=seed,
                    source_csv=str(csv_path),
                    row_index=row_index,
                    status="dry_run",
                    error=None,
                )
                with jsonl_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
                if progress:
                    progress.update(1)
                continue

            image_url = build_image_url(image_path, image_name, data_cfg)

            attempt = 0
            last_err: Optional[str] = None
            while attempt <= max_retries:
                try:
                    desc = call_model(
                        client=client,
                        model_name=model_cfg["name"],
                        prompt=prompt,
                        image_url=image_url,
                        params=params,
                        stream=stream,
                        timeout=timeout,
                    )
                    record = build_record(
                        row=row,
                        image_path=str(image_path),
                        prompt=prompt,
                        description=desc,
                        model_name=model_cfg["name"],
                        params=params,
                        seed=seed,
                        source_csv=str(csv_path),
                        row_index=row_index,
                        status="ok",
                        error=None,
                    )
                    with jsonl_path.open("a", encoding="utf-8") as out:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    last_err = None
                    break
                except Exception as exc:  # pragma: no cover - API failures
                    last_err = str(exc)
                    attempt += 1
                    if attempt > max_retries:
                        record = build_record(
                            row=row,
                            image_path=str(image_path),
                            prompt=prompt,
                            description=None,
                            model_name=model_cfg["name"],
                            params=params,
                            seed=seed,
                            source_csv=str(csv_path),
                            row_index=row_index,
                            status="api_error",
                            error=last_err,
                        )
                        with jsonl_path.open("a", encoding="utf-8") as out:
                            out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        break
                    time.sleep(backoff**attempt)

            processed += 1
            if progress:
                progress.update(1)

    if progress:
        progress.close()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/offline_caption.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    return run(config, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
