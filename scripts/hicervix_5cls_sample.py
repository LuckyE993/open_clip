#   要点

#   - 新增 --counts（对所有 split 生效）以及 --train-counts / --val-counts / --test-counts（单独覆盖）
#   - 支持 -1 表示该类全量抽取
#   - 保留 --seed（默认 42）
#   - 支持 --output-dir 指定输出目录

#   示例

#   python scripts/hicervix_5cls_sample.py \
#     --train-counts "ASC-US=500,ASC-H=500,LSIL=500,HSIL=500,NILM=-1" \
#     --val-counts "ASC-US=100,ASC-H=100,LSIL=100,HSIL=100,NILM=100" \
#     --test-counts "ASC-US=100,ASC-H=100,LSIL=100,HSIL=100,NILM=100" \
#     --seed 42

#   对所有 split 使用同一套数量：

#   python scripts/hicervix_5cls_sample.py \
#     --counts "ASC-US=500,ASC-H=500,LSIL=500,HSIL=500,NILM=-1" \
#     --seed 42

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

TRAIN_FILE = "../datasets/hicervix/train.csv"
VAL_FILE = "../datasets/hicervix/val.csv"
TEST_FILE = "../datasets/hicervix/test.csv"
CLASSES = ["ASC-US", "ASC-H", "LSIL", "HSIL", "NILM"]
TRAIN_N = 80
VAL_N = 20
TEST_N = 20
SEED = 42

REQUIRED_COLS = {"image_name", "class_name", "level_1"}


def load_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {path}: {sorted(missing)}")
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # NILM is defined as all samples with level_1 == "negative"
    out["target_class"] = None
    is_negative = out["level_1"].astype(str).str.lower() == "negative"
    out.loc[is_negative, "target_class"] = "NILM"

    # Non-negative: keep only specified abnormal classes
    keep_abnormal = out["class_name"].isin(["ASC-US", "ASC-H", "LSIL", "HSIL"])
    out.loc[~is_negative & keep_abnormal, "target_class"] = out.loc[
        ~is_negative & keep_abnormal, "class_name"
    ]

    return out


def parse_counts_spec(spec: str) -> Dict[str, int]:
    mapping = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(
                "Invalid counts spec. Expected CLASS=NUM pairs, e.g. "
                "'ASC-US=80,ASC-H=80,LSIL=80,HSIL=80,NILM=80'"
            )
        name, value = part.split("=", 1)
        name = name.strip()
        if name not in CLASSES:
            raise SystemExit(f"Unknown class in counts spec: {name}")
        try:
            num = int(value)
        except ValueError:
            raise SystemExit(f"Invalid count for {name}: {value}")
        if num < -1:
            raise SystemExit(f"Count for {name} must be -1 or >= 0, got {num}")
        mapping[name] = num

    missing = [cls for cls in CLASSES if cls not in mapping]
    if missing:
        raise SystemExit("Counts spec missing classes: " + ", ".join(missing))

    return mapping


def resolve_counts(spec: Optional[str], default_n: int) -> Dict[str, int]:
    if spec is None:
        return {cls: default_n for cls in CLASSES}
    return parse_counts_spec(spec)


def sample_split(df: pd.DataFrame, need_by_class: Dict[str, int], seed: int) -> pd.DataFrame:
    filtered = df[df["target_class"].isin(CLASSES)].copy()

    counts = (
        filtered["target_class"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    )
    insufficient = {}
    for cls in CLASSES:
        need = need_by_class[cls]
        if need >= 0 and counts[cls] < need:
            insufficient[cls] = counts[cls]
    if insufficient:
        msg = "Insufficient samples for classes: " + ", ".join(
            f"{k} have {v}, need {need_by_class[k]}" for k, v in insufficient.items()
        )
        raise SystemExit(msg)

    rows = []
    for cls in CLASSES:
        sub = filtered[filtered["target_class"] == cls]
        need = need_by_class[cls]
        if need == -1:
            picked = sub.copy()
        else:
            picked = sub.sample(n=need, random_state=seed)
        rows.append(picked.reset_index(drop=True))

    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Sample hicervix 5-class CSV splits with per-class counts."
    )
    parser.add_argument("--train-file", default=TRAIN_FILE)
    parser.add_argument("--val-file", default=VAL_FILE)
    parser.add_argument("--test-file", default=TEST_FILE)
    parser.add_argument(
        "--counts",
        default=None,
        help=(
            "Per-class counts for all splits, "
            "e.g. 'ASC-US=80,ASC-H=80,LSIL=80,HSIL=80,NILM=80' (-1 for all)."
        ),
    )
    parser.add_argument(
        "--train-counts",
        default=None,
        help="Per-class counts for train split, overrides --counts.",
    )
    parser.add_argument(
        "--val-counts",
        default=None,
        help="Per-class counts for val split, overrides --counts.",
    )
    parser.add_argument(
        "--test-counts",
        default=None,
        help="Per-class counts for test split, overrides --counts.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default="./outputs")
    args = parser.parse_args()

    train_df = build_target(load_one(args.train_file))
    val_df = build_target(load_one(args.val_file))
    test_df = build_target(load_one(args.test_file))

    train_counts = resolve_counts(args.train_counts or args.counts, TRAIN_N)
    val_counts = resolve_counts(args.val_counts or args.counts, VAL_N)
    test_counts = resolve_counts(args.test_counts or args.counts, TEST_N)

    train_out = sample_split(train_df, train_counts, args.seed)
    val_out = sample_split(val_df, val_counts, args.seed)
    test_out = sample_split(test_df, test_counts, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "hicervix_5cls_train.csv"
    val_path = output_dir / "hicervix_5cls_val.csv"
    test_path = output_dir / "hicervix_5cls_test.csv"
    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    test_out.to_csv(test_path, index=False)

    # Report counts
    train_counts = (
        train_out["target_class"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    )
    val_counts = (
        val_out["target_class"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    )
    test_counts = (
        test_out["target_class"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    )
    print("train_counts")
    print(train_counts.to_string())
    print("\nval_counts")
    print(val_counts.to_string())
    print("\ntest_counts")
    print(test_counts.to_string())
    print(
        f"\ntrain_rows {len(train_out)} val_rows {len(val_out)} test_rows {len(test_out)}"
    )

    # Sanity check for NILM
    nilm_bad = train_out[train_out["target_class"] == "NILM"]["level_1"].str.lower().ne("negative").sum()
    nilm_bad += val_out[val_out["target_class"] == "NILM"]["level_1"].str.lower().ne("negative").sum()
    nilm_bad += test_out[test_out["target_class"] == "NILM"]["level_1"].str.lower().ne("negative").sum()
    if nilm_bad:
        raise SystemExit("NILM sanity check failed: found non-negative rows")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
