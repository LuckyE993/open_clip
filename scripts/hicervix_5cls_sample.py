import sys
from pathlib import Path

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


def sample_split(df: pd.DataFrame, need: int) -> pd.DataFrame:
    filtered = df[df["target_class"].isin(CLASSES)].copy()

    counts = (
        filtered["target_class"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    )
    insufficient = counts[counts < need]
    if not insufficient.empty:
        msg = "Insufficient samples for classes: " + ", ".join(
            f"{k}={v}" for k, v in insufficient.items()
        )
        raise SystemExit(msg)

    rows = []
    for cls in CLASSES:
        sub = filtered[filtered["target_class"] == cls]
        sub = sub.sample(n=need, random_state=SEED).reset_index(drop=True)
        rows.append(sub)

    return pd.concat(rows, ignore_index=True)


def main():
    train_df = build_target(load_one(TRAIN_FILE))
    val_df = build_target(load_one(VAL_FILE))
    test_df = build_target(load_one(TEST_FILE))

    train_out = sample_split(train_df, TRAIN_N)
    val_out = sample_split(val_df, VAL_N)
    test_out = sample_split(test_df, TEST_N)

    train_path = Path("./outputs/hicervix_5cls_train.csv")
    val_path = Path("./outputs/hicervix_5cls_val.csv")
    test_path = Path("./outputs/hicervix_5cls_test.csv")
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
