from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = ROOT / "data" / "processed" / "research_splits"
DEFAULT_APPROVED = ROOT / "data" / "processed" / "approved_feedback.csv"


def norm(text: object) -> str:
    return " ".join(str(text or "").lower().split())


def digest(text: object) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, low_memory=False)


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), errors="coerce", utc=True, format="mixed")


def main() -> None:
    p = argparse.ArgumentParser(description="Merge expert-approved labels into training only, with leakage guards.")
    p.add_argument("--protocol", choices=["random", "temporal"], required=True)
    p.add_argument("--approved-csv", default=str(DEFAULT_APPROVED))
    p.add_argument("--output-csv", default=None)
    args = p.parse_args()

    split_dir = SPLIT_ROOT / args.protocol
    train_path, val_path, test_path = (split_dir / f"{name}.csv" for name in ("train", "validation", "test"))
    base_train, validation, test = read_csv(train_path), read_csv(val_path), read_csv(test_path)
    approved = read_csv(Path(args.approved_csv))
    if approved.empty:
        raise ValueError("Approved feedback file contains no rows.")

    # API export uses `article`; other exports may use `text`.
    if "text" not in approved and "article" in approved:
        approved = approved.rename(columns={"article": "text"})
    required = {"text", "label"}
    if not required.issubset(approved.columns):
        raise ValueError(f"Approved CSV must have columns {sorted(required)} or `article` and `label`.")
    approved["text"] = approved["text"].fillna("").astype(str).str.strip()
    approved["label"] = pd.to_numeric(approved["label"], errors="coerce")
    before = len(approved)
    approved = approved[approved["text"].str.len().ge(20) & approved["label"].isin([0, 1])].copy()
    approved["label"] = approved["label"].astype(int)
    valid_rows = len(approved)
    if "article_hash" not in approved:
        approved["article_hash"] = approved["text"].map(digest)
    else:
        missing_hash = approved["article_hash"].isna() | approved["article_hash"].astype(str).str.strip().eq("")
        approved.loc[missing_hash, "article_hash"] = approved.loc[missing_hash, "text"].map(digest)
    approved["article_hash"] = approved["article_hash"].astype(str)

    val_test_hashes = set(validation["article_hash"].astype(str)) | set(test["article_hash"].astype(str))
    base_train_hashes = set(base_train["article_hash"].astype(str))
    overlaps_val_test = approved["article_hash"].isin(val_test_hashes)
    overlaps_base_train = approved["article_hash"].isin(base_train_hashes)
    approved = approved[~overlaps_val_test & ~overlaps_base_train].copy()
    excluded_test_validation = int(overlaps_val_test.sum())
    excluded_train_duplicate = int((~overlaps_val_test & overlaps_base_train).sum())

    temporal_excluded = 0
    temporal_cutoff = None
    if args.protocol == "temporal":
        if "publication_date" not in approved or "label_available_at" not in approved:
            raise ValueError("Temporal protocol requires publication_date and label_available_at in approved CSV; missing dates must not be guessed.")
        val_dates = parse_dates(validation["publication_date"])
        val_dates = val_dates.dropna()
        if val_dates.empty:
            raise ValueError("Temporal validation split has no usable publication dates.")
        cutoff = val_dates.min()
        temporal_cutoff = cutoff.isoformat()
        published = parse_dates(approved["publication_date"])
        labeled = parse_dates(approved["label_available_at"])
        eligible = published.notna() & labeled.notna() & (published < cutoff) & (labeled <= cutoff)
        temporal_excluded = int((~eligible).sum())
        approved = approved[eligible].copy()

    if approved.empty:
        raise ValueError("No approved rows remain after leakage/date checks. This is a safe stop, not a failure; collect eligible reviewed examples or run without feedback.")

    # Use only schema needed for model training; preserve extra provenance in an audit report.
    add = approved[["text", "label", "article_hash"]].copy()
    add["source_dataset"] = "expert_approved_feedback"
    if "publication_date" in approved:
        add["publication_date"] = approved["publication_date"]
    else:
        add["publication_date"] = pd.NaT
    if "title" not in base_train:
        base_train["title"] = ""
    add["title"] = ""
    combined = pd.concat([base_train, add], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("article_hash", keep="first").reset_index(drop=True)

    out_path = Path(args.output_csv) if args.output_csv else SPLIT_ROOT / args.protocol / "train_with_approved_feedback.csv"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    report = {
        "protocol": args.protocol,
        "approved_input_rows": before,
        "valid_text_and_binary_label_rows": valid_rows,
        "excluded_overlap_with_validation_or_test": excluded_test_validation,
        "excluded_duplicate_of_base_training": excluded_train_duplicate,
        "excluded_temporal_ineligible_or_missing_dates": temporal_excluded,
        "temporal_label_cutoff": temporal_cutoff,
        "approved_rows_added": len(add),
        "base_training_rows": len(base_train),
        "combined_training_rows": len(combined),
        "validation_and_test_unchanged": True,
        "output_csv": str(out_path),
        "guardrails": ["pending/rejected records must not be exported as approved", "test/validation article hashes excluded", "temporal rows require publication and label-availability dates before the cutoff"],
    }
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
