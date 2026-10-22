from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "all_articles.csv"
OUT = ROOT / "data" / "processed" / "research_splits"
SEED = 42


def normalized_hash(text: str) -> str:
    return hashlib.sha256(" ".join(str(text).lower().split()).encode("utf-8")).hexdigest()


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in df["label"].value_counts().sort_index().items()}


def save_split(df: pd.DataFrame, directory: Path, name: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    df.to_csv(path, index=False)
    d = pd.to_datetime(df["publication_date"], errors="coerce", utc=True)
    return {"file": str(path.relative_to(ROOT)), "rows": len(df), "label_counts": label_counts(df),
            "date_min": None if d.notna().sum() == 0 else str(d.min()),
            "date_max": None if d.notna().sum() == 0 else str(d.max()),
            "article_hashes": set(df["article_hash"].astype(str))}


def clean_data() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run: python -m backend.training.prepare_dataset")
    df = pd.read_csv(INPUT, low_memory=False)
    required = {"text", "label", "article_hash", "publication_date"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {sorted(required)}")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0=Fake News and 1=True News.")
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce", utc=True, format="mixed")
    return df



def temporal_split_by_date(dated: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.10):
    """Split on whole publication-date groups so one calendar date never crosses sets."""
    dated = dated.dropna(subset=["publication_date"]).sort_values(["publication_date", "article_hash"], kind="mergesort").reset_index(drop=True)
    date_counts = dated.groupby("publication_date", sort=True).size()
    dates = list(date_counts.index)
    if len(dates) < 3:
        raise ValueError("Temporal protocol needs at least three distinct publication dates.")
    cumulative = np.cumsum(date_counts.to_numpy())
    val_start_idx = next((i + 1 for i, count in enumerate(cumulative) if count >= int(np.ceil(len(dated) * train_fraction))), 1)
    test_target = int(np.ceil(len(dated) * (train_fraction + validation_fraction)))
    test_start_idx = next((i + 1 for i, count in enumerate(cumulative) if count >= test_target), val_start_idx + 1)
    val_start_idx = min(max(val_start_idx, 1), len(dates) - 2)
    test_start_idx = min(max(test_start_idx, val_start_idx + 1), len(dates) - 1)
    validation_start = pd.Timestamp(dates[val_start_idx])
    test_start = pd.Timestamp(dates[test_start_idx])
    train = dated[dated["publication_date"] < validation_start].copy()
    validation = dated[(dated["publication_date"] >= validation_start) & (dated["publication_date"] < test_start)].copy()
    test = dated[dated["publication_date"] >= test_start].copy()
    if any(x.empty for x in (train, validation, test)):
        raise ValueError("Date-grouped temporal split produced an empty partition; inspect publication dates.")
    return train, validation, test, validation_start, test_start


def main() -> None:
    df = clean_data()
    report = {"seed": SEED, "random": {}, "temporal": {}, "notes": [
        "Exact duplicate and conflicting-label checks must be performed by prepare_dataset.py before splitting.",
        "Temporal protocol excludes undated rows and sorts dated articles by publication date.",
        "A temporal split can show label prevalence drift; report each split's counts.",
    ]}

    train_val, test = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["label"])
    train, val = train_test_split(train_val, test_size=0.125, random_state=SEED, stratify=train_val["label"])
    random_dir = OUT / "random"
    random_meta = {name: save_split(frame, random_dir, name) for name, frame in [("train", train), ("validation", val), ("test", test)]}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = random_meta[a]["article_hashes"] & random_meta[b]["article_hashes"]
        if overlap:
            raise RuntimeError(f"Random split leakage: {len(overlap)} overlapping hashes between {a} and {b}.")
    report["random"] = {k: {kk: vv for kk, vv in v.items() if kk != "article_hashes"} for k, v in random_meta.items()}

    dated = df[df["publication_date"].notna()].copy()
    if len(dated) < 100 or dated["label"].nunique() != 2:
        raise ValueError(f"Not enough dated articles or both labels missing for temporal split: {len(dated)} rows.")
    t_train, t_val, t_test, validation_start, test_start = temporal_split_by_date(dated)
    temporal_dir = OUT / "temporal"
    temporal_meta = {name: save_split(frame, temporal_dir, name) for name, frame in [("train", t_train), ("validation", t_val), ("test", t_test)]}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = temporal_meta[a]["article_hashes"] & temporal_meta[b]["article_hashes"]
        if overlap:
            raise RuntimeError(f"Temporal split leakage: {len(overlap)} overlapping hashes between {a} and {b}.")
    report["temporal"] = {k: {kk: vv for kk, vv in v.items() if kk != "article_hashes"} for k, v in temporal_meta.items()}
    report["temporal"]["undated_rows_excluded"] = int(df["publication_date"].isna().sum())
    report["temporal"]["dated_rows_used"] = int(len(dated))
    report["temporal"]["split_policy"] = "approximate 70/10/20 chronological split by whole publication dates; dates never cross partitions"
    report["temporal"]["validation_start_exclusive_for_train"] = str(validation_start)
    report["temporal"]["test_start"] = str(test_start)
    report["random"]["split_policy"] = "stratified 70% train; 10% validation; 20% test"
    report_path = OUT / "split_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"split_report": str(report_path.relative_to(ROOT)), "random": report["random"], "temporal": report["temporal"]}, indent=2))


if __name__ == "__main__":
    main()
