from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "all_articles.csv"
REPORT = ROOT / "outputs" / "research" / "dataset_preparation_report.json"


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_dates(series: pd.Series) -> pd.Series:
    # ISOT has inconsistent date formats; format='mixed' is supported by pandas 2.x.
    try:
        return pd.to_datetime(series.astype(str), errors="coerce", format="mixed", utc=True)
    except (TypeError, ValueError):
        return pd.to_datetime(series.astype(str), errors="coerce", utc=True)


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def digest(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def read_source(path: Path, label: int, source: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}. Copy the original {path.name} into data/raw/.")
    df = pd.read_csv(path, low_memory=False)
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    body_col = lower_map.get("text")
    title_col = lower_map.get("title")
    date_col = lower_map.get("date")
    if body_col is None and title_col is None:
        raise ValueError(f"{path.name} must contain a 'text' and/or 'title' column. Found: {list(df.columns)}")
    body = df[body_col].map(clean_text) if body_col else pd.Series([""] * len(df), index=df.index)
    title = df[title_col].map(clean_text) if title_col else pd.Series([""] * len(df), index=df.index)
    text = (title.where(title.eq(""), title + "\n\n") + body).str.strip()
    dates = parse_dates(df[date_col]) if date_col else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    return pd.DataFrame({"text": text, "label": label, "source_dataset": source,
                         "publication_date": dates, "title": title})


def main() -> None:
    fake = read_source(RAW / "Fake.csv", 0, "ISOT_Fake.csv")
    true = read_source(RAW / "True.csv", 1, "ISOT_True.csv")
    df = pd.concat([fake, true], ignore_index=True)
    before = len(df)
    df["text"] = df["text"].map(clean_text)
    short_empty_removed = int((df["text"].str.len() < 20).sum())
    df = df[df["text"].str.len() >= 20].copy()
    df["article_hash"] = df["text"].map(digest)

    # If identical normalized text has conflicting labels, remove all rows in that group.
    conflicts = df.groupby("article_hash")["label"].nunique()
    conflict_hashes = set(conflicts[conflicts > 1].index)
    conflict_count = int(df["article_hash"].isin(conflict_hashes).sum())
    df = df[~df["article_hash"].isin(conflict_hashes)].copy()
    duplicate_count = int(df.duplicated("article_hash").sum())
    df = df.drop_duplicates("article_hash", keep="first").copy()
    df["publication_date"] = pd.to_datetime(df["publication_date"], utc=True, errors="coerce")
    df = df[["text", "label", "source_dataset", "publication_date", "title", "article_hash"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    report = {
        "source_files": {"Fake.csv_rows": len(fake), "True.csv_rows": len(true)},
        "rows_before_filtering": before,
        "rows_short_or_empty_removed": short_empty_removed,
        "conflicting_duplicate_rows_removed": conflict_count,
        "exact_duplicate_rows_removed_after_conflict_filter": duplicate_count,
        "rows_written": len(df),
        "label_counts": {str(int(k)): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "rows_with_publication_date": int(df["publication_date"].notna().sum()),
        "date_parse_rate_by_source": {
            src: round(float(pd.to_datetime(group["publication_date"], errors="coerce", utc=True).notna().mean()), 6)
            for src, group in df.groupby("source_dataset")
        },
        "output": str(OUT.relative_to(ROOT)),
        "duplicate_policy": "exact normalized text SHA-256; conflicting labels removed; near duplicates not detected",
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nPrepared dataset: {OUT}")


if __name__ == "__main__":
    main()
