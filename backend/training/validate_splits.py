from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "processed" / "research_splits"


def date_range(path: Path) -> dict[str, str | None]:
    df = pd.read_csv(path, low_memory=False)
    if "publication_date" not in df:
        return {"min": None, "max": None}
    dates = pd.to_datetime(df["publication_date"], errors="coerce", utc=True, format="mixed").dropna()
    return {"min": None if dates.empty else str(dates.min()), "max": None if dates.empty else str(dates.max())}


def main() -> None:
    report: dict = {"passed": True, "protocols": {}}
    for protocol in ("random", "temporal"):
        group: dict[str, set[str]] = {}
        row_counts: dict[str, int] = {}
        paths = {}
        for name in ("train", "validation", "test"):
            path = SPLITS / protocol / f"{name}.csv"
            paths[name] = path
            if not path.exists():
                report["passed"] = False
                report.setdefault("errors", []).append(f"Missing {path.relative_to(ROOT)}")
                continue
            df = pd.read_csv(path, low_memory=False)
            if not {"text", "label", "article_hash"}.issubset(df.columns):
                report["passed"] = False
                report.setdefault("errors", []).append(f"Missing required columns in {path}")
            if "label" in df and not set(pd.to_numeric(df["label"], errors="coerce").dropna().astype(int).unique()).issubset({0, 1}):
                report["passed"] = False
                report.setdefault("errors", []).append(f"Invalid labels in {path}; expected 0=Fake, 1=True.")
            group[name] = set(df["article_hash"].astype(str)) if "article_hash" in df else set()
            row_counts[name] = len(df)
        overlaps = {}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlaps[f"{a}_vs_{b}"] = len(group.get(a, set()) & group.get(b, set()))
            if overlaps[f"{a}_vs_{b}"]:
                report["passed"] = False
        record = {"overlap_counts": overlaps, "rows": row_counts}
        if protocol == "temporal" and all(paths[n].exists() for n in ("train", "validation", "test")):
            ranges = {name: date_range(paths[name]) for name in ("train", "validation", "test")}
            date_checks = {}
            for earlier, later in (("train", "validation"), ("validation", "test")):
                left = ranges[earlier]["max"]
                right = ranges[later]["min"]
                passed = bool(left and right and pd.Timestamp(left) < pd.Timestamp(right))
                date_checks[f"{earlier}_before_{later}"] = passed
                if not passed:
                    report["passed"] = False
            record["publication_date_ranges"] = ranges
            record["strict_date_order_checks"] = date_checks
        report["protocols"][protocol] = record
    out = ROOT / "outputs" / "research" / "split_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
