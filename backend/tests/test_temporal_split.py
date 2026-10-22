import pandas as pd
from backend.training.create_research_splits import temporal_split_by_date


def test_temporal_split_keeps_each_date_in_one_partition():
    rows = []
    for day in range(1, 11):
        for j in range(3):
            rows.append({
                "text": f"example article text {day} {j}",
                "label": j % 2,
                "article_hash": f"{day}-{j}",
                "publication_date": pd.Timestamp(f"2020-01-{day:02d}", tz="UTC"),
            })
    df = pd.DataFrame(rows)
    train, val, test, val_start, test_start = temporal_split_by_date(df)
    assert train.publication_date.max() < val.publication_date.min()
    assert val.publication_date.max() < test.publication_date.min()
    assert val_start == val.publication_date.min()
    assert test_start == test.publication_date.min()
    assert len(train) + len(val) + len(test) == len(df)
