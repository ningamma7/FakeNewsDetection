from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reproducible TF-IDF + Logistic Regression baseline on the chosen training split.")
    parser.add_argument("--protocol", choices=["random", "temporal"], default="random")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--output-model", default=None)
    parser.add_argument("--max-features", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_path = Path(args.train_csv) if args.train_csv else ROOT / "data" / "processed" / "research_splits" / args.protocol / "train.csv"
    model_path = Path(args.output_model) if args.output_model else ROOT / "backend" / "models" / f"tfidf_{args.protocol}.joblib"
    if not train_path.is_absolute(): train_path = ROOT / train_path
    if not model_path.is_absolute(): model_path = ROOT / model_path
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found. Run the dataset preparation and split commands first.")
    train = pd.read_csv(train_path, low_memory=False).dropna(subset=["text", "label"])
    train["text"] = train["text"].astype(str)
    train["label"] = train["label"].astype(int)
    if train["label"].nunique() < 2:
        raise ValueError("Training split must contain both label classes 0 and 1.")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2), min_df=2, max_df=0.995, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed, solver="liblinear")),
    ])
    model.fit(train["text"], train["label"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {"model_type": "TF-IDF + Logistic Regression", "protocol": args.protocol,
                "train_csv": str(train_path), "output_model": str(model_path), "train_rows": len(train),
                "train_label_counts": {str(int(k)): int(v) for k, v in train["label"].value_counts().sort_index().items()},
                "seed": args.seed, "max_features": args.max_features, "scikit_learn_version": sklearn.__version__, "labels": {"0": "Fake News", "1": "True News"}}
    meta_path = model_path.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__": main()
