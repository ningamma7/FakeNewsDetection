from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[2]


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def transformer_logits(model_path: Path, texts: list[str], max_length: int = 256, batch_size: int = 16) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True).to(device)
    model.eval(); rows = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(texts[start:start+batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            rows.extend(model(**enc).logits.cpu().numpy().tolist())
    return np.asarray(rows, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on an untouched test CSV. Optionally apply a validation-fitted temperature.")
    parser.add_argument("--model-type", choices=["tfidf", "transformer"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature-json", default=None, help="Optional temperature fitted on validation data only.")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()
    model_path = Path(args.model_path); test_path = Path(args.test_csv)
    if not model_path.is_absolute(): model_path = ROOT / model_path
    if not test_path.is_absolute(): test_path = ROOT / test_path
    if not model_path.exists(): raise FileNotFoundError(model_path)
    if not test_path.exists(): raise FileNotFoundError(test_path)
    df = pd.read_csv(test_path, low_memory=False).dropna(subset=[args.text_column, args.label_column]).copy()
    if df.empty: raise ValueError("Test CSV is empty after dropping null text/label rows.")
    texts = df[args.text_column].astype(str).tolist(); y = df[args.label_column].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}): raise ValueError("Labels must map to 0=Fake and 1=True.")
    if args.model_type == "tfidf":
        model = joblib.load(model_path)
        raw = model.predict_proba(texts)
        probs_raw = np.zeros((len(texts), 2), dtype=float)
        for j, label in enumerate(model.classes_): probs_raw[:, int(label)] = raw[:, j]
        logits = np.log(np.clip(probs_raw, 1e-12, 1.0))
    else:
        logits = transformer_logits(model_path, texts, args.max_length, args.batch_size)
    temperature = 1.0
    if args.temperature_json:
        temp_path = Path(args.temperature_json)
        if not temp_path.is_absolute(): temp_path = ROOT / temp_path
        temperature = float(json.loads(temp_path.read_text(encoding="utf-8"))["temperature"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Calibration temperature must be finite and greater than zero.")
    probs = softmax_np(logits / temperature)
    metrics = compute_metrics(y, probs)
    metrics.update({"model_type": args.model_type, "model_path": str(model_path), "test_csv": str(test_path),
                    "labels": {"0": "Fake News", "1": "True News"},
                    "max_length": args.max_length if args.model_type == "transformer" else None,
                    "temperature_applied": temperature, "temperature_calibration_file": args.temperature_json,
                    "warning": "Temperature must be fit on validation data only; never optimize against this test set."})
    prefix = Path(args.output_prefix) if args.output_prefix else ROOT / "outputs" / "research" / f"evaluation_{args.model_type}_{test_path.parent.name}"
    if not prefix.is_absolute(): prefix = ROOT / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = prefix.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pred_df = pd.DataFrame({"text": texts, "y_true": y, "y_pred": probs.argmax(axis=1), "p_fake": probs[:, 0], "p_true": probs[:, 1]})
    pred_df.to_csv(prefix.with_name(prefix.name + "_predictions.csv"), index=False)
    print(json.dumps(metrics, indent=2)); print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__": main()
