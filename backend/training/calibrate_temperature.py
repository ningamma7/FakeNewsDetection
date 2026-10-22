from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def get_logits(model_type: str, model_path: Path, texts: list[str], max_length: int, batch_size: int) -> np.ndarray:
    if model_type == "tfidf":
        model = joblib.load(model_path)
        probs_raw = model.predict_proba(texts)
        probs = np.zeros((len(texts), 2), dtype=float)
        for i, cls in enumerate(model.classes_): probs[:, int(cls)] = probs_raw[:, i]
        return np.log(np.clip(probs, 1e-12, 1.0))
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True).to(device)
    model.eval(); rows = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tok(texts[start:start + batch_size], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            rows.extend(model(**enc).logits.cpu().numpy().tolist())
    return np.asarray(rows, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description="Fit a scalar temperature on validation data only.")
    p.add_argument("--model-type", choices=["tfidf", "transformer"], required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--validation-csv", required=True)
    p.add_argument("--text-column", default="text")
    p.add_argument("--label-column", default="label")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()
    model_path = Path(args.model_path); val_path = Path(args.validation_csv)
    if not model_path.is_absolute(): model_path = ROOT / model_path
    if not val_path.is_absolute(): val_path = ROOT / val_path
    frame = pd.read_csv(val_path, low_memory=False).dropna(subset=[args.text_column, args.label_column]).copy()
    if frame.empty: raise ValueError("Validation CSV contains no valid rows.")
    y = pd.to_numeric(frame[args.label_column], errors="raise").astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}): raise ValueError("Labels must be 0=Fake, 1=True.")
    logits = get_logits(args.model_type, model_path, frame[args.text_column].astype(str).tolist(), args.max_length, args.batch_size)
    def nll(log_t: float) -> float:
        p = np.clip(softmax_np(logits / float(np.exp(log_t))), 1e-12, 1.0)
        return float(-np.log(p[np.arange(len(y)), y]).mean())
    opt = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
    temperature = float(np.exp(opt.x))
    result = {"model_type": args.model_type, "model_path": str(model_path), "validation_csv": str(val_path),
              "n_validation": int(len(y)), "temperature": temperature, "nll_before": nll(0.0), "nll_after": float(opt.fun),
              "fit_on": "validation only", "warning": "Do not use test data to fit or adjust temperature."}
    out = Path(args.output_json) if args.output_json else ROOT / "outputs" / "research" / f"temperature_{args.model_type}_{val_path.parent.name}.json"
    if not out.is_absolute(): out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); print(f"Saved {out}")

if __name__ == "__main__": main()
