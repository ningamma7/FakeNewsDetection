from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import Saliency

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description="Perturbation-based audit of DistilBERT token-attribution faithfulness (not factual verification).")
    p.add_argument("--model-path", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--text-column", default="text")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--mask-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()
    if not 0 < args.mask_fraction < 1:
        raise ValueError("mask-fraction must be greater than 0 and less than 1.")
    model_path, csv_path = Path(args.model_path), Path(args.test_csv)
    if not model_path.is_absolute(): model_path = ROOT / model_path
    if not csv_path.is_absolute(): csv_path = ROOT / csv_path
    if not model_path.exists() or not csv_path.exists(): raise FileNotFoundError(f"Missing model or test CSV: {model_path}, {csv_path}")

    frame = pd.read_csv(csv_path, low_memory=False).dropna(subset=[args.text_column]).copy()
    if frame.empty: raise ValueError("Test CSV has no non-empty text rows.")
    frame = frame.sample(min(args.samples, len(frame)), random_state=args.seed).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True).to(device)
    model.eval()
    mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.unk_token_id
    if mask_id is None: raise ValueError("Tokenizer has neither a mask token nor an unknown token for perturbation.")
    special_ids = set(tokenizer.all_special_ids)
    records = []

    def get_logits(ids, attention_mask):
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=attention_mask).logits

    for row_i, text in enumerate(frame[args.text_column].astype(str).tolist()):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length, padding=False)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        logits = get_logits(ids, mask)
        target = int(logits.argmax(dim=-1).item())
        original_logit = float(logits[0, target].item())
        embeddings = model.get_input_embeddings()(ids).detach().requires_grad_(True)
        def forward_embeds(embeds, attn):
            return model(inputs_embeds=embeds, attention_mask=attn).logits
        model.zero_grad(set_to_none=True)
        attrs = Saliency(forward_embeds).attribute(embeddings, target=target, additional_forward_args=(mask,), abs=True)
        scores = attrs.abs().sum(dim=-1)[0].detach().cpu().numpy()
        token_ids = ids[0].detach().cpu().numpy()
        eligible = [j for j, tok in enumerate(token_ids) if int(tok) not in special_ids]
        if len(eligible) < 2:
            continue
        k = max(1, min(len(eligible) - 1, int(round(len(eligible) * args.mask_fraction))))
        high = sorted(eligible, key=lambda j: scores[j], reverse=True)[:k]
        low = sorted(eligible, key=lambda j: scores[j])[:k]
        random_positions = rng.choice(eligible, size=k, replace=False).tolist()

        def measure(positions):
            changed = ids.clone()
            changed[0, positions] = mask_id
            changed_logits = get_logits(changed, mask)
            new_logit = float(changed_logits[0, target].item())
            return {"logit_abs_change": abs(original_logit - new_logit), "logit_drop": original_logit - new_logit,
                    "new_prediction": int(changed_logits.argmax(dim=-1).item())}

        high_metrics = measure(high)
        random_metrics = measure(random_positions)
        low_metrics = measure(low)
        tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
        records.append({
            "row_index": int(row_i), "target_class": target, "token_count": len(eligible), "masked_tokens": k,
            "high_attribution": high_metrics, "random_tokens": random_metrics, "low_attribution": low_metrics,
            "top_tokens": [{"token": str(tokens[j]), "saliency": float(scores[j])} for j in high[:min(10, len(high))]],
        })

    if not records: raise ValueError("No usable examples remained after tokenization.")
    high_change = np.array([r["high_attribution"]["logit_abs_change"] for r in records])
    rand_change = np.array([r["random_tokens"]["logit_abs_change"] for r in records])
    low_change = np.array([r["low_attribution"]["logit_abs_change"] for r in records])
    result = {
        "model_path": str(model_path), "test_csv": str(csv_path), "n_examples": len(records),
        "seed": args.seed, "max_length": args.max_length, "mask_fraction": args.mask_fraction,
        "mean_abs_logit_change_high_attribution": float(high_change.mean()),
        "mean_abs_logit_change_random_tokens": float(rand_change.mean()),
        "mean_abs_logit_change_low_attribution": float(low_change.mean()),
        "high_minus_random": float((high_change - rand_change).mean()),
        "high_minus_low": float((high_change - low_change).mean()),
        "interpretation": "A perturbation proxy for model sensitivity, not proof of explanation correctness or article truth. Report this as exploratory; include controls and uncertainty intervals in the paper.",
        "examples": records,
    }
    out = Path(args.output_json) if args.output_json else ROOT / "outputs" / "research" / "explanation_faithfulness.json"
    if not out.is_absolute(): out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))
    print(f"Saved full per-example analysis to {out}")

if __name__ == "__main__": main()
