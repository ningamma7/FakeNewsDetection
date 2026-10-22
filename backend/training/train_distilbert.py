from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from .metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length, padding=False)
        enc["labels"] = self.labels[idx]
        return enc


def collate_batch(features, tokenizer):
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    tokenizer_features = [{k: v for k, v in f.items() if k != "labels"} for f in features]
    batch = tokenizer.pad(tokenizer_features, padding=True, return_tensors="pt")
    batch["labels"] = labels
    return batch


def evaluate_loop(model, loader, device):
    model.eval(); all_y, all_p, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            bs = int(batch["labels"].shape[0])
            loss_sum += float(outputs.loss.item()) * bs; n += bs
            all_y.extend(batch["labels"].cpu().numpy().tolist())
            all_p.extend(torch.softmax(outputs.logits, dim=-1).cpu().numpy().tolist())
    probs = np.asarray(all_p, dtype=float)
    metrics = compute_metrics(all_y, probs)
    metrics["loss"] = loss_sum / max(n, 1)
    return metrics, np.asarray(all_y, dtype=int), probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT; run on a Colab GPU for practical speed.")
    parser.add_argument("--protocol", choices=["random", "temporal"], default="random")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--validation-csv", default=None)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=0, help="Optional small smoke test; 0 uses all rows")
    args = parser.parse_args()
    set_seed(args.seed)
    split_dir = ROOT / "data" / "processed" / "research_splits" / args.protocol
    train_path = Path(args.train_csv) if args.train_csv else split_dir / "train.csv"
    val_path = Path(args.validation_csv) if args.validation_csv else split_dir / "validation.csv"
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "backend" / "models" / f"research_distilbert_{args.protocol}"
    if not train_path.is_absolute(): train_path = ROOT / train_path
    if not val_path.is_absolute(): val_path = ROOT / val_path
    if not output_dir.is_absolute(): output_dir = ROOT / output_dir
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing train/validation files: {train_path}, {val_path}. Prepare splits first.")
    train = pd.read_csv(train_path, low_memory=False).dropna(subset=["text", "label"]).copy()
    val = pd.read_csv(val_path, low_memory=False).dropna(subset=["text", "label"]).copy()
    train["label"] = train["label"].astype(int); val["label"] = val["label"].astype(int)
    if args.max_train_rows > 0:
        train = train.sample(min(args.max_train_rows, len(train)), random_state=args.seed).copy()
    if train["label"].nunique() < 2 or val["label"].nunique() < 2:
        raise ValueError("Train and validation files must contain both classes (0=Fake, 1=True).")

    print(f"Runtime: torch={torch.__version__}; device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training {len(train)} articles; validation {len(val)} articles; protocol={args.protocol}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2,
        id2label={0: "Fake News", 1: "True News"},
        label2id={"Fake News": 0, "True News": 1},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_ds = TextDataset(train, tokenizer, args.max_length); val_ds = TextDataset(val, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda x: collate_batch(x, tokenizer), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda x: collate_batch(x, tokenizer), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except (AttributeError, TypeError):  # Compatibility with older PyTorch runtimes
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0; history = []; start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0; seen = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            bs = int(batch["labels"].shape[0]); loss_sum += float(loss.item()) * bs; seen += bs
            if step % 100 == 0:
                print(f"epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} train_loss={loss_sum/max(seen,1):.4f}")
        val_metrics, _, _ = evaluate_loop(model, val_loader, device)
        epoch_info = {"epoch": epoch, "train_loss": loss_sum / max(seen, 1), "validation": val_metrics}
        history.append(epoch_info)
        print(json.dumps(epoch_info, indent=2))
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            (output_dir / "training_metadata.json").write_text(json.dumps({
                "protocol": args.protocol, "model_name": args.model_name, "seed": args.seed,
                "epochs_requested": args.epochs, "best_epoch": epoch, "max_length": args.max_length,
                "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "train_rows": len(train), "validation_rows": len(val), "best_validation_macro_f1": best_f1,
                "label_mapping": {"0": "Fake News", "1": "True News"},
                "history": history,
            }, indent=2), encoding="utf-8")
    elapsed = time.time() - start
    result = {"protocol": args.protocol, "output_dir": str(output_dir), "best_validation_macro_f1": best_f1,
              "epochs_ran": len(history), "elapsed_seconds": round(elapsed, 2), "history": history,
              "device": str(device), "torch_version": torch.__version__}
    result_path = ROOT / "outputs" / "research" / f"distilbert_training_{args.protocol}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint: {output_dir}\nTraining report: {result_path}")

if __name__ == "__main__": main()
