import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from config import REPORTS_DIR, Config, ensure_dirs
from dataset import (holdout_split, load_dataset, make_pipeline, naive_folds,
                     group_folds, leakage_rate)
from train import train_fold


def expected_leakage(index: pd.DataFrame, val_fraction: float) -> float:
    sizes = index.groupby("lesion_id").size()
    per_image = index["lesion_id"].map(sizes).to_numpy()
    return float(np.mean(1.0 - val_fraction ** (per_image - 1)))


def score(model, images, labels, cfg) -> dict:
    ds = make_pipeline(images, labels, cfg, training=False)
    probs = model.predict(ds, verbose=0)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float((pred == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro",
                                   zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    ensure_dirs()
    cfg = Config(image_size=args.image_size, epochs=args.epochs)
    images, index = load_dataset(cfg)
    labels = index["label"].to_numpy()

    dev_idx, test_idx = holdout_split(index, cfg)
    dev_index = index.iloc[dev_idx].reset_index(drop=True)
    dev_images = images[dev_idx]

    print("expected leakage under a random image-level split")
    for frac in (0.1, 0.15, 0.2, 0.25, 0.3):
        print(f"  val_fraction={frac:.2f} -> "
              f"{expected_leakage(index, frac):.4f}")
    print()

    results = {}
    for name, splitter in (("naive", naive_folds), ("group", group_folds)):
        train_idx, val_idx = next(splitter(dev_index, cfg))
        contamination = leakage_rate(dev_index, train_idx, val_idx)
        print(f"===== {name} split (measured leakage={contamination:.4f}) =====")
        model, result = train_fold(dev_images, dev_index, train_idx, val_idx,
                                   cfg, f"leakage_{name}")
        results[name] = {
            "measured_leakage": contamination,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "validation": score(model, dev_images[val_idx],
                                dev_index["label"].to_numpy()[val_idx], cfg),
            "holdout_test": score(model, images[test_idx], labels[test_idx],
                                  cfg),
            "epochs_run": result["epochs_run"],
        }
        print(f"  validation   : {results[name]['validation']}")
        print(f"  holdout test : {results[name]['holdout_test']}\n")

    rows = []
    for name, res in results.items():
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            rows.append({
                "split": name,
                "metric": metric,
                "validation": round(res["validation"][metric], 4),
                "holdout_test": round(res["holdout_test"][metric], 4),
                "optimism": round(res["validation"][metric]
                                  - res["holdout_test"][metric], 4),
            })
    table = pd.DataFrame(rows)

    out_json = REPORTS_DIR / "leakage_study.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out_md = REPORTS_DIR / "leakage_study.md"
    out_md.write_text(
        "# Data leakage study - random vs lesion-grouped partitioning\n\n"
        f"Measured leakage, random split: "
        f"{results['naive']['measured_leakage']:.4f}\n\n"
        f"Measured leakage, group split: "
        f"{results['group']['measured_leakage']:.4f}\n\n"
        + table.to_markdown(index=False) + "\n",
        encoding="utf-8")

    print(table.to_string(index=False))
    print(f"\nsaved -> {out_md}")


if __name__ == "__main__":
    main()
