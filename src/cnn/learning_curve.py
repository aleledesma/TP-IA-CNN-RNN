import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import REPORTS_DIR, Config, ensure_dirs
from dataset import holdout_split, leakage_rate, load_dataset
from train import train_fold


def subsample_by_group(index: pd.DataFrame, train_idx: np.ndarray,
                       fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return train_idx
    subset = index.iloc[train_idx]
    splitter = GroupShuffleSplit(n_splits=1, train_size=fraction,
                                 random_state=seed)
    keep, _ = next(splitter.split(subset, subset["label"],
                                  groups=subset["lesion_id"]))
    return train_idx[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.15, 0.3, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    ensure_dirs()
    cfg = Config(image_size=args.image_size, epochs=args.epochs)
    images, index = load_dataset(cfg)

    dev_idx, _ = holdout_split(index, cfg)
    dev_index = index.iloc[dev_idx].reset_index(drop=True)
    dev_images = images[dev_idx]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                 random_state=cfg.seed)
    train_idx, val_idx = next(splitter.split(dev_index, dev_index["label"],
                                             groups=dev_index["lesion_id"]))

    rows = []
    for fraction in args.fractions:
        subset = subsample_by_group(dev_index, train_idx, fraction, cfg.seed)
        print(f"===== train fraction {fraction:.2f} "
              f"({len(subset)} images, "
              f"{dev_index['lesion_id'].iloc[subset].nunique()} lesions) =====")
        _, result = train_fold(dev_images, dev_index, subset, val_idx, cfg,
                               f"lc_{int(fraction * 100)}")
        rows.append({
            "fraction": fraction,
            "n_train": len(subset),
            "n_lesions": int(dev_index["lesion_id"].iloc[subset].nunique()),
            "leakage": leakage_rate(dev_index, subset, val_idx),
            "train_loss": round(result["train"]["loss"], 4),
            "val_loss": round(result["val"]["loss"], 4),
            "train_macro_f1": round(result["train"]["macro_f1"], 4),
            "val_macro_f1": round(result["val"]["macro_f1"], 4),
        })
        print(f"  {rows[-1]}\n")

    table = pd.DataFrame(rows)
    table["gap_macro_f1"] = (table["train_macro_f1"]
                             - table["val_macro_f1"]).round(4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(table["n_train"], table["train_loss"], "o-", label="train")
    axes[0].plot(table["n_train"], table["val_loss"], "s-", label="validation")
    axes[0].set_ylabel("loss")
    axes[1].plot(table["n_train"], table["train_macro_f1"], "o-", label="train")
    axes[1].plot(table["n_train"], table["val_macro_f1"], "s-",
                 label="validation")
    axes[1].set_ylabel("macro F1")
    for ax in axes:
        ax.set_xlabel("training images (lesion-disjoint subsample)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Learning curve vs training-set size (bias-variance diagnosis)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "learning_curve_size.png", dpi=140)
    plt.close(fig)

    (REPORTS_DIR / "learning_curve_size.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    (REPORTS_DIR / "learning_curve_size.md").write_text(
        "# Learning curve vs training-set size\n\n"
        + table.to_markdown(index=False) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
