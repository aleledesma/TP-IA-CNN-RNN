import argparse
import json

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             roc_auc_score)

from config import CLASSES, MODELS_DIR, REPORTS_DIR, Config
from dataset import holdout_split, load_dataset, make_pipeline


def load_fold_models(split: str) -> list:
    paths = sorted(MODELS_DIR.glob(f"{split}_fold*.keras"))
    if not paths:
        raise FileNotFoundError(f"no trained models for split '{split}'")
    return [keras.models.load_model(p) for p in paths]


def predict_ensemble(models: list, ds) -> np.ndarray:
    return np.mean([m.predict(ds, verbose=0) for m in models], axis=0)


def metric_table(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = probs.argmax(axis=1)
    present = np.unique(y_true)
    one_hot = np.eye(len(CLASSES))[y_true]
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=present, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_auc_ovr": float(roc_auc_score(one_hot, probs, average="macro",
                                             multi_class="ovr")),
    }


def plot_history(summary: dict, path) -> None:
    folds = summary["folds"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for key, ax, title in zip(["loss", "acc", "macro_f1"], axes,
                              ["Loss", "Accuracy", "Macro F1"]):
        for i, fold in enumerate(folds):
            ax.plot(fold["history"][key], color="tab:blue", alpha=0.45,
                    label="train" if i == 0 else None)
            ax.plot(fold["history"][f"val_{key}"], color="tab:red", alpha=0.45,
                    label="validation" if i == 0 else None)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Learning curves across GroupKFold folds (bias-variance view)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path) -> None:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Held-out test confusion matrix (row-normalised)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def bias_variance(summary: dict) -> pd.DataFrame:
    rows = []
    for fold in summary["folds"]:
        rows.append({
            "fold": fold["tag"],
            "train_loss": fold["train"]["loss"],
            "val_loss": fold["val"]["loss"],
            "gap_loss": fold["val"]["loss"] - fold["train"]["loss"],
            "train_macro_f1": fold["train"]["macro_f1"],
            "val_macro_f1": fold["val"]["macro_f1"],
            "gap_macro_f1": fold["train"]["macro_f1"] - fold["val"]["macro_f1"],
            "leakage": fold["leakage_rate"],
        })
    df = pd.DataFrame(rows).round(4)
    df.loc["mean"] = df.drop(columns="fold").mean(numeric_only=True).round(4)
    df.loc["std"] = df.drop(columns="fold").std(numeric_only=True).round(4)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="group")
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    args = parser.parse_args()

    cfg = Config(image_size=args.image_size)
    images, index = load_dataset(cfg)
    _, test_idx = holdout_split(index, cfg)

    labels = index["label"].to_numpy()
    test_ds = make_pipeline(images[test_idx], labels[test_idx], cfg,
                            training=False)
    models = load_fold_models(args.split)
    probs = predict_ensemble(models, test_ds)
    y_true = labels[test_idx]

    summary = json.loads((REPORTS_DIR / f"cv_{args.split}.json")
                         .read_text(encoding="utf-8"))
    bv = bias_variance(summary)
    scores = metric_table(y_true, probs)
    report = classification_report(y_true, probs.argmax(axis=1),
                                   target_names=CLASSES, zero_division=0,
                                   digits=4)
    cm = confusion_matrix(y_true, probs.argmax(axis=1),
                          labels=range(len(CLASSES)))

    plot_history(summary, REPORTS_DIR / f"learning_curves_{args.split}.png")
    plot_confusion(cm, REPORTS_DIR / f"confusion_matrix_{args.split}.png")

    lines = [
        f"# CNN evaluation - HAM10000 ({args.split} split)", "",
        "## Cross-validation bias / variance", "",
        bv.to_markdown(), "",
        "## Held-out test set (lesion-disjoint, never seen in CV)", "",
        f"- images: {len(test_idx)}",
        f"- lesions: {index['lesion_id'].iloc[test_idx].nunique()}",
        f"- ensemble of {len(models)} fold models", "",
        pd.Series(scores).round(4).to_markdown(), "",
        "## Per-class report", "", "```", report, "```", "",
        "## Confusion matrix (counts)", "",
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_markdown(), "",
    ]
    out = REPORTS_DIR / f"evaluation_{args.split}.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    print(bv.to_string(), "\n")
    print(pd.Series(scores).round(4).to_string(), "\n")
    print(report)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
