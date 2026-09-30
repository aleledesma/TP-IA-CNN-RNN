import argparse
import json
import time

import keras
import numpy as np

from config import MODELS_DIR, REPORTS_DIR, Config, ensure_dirs
from dataset import (class_weights, describe, group_folds, holdout_split,
                     leakage_rate, load_dataset, make_pipeline, naive_folds)
from model import build_model, compile_model, receptive_field_report


def make_callbacks(cfg: Config, checkpoint_path) -> list:
    return [
        keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max",
                                      patience=cfg.patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max",
                                          factor=0.5, patience=cfg.lr_patience,
                                          min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_macro_f1",
                                        mode="max", save_best_only=True,
                                        verbose=0),
    ]


def train_fold(images, index, train_idx, val_idx, cfg, tag):
    labels = index["label"].to_numpy()
    weights = class_weights(labels[train_idx], cfg.num_classes) \
        if cfg.use_class_weights else None

    train_ds = make_pipeline(images[train_idx], labels[train_idx], cfg,
                             training=True, weights=weights)
    val_ds = make_pipeline(images[val_idx], labels[val_idx], cfg,
                           training=False)

    keras.utils.set_random_seed(cfg.seed)
    model = compile_model(build_model(cfg), cfg)
    checkpoint = MODELS_DIR / f"{tag}.keras"

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs,
                        callbacks=make_callbacks(cfg, checkpoint), shuffle=False,
                        verbose=2)
    elapsed = time.time() - start

    train_eval = model.evaluate(
        make_pipeline(images[train_idx], labels[train_idx], cfg,
                      training=False), verbose=0, return_dict=True)
    val_eval = model.evaluate(val_ds, verbose=0, return_dict=True)

    return model, {
        "tag": tag,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "leakage_rate": leakage_rate(index, train_idx, val_idx),
        "epochs_run": len(history.history["loss"]),
        "seconds": round(elapsed, 1),
        "train": {k: float(v) for k, v in train_eval.items()},
        "val": {k: float(v) for k, v in val_eval.items()},
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--folds", type=int, default=Config.n_folds)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--split", choices=["group", "naive"], default="group")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    cfg = Config(image_size=args.image_size, epochs=args.epochs,
                 n_folds=args.folds, batch_size=args.batch_size,
                 augment=not args.no_augment,
                 use_class_weights=not args.no_class_weights)

    images, index = load_dataset(cfg)
    print(describe(index).to_string(), "\n")
    print(receptive_field_report(cfg).to_string(index=False), "\n")

    dev_idx, test_idx = holdout_split(index, cfg)
    np.save(REPORTS_DIR / "test_idx.npy", test_idx)
    print(f"dev={len(dev_idx)}  held-out test={len(test_idx)}  "
          f"test leakage={leakage_rate(index, dev_idx, test_idx):.4f}\n")

    dev_index = index.iloc[dev_idx].reset_index(drop=True)
    dev_images = images[dev_idx]
    splitter = group_folds if args.split == "group" else naive_folds

    folds = []
    for k, (tr, va) in enumerate(splitter(dev_index, cfg)):
        tag = f"{args.split}_fold{k}"
        print(f"===== {tag} =====")
        _, result = train_fold(dev_images, dev_index, tr, va, cfg, tag)
        print(f"  val macro_f1={result['val']['macro_f1']:.4f} "
              f"acc={result['val']['acc']:.4f} "
              f"leakage={result['leakage_rate']:.4f}\n")
        folds.append(result)

    summary = {
        "split": args.split,
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in cfg.__dict__.items()},
        "folds": folds,
        "mean": {m: float(np.mean([f["val"][m] for f in folds]))
                 for m in folds[0]["val"]},
        "std": {m: float(np.std([f["val"][m] for f in folds]))
                for m in folds[0]["val"]},
    }
    out = REPORTS_DIR / f"cv_{args.split}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("===== cross-validation summary =====")
    for metric in ("loss", "acc", "macro_f1"):
        print(f"  val {metric:9s} {summary['mean'][metric]:.4f} "
              f"+/- {summary['std'][metric]:.4f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
