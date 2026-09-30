import argparse
import json
import time

import keras
import numpy as np

from config import MODELS_DIR, REPORTS_DIR, Config, ensure_dirs
from dataset import prepare, split_report
from model import build_model, compile_model


def make_callbacks(cfg: Config, checkpoint) -> list:
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=cfg.lr_patience,
                                          min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss",
                                        save_best_only=True, verbose=0),
    ]


def run(cfg: Config, data: dict) -> dict:
    keras.utils.set_random_seed(cfg.seed)
    model = compile_model(build_model(cfg), cfg)
    checkpoint = MODELS_DIR / f"jena_{cfg.cell}.keras"

    start = time.time()
    history = model.fit(data["datasets"]["train"],
                        validation_data=data["datasets"]["val"],
                        epochs=cfg.epochs,
                        callbacks=make_callbacks(cfg, checkpoint), verbose=2)
    elapsed = time.time() - start

    scores = {name: model.evaluate(ds, verbose=0, return_dict=True)
              for name, ds in data["datasets"].items()}
    return {
        "cell": cfg.cell,
        "params": int(model.count_params()),
        "epochs_run": len(history.history["loss"]),
        "seconds": round(elapsed, 1),
        "scores": {k: {m: float(v) for m, v in s.items()}
                   for k, s in scores.items()},
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default=Config.cell,
                        choices=["lstm", "gru", "simple"])
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--sequence-length", type=int,
                        default=Config.sequence_length)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--no-clipnorm", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    cfg = Config(cell=args.cell, epochs=args.epochs,
                 sequence_length=args.sequence_length, horizon=args.horizon,
                 clipnorm=None if args.no_clipnorm else Config.clipnorm)

    data = prepare(cfg)
    print(split_report(data["frame"], list(data["splits"].values()),
                       cfg).to_string(index=False), "\n")

    cells = ["lstm", "gru", "simple"] if args.all_cells else [cfg.cell]
    results = []
    for cell in cells:
        print(f"===== {cell} =====")
        variant = Config(**{**cfg.__dict__, "cell": cell})
        result = run(variant, data)
        print(f"  val  loss={result['scores']['val']['loss']:.5f} "
              f"mae={result['scores']['val']['mae']:.5f}")
        print(f"  test loss={result['scores']['test']['loss']:.5f} "
              f"mae={result['scores']['test']['mae']:.5f}\n")
        results.append(result)
        keras.backend.clear_session()

    payload = {
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in cfg.__dict__.items()},
        "results": results,
    }
    tag = "all_cells" if args.all_cells else cfg.cell
    out = REPORTS_DIR / f"training_{tag}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
