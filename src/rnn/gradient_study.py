import argparse
import json

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import REPORTS_DIR, Config, ensure_dirs
from dataset import prepare
from model import CELLS, build_model, compile_model


def temporal_gradient_profile(model, batch, loss_fn) -> np.ndarray:
    x, y = batch
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = loss_fn(y, model(x, training=False))
    grads = tape.gradient(loss, x)
    return tf.reduce_mean(tf.norm(grads, axis=-1), axis=0).numpy()


def weight_gradient_norm(model, batch, loss_fn) -> float:
    x, y = batch
    with tf.GradientTape() as tape:
        loss = loss_fn(y, model(x, training=True))
    grads = tape.gradient(loss, model.trainable_variables)
    return float(tf.linalg.global_norm([g for g in grads if g is not None]))


def effective_memory(profile: np.ndarray, threshold: float = 0.01) -> int:
    reference = profile[-1]
    if reference <= 0:
        return 0
    relative = profile / reference
    above = np.where(relative >= threshold)[0]
    return int(len(profile) - above[0]) if len(above) else 0


def decay_ratio(profile: np.ndarray, lag: int = 100) -> float:
    if profile[-1] <= 0:
        return float("nan")
    return float(profile[max(0, len(profile) - 1 - lag)] / profile[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batches", type=int, default=8)
    args = parser.parse_args()

    ensure_dirs()
    cfg = Config()
    data = prepare(cfg)
    train_ds = data["datasets"]["train"]
    probe = [(x.numpy(), y.numpy()) for x, y in train_ds.take(args.batches)]
    loss_fn = keras.losses.Huber(delta=1.0)

    profiles, records = {}, []
    for cell in CELLS:
        keras.utils.set_random_seed(cfg.seed)
        variant = Config(**{**cfg.__dict__, "cell": cell})
        model = compile_model(build_model(variant), variant)

        untrained = np.mean([temporal_gradient_profile(model, b, loss_fn)
                             for b in probe], axis=0)
        norm_before = np.mean([weight_gradient_norm(model, b, loss_fn)
                               for b in probe])

        if args.warmup_epochs > 0:
            model.fit(train_ds.take(args.steps), epochs=args.warmup_epochs,
                      verbose=0)

        trained = np.mean([temporal_gradient_profile(model, b, loss_fn)
                           for b in probe], axis=0)
        norm_after = np.mean([weight_gradient_norm(model, b, loss_fn)
                              for b in probe])

        profiles[cell] = {"untrained": untrained, "trained": trained}
        records.append({
            "cell": cell,
            "gates": {"simple": 1, "gru": 3, "lstm": 4}[cell],
            "params": int(model.count_params()),
            "grad_norm_init": round(float(norm_before), 5),
            "grad_norm_warm": round(float(norm_after), 5),
            "clipped_at_1.0": bool(norm_after > cfg.clipnorm),
            "decay_ratio_100_steps": round(decay_ratio(trained, 100), 6),
            "effective_memory_steps": effective_memory(trained),
        })
        print(f"{cell:7s} {records[-1]}")
        keras.backend.clear_session()

    table = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    lag = np.arange(cfg.sequence_length - 1, -1, -1)
    for key, ax, title in zip(["untrained", "trained"], axes,
                              ["Random initialisation",
                               f"After {args.warmup_epochs} warm-up epochs"]):
        for cell in CELLS:
            profile = profiles[cell][key]
            ax.semilogy(lag, np.maximum(profile, 1e-20), label=cell)
        ax.set_xlabel("time steps before the forecast anchor")
        ax.set_ylabel(r"$\|\partial L / \partial x_t\|$")
        ax.set_title(title)
        ax.invert_xaxis()
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.suptitle("Gradient flow through BPTT: how far back the loss still "
                 "reaches each cell")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "bptt_gradient_flow.png", dpi=140)
    plt.close(fig)

    (REPORTS_DIR / "gradient_study.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    (REPORTS_DIR / "gradient_study.md").write_text(
        "# BPTT gradient flow - recurrent cell comparison\n\n"
        "`decay_ratio_100_steps` is the gradient norm 100 steps before the "
        "anchor divided by the norm at the anchor. Values near 0 mean the "
        "gradient vanished before reaching that depth.\n\n"
        + table.to_markdown(index=False) + "\n",
        encoding="utf-8")

    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
