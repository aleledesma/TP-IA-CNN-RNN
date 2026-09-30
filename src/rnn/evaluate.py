import argparse
import json

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MODELS_DIR, REPORTS_DIR, TARGET, Config
from dataset import baselines, make_dataset, prepare


def horizon_scores(truth: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    error = pred - truth
    return pd.DataFrame({
        "horizon_h": np.arange(1, truth.shape[1] + 1),
        "mae": np.abs(error).mean(axis=0).round(4),
        "rmse": np.sqrt((error ** 2).mean(axis=0)).round(4),
        "bias": error.mean(axis=0).round(4),
    })


def skill_score(model_mae: float, reference_mae: float) -> float:
    return float(1.0 - model_mae / reference_mae)


def plot_horizon(scores: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, table in scores.items():
        style = "o-" if name == "model" else "--"
        ax.plot(table["horizon_h"], table["mae"], style, label=name)
    ax.set_xlabel("forecast horizon (hours ahead)")
    ax.set_ylabel("MAE (degC)")
    ax.set_title("Predictive stability: error growth with the horizon")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_forecast(truth: np.ndarray, pred: np.ndarray, index, path,
                  samples: int = 3) -> None:
    step = max(1, len(truth) // (samples + 1))
    fig, axes = plt.subplots(samples, 1, figsize=(10, 3 * samples),
                             sharex=False)
    for ax, i in zip(np.atleast_1d(axes), range(step, step * (samples + 1), step)):
        hours = np.arange(1, truth.shape[1] + 1)
        ax.plot(hours, truth[i], "o-", label="observed")
        ax.plot(hours, pred[i], "s--", label="forecast")
        ax.set_title(str(index[i]))
        ax.set_xlabel("hours ahead")
        ax.set_ylabel("T (degC)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Sampled 24 h forecast trajectories on the held-out test span")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_residuals(truth: np.ndarray, pred: np.ndarray, path) -> None:
    error = (pred - truth).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(error, bins=80, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("forecast error (degC)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Residuals  mean={error.mean():.3f}  std={error.std():.3f}")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(truth.ravel()[::37], pred.ravel()[::37], s=4, alpha=0.3)
    lo, hi = truth.min(), truth.max()
    axes[1].plot([lo, hi], [lo, hi], "r--")
    axes[1].set_xlabel("observed (degC)")
    axes[1].set_ylabel("predicted (degC)")
    axes[1].set_title("Calibration")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default=Config.cell)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = Config(cell=args.cell)
    data = prepare(cfg)
    frame, splits = data["frame"], data["splits"]
    split = splits[args.split]

    model = keras.models.load_model(MODELS_DIR / f"jena_{cfg.cell}.keras")
    ordered = make_dataset(data["features"], data["targets"], split, cfg,
                           shuffle=False)
    pred = model.predict(ordered, verbose=0)

    refs = baselines(frame, split, cfg, splits["train"])
    truth = refs["truth"][:len(pred)]
    pred = pred[:len(truth)]

    scores = {"model": horizon_scores(truth, pred)}
    for name in ("persistence", "seasonal_naive", "climatology"):
        scores[name] = horizon_scores(truth, refs[name][:len(truth)])

    series = {"model": pred}
    series.update({name: refs[name][:len(truth)]
                   for name in ("persistence", "seasonal_naive", "climatology")})

    rows = []
    for name, values in series.items():
        error = values - truth
        rows.append({
            "series": name,
            "mae": round(float(np.abs(error).mean()), 4),
            "rmse": round(float(np.sqrt((error ** 2).mean())), 4),
        })
    overall = pd.DataFrame(rows)
    reference = float(overall.loc[overall.series == "persistence",
                                  "mae"].iloc[0])
    overall["skill_vs_persistence"] = overall["mae"].apply(
        lambda m: round(skill_score(m, reference), 4))

    anchors = frame.index[split.start + cfg.sequence_length - 1:][:len(truth)]
    plot_horizon(scores, REPORTS_DIR / f"horizon_error_{cfg.cell}.png")
    plot_forecast(truth, pred, anchors,
                  REPORTS_DIR / f"forecast_samples_{cfg.cell}.png")
    plot_residuals(truth, pred, REPORTS_DIR / f"residuals_{cfg.cell}.png")

    degradation = scores["model"]
    lines = [
        f"# RNN evaluation - Jena Climate ({cfg.cell.upper()}, {args.split} split)",
        "", f"- window: {cfg.sequence_length} h, horizon: {cfg.horizon} h",
        f"- evaluated windows: {len(truth)}",
        f"- span: {anchors[0]} -> {anchors[-1]}", "",
        "## Overall (all horizons pooled)", "",
        overall.to_markdown(index=False), "",
        "## Predictive stability by horizon", "",
        degradation.to_markdown(index=False), "",
        f"MAE at +1 h: {degradation.mae.iloc[0]:.4f} degC, "
        f"at +{cfg.horizon} h: {degradation.mae.iloc[-1]:.4f} degC "
        f"(x{degradation.mae.iloc[-1] / degradation.mae.iloc[0]:.2f})", "",
    ]
    out = REPORTS_DIR / f"evaluation_{cfg.cell}_{args.split}.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    print(overall.to_string(index=False), "\n")
    print(degradation.to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
