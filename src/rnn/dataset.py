from dataclasses import dataclass
from typing import Tuple

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from config import FEATURES, TARGET, Config


@dataclass
class Split:
    name: str
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def chronological_splits(n_rows: int, cfg: Config) -> Tuple[Split, Split, Split]:
    train_stop = int(n_rows * cfg.train_fraction)
    val_stop = int(n_rows * (cfg.train_fraction + cfg.val_fraction))
    return (
        Split("train", 0, train_stop),
        Split("val", train_stop + cfg.purge, val_stop),
        Split("test", val_stop + cfg.purge, n_rows),
    )


def fit_scaler(frame: pd.DataFrame, split: Split) -> Scaler:
    block = frame[FEATURES].to_numpy(dtype=np.float32)[split.start:split.stop]
    std = block.std(axis=0)
    std[std < 1e-8] = 1.0
    return Scaler(block.mean(axis=0), std)


def build_targets(frame: pd.DataFrame, cfg: Config) -> np.ndarray:
    target = frame[TARGET].to_numpy(dtype=np.float32)
    n_rows = len(target)
    targets = np.full((n_rows, cfg.horizon), np.nan, dtype=np.float32)
    for h in range(1, cfg.horizon + 1):
        targets[:n_rows - h, h - 1] = target[h:]
    return targets


def make_dataset(features: np.ndarray, targets: np.ndarray, split: Split,
                 cfg: Config, shuffle: bool) -> tf.data.Dataset:
    last = split.stop - cfg.horizon
    start = split.start
    if last - start < cfg.sequence_length:
        raise ValueError(f"split {split.name} is too short")

    windows = features[start:last]
    aligned = targets[start + cfg.sequence_length - 1:last]

    return keras.utils.timeseries_dataset_from_array(
        data=windows,
        targets=aligned,
        sequence_length=cfg.sequence_length,
        sequence_stride=cfg.stride,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        seed=cfg.seed,
    )


def prepare(cfg: Config) -> dict:
    frame = pd.read_csv(cfg.frame_csv, index_col=0, parse_dates=True)
    splits = chronological_splits(len(frame), cfg)
    scaler = fit_scaler(frame, splits[0])

    features = scaler.transform(frame[FEATURES].to_numpy(dtype=np.float32))
    targets = build_targets(frame, cfg)

    datasets = {s.name: make_dataset(features, targets, s, cfg,
                                     shuffle=(s.name == "train"))
                for s in splits}
    return {
        "frame": frame,
        "splits": {s.name: s for s in splits},
        "scaler": scaler,
        "features": features,
        "targets": targets,
        "datasets": datasets,
    }


def split_report(frame: pd.DataFrame, splits, cfg: Config) -> pd.DataFrame:
    rows = []
    for split in splits:
        index = frame.index[split.start:split.stop]
        rows.append({
            "split": split.name,
            "rows": len(split),
            "from": index[0],
            "to": index[-1],
            "share_%": round(len(split) / len(frame) * 100, 1),
            "windows": max(0, len(split) - cfg.horizon - cfg.sequence_length + 1),
        })
    return pd.DataFrame(rows)


def baselines(frame: pd.DataFrame, split: Split, cfg: Config,
              train_split: Split) -> dict:
    target = frame[TARGET].to_numpy(dtype=np.float32)
    starts = np.arange(split.start,
                       split.stop - cfg.horizon - cfg.sequence_length + 1,
                       cfg.stride)
    anchors = starts + cfg.sequence_length - 1
    truth = np.stack([target[a + 1:a + 1 + cfg.horizon] for a in anchors])

    persistence = np.repeat(target[anchors][:, None], cfg.horizon, axis=1)
    seasonal = np.stack([target[a + 1 - 24:a + 1 - 24 + cfg.horizon]
                         for a in anchors])
    climatology = np.full_like(truth, target[train_split.start:train_split.stop].mean())

    return {
        "truth": truth,
        "persistence": persistence,
        "seasonal_naive": seasonal,
        "climatology": climatology,
    }
