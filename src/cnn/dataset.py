from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)

from config import CLASSES, Config

AUTOTUNE = tf.data.AUTOTUNE


def load_dataset(cfg: Config) -> Tuple[np.ndarray, pd.DataFrame]:
    if not cfg.images_npy.exists():
        raise FileNotFoundError(
            f"{cfg.images_npy} not found. Run data_download.py first.")
    images = np.load(cfg.images_npy)
    index = pd.read_csv(cfg.index_csv)
    if len(images) != len(index):
        raise ValueError("image cache and index are not aligned")
    return images, index


def class_weights(labels: np.ndarray, num_classes: int) -> dict:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def holdout_split(index: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.holdout_fraction,
                                 random_state=cfg.seed)
    dev_idx, test_idx = next(splitter.split(index, index["label"],
                                            groups=index["lesion_id"]))
    return dev_idx, test_idx


def group_folds(index: pd.DataFrame, cfg: Config) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True,
                                    random_state=cfg.seed)
    yield from splitter.split(index, index["label"], groups=index["lesion_id"])


def naive_folds(index: pd.DataFrame, cfg: Config) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                               random_state=cfg.seed)
    yield from splitter.split(index, index["label"])


def naive_holdout(index: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    return train_test_split(np.arange(len(index)), test_size=cfg.holdout_fraction,
                            random_state=cfg.seed, stratify=index["label"])


def leakage_rate(index: pd.DataFrame, train_idx: np.ndarray,
                 val_idx: np.ndarray) -> float:
    train_groups = set(index["lesion_id"].iloc[train_idx])
    val_groups = index["lesion_id"].iloc[val_idx]
    return float(val_groups.isin(train_groups).mean())


def build_augmenter(cfg: Config) -> tf.keras.layers.Layer:
    import keras
    return keras.layers.Pipeline([
        keras.layers.RandomFlip("horizontal_and_vertical", seed=cfg.seed),
        keras.layers.RandomRotation(cfg.rotation_factor, fill_mode="reflect",
                                    seed=cfg.seed),
        keras.layers.RandomTranslation(cfg.translation_factor,
                                       cfg.translation_factor,
                                       fill_mode="reflect", seed=cfg.seed),
        keras.layers.RandomZoom(cfg.zoom_factor, cfg.zoom_factor,
                                fill_mode="reflect", seed=cfg.seed),
        keras.layers.RandomContrast(cfg.contrast_factor, value_range=(0, 255),
                                    seed=cfg.seed),
    ], name="augmenter")


def make_pipeline(images: np.ndarray, labels: np.ndarray, cfg: Config,
                  training: bool, weights: dict = None) -> tf.data.Dataset:
    one_hot = tf.keras.utils.to_categorical(labels, cfg.num_classes)
    if weights is None:
        tensors = (images, one_hot)
    else:
        sample_w = np.array([weights[int(l)] for l in labels], dtype=np.float32)
        tensors = (images, one_hot, sample_w)

    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if training:
        ds = ds.shuffle(min(len(images), 4096), seed=cfg.seed,
                        reshuffle_each_iteration=True)
    ds = ds.batch(cfg.batch_size)

    if training and cfg.augment:
        augmenter = build_augmenter(cfg)
        if weights is None:
            ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(lambda x, y, w: (augmenter(x, training=True), y, w),
                        num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def describe(index: pd.DataFrame) -> pd.DataFrame:
    counts = index["dx"].value_counts().reindex(CLASSES)
    lesions = index.groupby("dx")["lesion_id"].nunique().reindex(CLASSES)
    return pd.DataFrame({
        "images": counts,
        "lesions": lesions,
        "share_%": (counts / counts.sum() * 100).round(2),
        "images_per_lesion": (counts / lesions).round(2),
    })
