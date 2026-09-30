import argparse
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from config import (BASE_FEATURES, DATA_RAW, JENA_CSV, JENA_URL,
                    SENTINEL_COLUMNS, SENTINEL_VALUE, Config, ensure_dirs)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
DAY_SECONDS = 24 * 3600
YEAR_SECONDS = 365.2425 * DAY_SECONDS


def download() -> None:
    ensure_dirs()
    csv_path = DATA_RAW / JENA_CSV
    if csv_path.exists():
        print("[skip] " + JENA_CSV)
        return
    archive = DATA_RAW / "jena_climate.csv.zip"
    if not archive.exists():
        print("[get ] " + JENA_URL)
        request = urllib.request.Request(JENA_URL,
                                         headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=180) as response, \
                open(archive, "wb") as handle:
            handle.write(response.read())
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA_RAW)
    print("[done] " + JENA_CSV)


def load_ordered() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / JENA_CSV)
    df["timestamp"] = pd.to_datetime(df["Date Time"],
                                     format="%d.%m.%Y %H:%M:%S")
    before = len(df)
    df = df.sort_values("timestamp", kind="mergesort")
    df = df.drop_duplicates(subset="timestamp", keep="first")
    print(f"[order] {before} rows -> {len(df)} after sort and dedup")

    for column in SENTINEL_COLUMNS:
        mask = df[column] <= SENTINEL_VALUE + 1
        print(f"[clean] {column}: {int(mask.sum())} sentinel values -> NaN")
        df.loc[mask, column] = np.nan

    return df.set_index("timestamp").drop(columns=["Date Time"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    radians = np.deg2rad(df["wd (deg)"])
    df["wx"] = df["wv (m/s)"] * np.cos(radians)
    df["wy"] = df["wv (m/s)"] * np.sin(radians)
    df["max_wx"] = df["max. wv (m/s)"] * np.cos(radians)
    df["max_wy"] = df["max. wv (m/s)"] * np.sin(radians)

    seconds = df.index.map(pd.Timestamp.timestamp)
    df["day_sin"] = np.sin(seconds * 2 * np.pi / DAY_SECONDS)
    df["day_cos"] = np.cos(seconds * 2 * np.pi / DAY_SECONDS)
    df["year_sin"] = np.sin(seconds * 2 * np.pi / YEAR_SECONDS)
    df["year_cos"] = np.cos(seconds * 2 * np.pi / YEAR_SECONDS)
    return df


def build_frame(cfg: Config) -> pd.DataFrame:
    if cfg.frame_csv.exists():
        print("[skip] " + cfg.frame_csv.name)
        return pd.read_csv(cfg.frame_csv, index_col=0, parse_dates=True)

    raw = load_ordered()
    numeric = raw[BASE_FEATURES + SENTINEL_COLUMNS + ["wd (deg)"]]
    frame = numeric.resample(cfg.resample_rule).mean()
    gaps = int(frame.isna().any(axis=1).sum())
    print(f"[grid ] {len(frame)} rows at {cfg.resample_rule}, "
          f"{gaps} with missing values")
    frame = frame.interpolate(method="time", limit_direction="both")

    frame = add_features(frame)
    frame = frame.drop(columns=SENTINEL_COLUMNS + ["wd (deg)"])
    if frame.isna().any().any():
        raise ValueError("NaN values remain after interpolation")

    frame.to_csv(cfg.frame_csv)
    print(f"[done] {cfg.frame_csv.name} {frame.shape} "
          f"({frame.index[0]} -> {frame.index[-1]})")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resample", default=Config.resample_rule)
    args = parser.parse_args()
    cfg = Config(resample_rule=args.resample)
    download()
    build_frame(cfg)


if __name__ == "__main__":
    main()
