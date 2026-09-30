import argparse
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from config import (DATA_RAW, DATAVERSE_FILES, CLASS_INDEX, Config,
                    ensure_dirs)

ACCESS_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
IMAGE_DIR = DATA_RAW / "images"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
CHUNK = 1 << 20


def _fetch(url: str, target) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with open(partial, "wb") as handle:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    print("\r  {:6.2f}% ({:.0f}/{:.0f} MB)".format(
                        done / total * 100, done / 1e6, total / 1e6),
                        end="", flush=True)
    if total and done != total:
        partial.unlink(missing_ok=True)
        raise IOError("incomplete download: {}/{} bytes".format(done, total))
    partial.replace(target)
    print()


def download_files() -> None:
    ensure_dirs()
    for name, file_id in DATAVERSE_FILES.items():
        target = DATA_RAW / name
        if target.exists() and target.stat().st_size > 0:
            print("[skip] " + name)
            continue
        print("[get ] " + name)
        _fetch(ACCESS_URL.format(file_id=file_id), target)


def extract_images() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for name in DATAVERSE_FILES:
        if not name.endswith(".zip"):
            continue
        with zipfile.ZipFile(DATA_RAW / name) as archive:
            members = [m for m in archive.namelist() if m.lower().endswith(".jpg")]
            pending = [m for m in members
                       if not (IMAGE_DIR / m.split("/")[-1]).exists()]
            if not pending:
                print("[skip] extract " + name)
                continue
            print("[open] {}: {} images".format(name, len(pending)))
            for member in pending:
                with archive.open(member) as src, \
                        open(IMAGE_DIR / member.split("/")[-1], "wb") as dst:
                    shutil.copyfileobj(src, dst)


def _load_square(path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)


def build_cache(cfg: Config) -> None:
    if cfg.images_npy.exists() and cfg.index_csv.exists():
        print("[skip] cache " + cfg.images_npy.name)
        return

    meta = pd.read_csv(DATA_RAW / "HAM10000_metadata.tab", sep="\t")
    meta = meta.sort_values("image_id").reset_index(drop=True)
    meta["label"] = meta["dx"].map(CLASS_INDEX)
    if meta["label"].isna().any():
        raise ValueError("unmapped diagnosis label in metadata")

    images = np.zeros((len(meta), cfg.image_size, cfg.image_size, 3),
                      dtype=np.uint8)
    missing = []
    for i, image_id in enumerate(meta["image_id"]):
        path = IMAGE_DIR / (image_id + ".jpg")
        if not path.exists():
            missing.append(image_id)
            continue
        images[i] = _load_square(path, cfg.image_size)
        if (i + 1) % 1000 == 0:
            print("  {}/{}".format(i + 1, len(meta)))

    if missing:
        raise FileNotFoundError(
            "{} images missing, e.g. {}".format(len(missing), missing[:5]))

    np.save(cfg.images_npy, images)
    meta[["image_id", "lesion_id", "dx", "dx_type", "label", "age", "sex",
          "localization"]].to_csv(cfg.index_csv, index=False)
    print("[done] {} {} ({:.1f} MB)".format(
        cfg.images_npy.name, images.shape, images.nbytes / 1e6))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    cfg = Config(image_size=args.image_size)
    download_files()
    extract_images()
    build_cache(cfg)

    if not args.keep_archives:
        for name in DATAVERSE_FILES:
            if name.endswith(".zip"):
                (DATA_RAW / name).unlink(missing_ok=True)
        print("[clean] archives removed")


if __name__ == "__main__":
    main()
