from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "cnn"
REPORTS_DIR = ROOT / "reports" / "cnn"

DATAVERSE_DOI = "doi:10.7910/DVN/DBW86T"
DATAVERSE_FILES = {
    "HAM10000_metadata.tab": 4338392,
    "HAM10000_images_part_1.zip": 3172585,
    "HAM10000_images_part_2.zip": 3172584,
}

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
CLASS_LONG_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


@dataclass
class Config:
    image_size: int = 64
    num_classes: int = len(CLASSES)
    seed: int = 42

    n_folds: int = 5
    batch_size: int = 64
    epochs: int = 60
    patience: int = 12
    lr_patience: int = 5

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    clipnorm: float = 1.0

    stem_filters: int = 32
    stage_filters: tuple = (64, 128, 256)
    dense_units: int = 256
    dropout: float = 0.4
    spatial_dropout: tuple = (0.0, 0.1, 0.2, 0.3)

    use_class_weights: bool = True
    augment: bool = True

    rotation_factor: float = 0.15
    zoom_factor: float = 0.15
    contrast_factor: float = 0.15
    translation_factor: float = 0.10

    holdout_fraction: float = 0.15

    @property
    def input_shape(self) -> tuple:
        return (self.image_size, self.image_size, 3)

    @property
    def images_npy(self) -> Path:
        return DATA_PROCESSED / f"ham10000_images_{self.image_size}.npy"

    @property
    def index_csv(self) -> Path:
        return DATA_PROCESSED / f"ham10000_index_{self.image_size}.csv"


def ensure_dirs() -> None:
    for path in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
