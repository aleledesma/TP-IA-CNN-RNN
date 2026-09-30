from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "rnn"
REPORTS_DIR = ROOT / "reports" / "rnn"

JENA_URL = ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
            "jena_climate_2009_2016.csv.zip")
JENA_CSV = "jena_climate_2009_2016.csv"

TARGET = "T (degC)"
SENTINEL_COLUMNS = ["wv (m/s)", "max. wv (m/s)"]
SENTINEL_VALUE = -9999.0

BASE_FEATURES = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
    "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)",
    "H2OC (mmol/mol)", "rho (g/m**3)",
]
DERIVED_FEATURES = ["wx", "wy", "max_wx", "max_wy",
                    "day_sin", "day_cos", "year_sin", "year_cos"]
FEATURES = BASE_FEATURES + DERIVED_FEATURES


@dataclass
class Config:
    resample_rule: str = "1h"
    sequence_length: int = 120
    horizon: int = 24
    stride: int = 1

    train_fraction: float = 0.70
    val_fraction: float = 0.15

    cell: str = "lstm"
    units: tuple = (64, 64)
    dropout: float = 0.2
    recurrent_dropout: float = 0.0

    batch_size: int = 128
    epochs: int = 40
    patience: int = 8
    lr_patience: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    clipnorm: float = 1.0
    loss: str = "huber"

    seed: int = 42

    @property
    def num_features(self) -> int:
        return len(FEATURES)

    @property
    def input_shape(self) -> tuple:
        return (self.sequence_length, self.num_features)

    @property
    def purge(self) -> int:
        return self.sequence_length + self.horizon

    @property
    def frame_csv(self) -> Path:
        return DATA_PROCESSED / f"jena_{self.resample_rule}.csv"


def ensure_dirs() -> None:
    for path in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
