# src_cnn/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Data layout: data/<class_name>/*.dat
    data_root: str = "data"
    wasp_dir: str = "wasp-waisted"
    goose_dir: str = "gooseneck"
    regular_dir: str = "regular"

    # Preprocessing
    n_points: int = 512
    n_classes: int = 3

    # Cross-validation
    n_splits: int = 4
    seed: int = 42

    # Training
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3

    # Callbacks
    es_patience: int = 10
    rlrop_patience: int = 5
    rlrop_factor: float = 0.2
    min_lr: float = 1e-4

    # Outputs
    results_dir: str = "results_1d_cnn"
    model_name: str = "CNN1D"
