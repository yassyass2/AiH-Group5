from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetSummary:
    train_samples: int
    val_samples: int
    test_samples: int
    image_shape: tuple[int, ...]
    class_distribution: dict[str, dict[int, int]]


@dataclass(frozen=True)
class PreprocessedDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    summary: DatasetSummary


REQUIRED_ARRAYS = (
    "X_train",
    "y_train",
    "X_val",
    "y_val",
    "X_test",
    "y_test",
)


def load_preprocessed_data(data_source: str | Path) -> PreprocessedDataset:
    data_source = Path(data_source)
    arrays = _load_arrays(data_source)

    _validate_alignment(arrays["X_train"], arrays["y_train"], "train")
    _validate_alignment(arrays["X_val"], arrays["y_val"], "val")
    _validate_alignment(arrays["X_test"], arrays["y_test"], "test")

    summary = DatasetSummary(
        train_samples=int(arrays["X_train"].shape[0]),
        val_samples=int(arrays["X_val"].shape[0]),
        test_samples=int(arrays["X_test"].shape[0]),
        image_shape=tuple(arrays["X_train"].shape[1:]),
        class_distribution={
            "train": _class_distribution(arrays["y_train"]),
            "val": _class_distribution(arrays["y_val"]),
            "test": _class_distribution(arrays["y_test"]),
        },
    )

    return PreprocessedDataset(summary=summary, **arrays)


def _load_arrays(data_source: Path) -> dict[str, np.ndarray]:
    if data_source.is_file():
        return _load_npz_archive(data_source)
    if data_source.is_dir():
        return {name: _load_array(data_source, name) for name in REQUIRED_ARRAYS}
    raise FileNotFoundError(f"Preprocessed data source does not exist: {data_source}")


def _load_array(data_dir: Path, name: str) -> np.ndarray:
    path = data_dir / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessed array: {path}")
    return np.load(path)


def _load_npz_archive(archive_path: Path) -> dict[str, np.ndarray]:
    if archive_path.suffix != ".npz":
        raise ValueError(
            f"Expected a directory of .npy files or a .npz archive, got: {archive_path}"
        )

    with np.load(archive_path) as data:
        missing = [name for name in REQUIRED_ARRAYS if name not in data.files]
        if missing:
            missing_names = ", ".join(missing)
            raise KeyError(f"Missing arrays in {archive_path}: {missing_names}")
        return {name: data[name] for name in REQUIRED_ARRAYS}


def _validate_alignment(X: np.ndarray, y: np.ndarray, split: str) -> None:
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{split} split is misaligned: X has {X.shape[0]} samples, "
            f"y has {y.shape[0]} labels"
        )


def _class_distribution(y: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {
        int(label): int(count)
        for label, count in zip(labels.tolist(), counts.tolist(), strict=True)
    }
