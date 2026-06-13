from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def class_distribution(values) -> dict[str, int]:
    """Return a JSON-stable class distribution with sorted string keys."""
    classes, counts = np.unique(values, return_counts=True)
    return {str(int(cls)): int(count) for cls, count in zip(classes, counts)}


def make_dataset_provided_split_metadata(
    *,
    split: str,
    csv_name: str,
    image_folder: str,
    df: pd.DataFrame,
    y_saved: np.ndarray,
    id_column: str,
    label_column: str,
    smote_applied_to_saved_arrays: bool,
) -> dict[str, Any]:
    """Describe one dataset-provided split before and after preprocessing."""
    return {
        "split": split,
        "csv": csv_name,
        "image_folder": image_folder,
        "id_column": id_column,
        "label_column": label_column,
        "raw_samples": int(len(df)),
        "raw_class_distribution": class_distribution(df[label_column].to_numpy()),
        "saved_samples": int(len(y_saved)),
        "saved_class_distribution": class_distribution(y_saved),
        "smote_applied_to_saved_arrays": bool(smote_applied_to_saved_arrays),
    }
