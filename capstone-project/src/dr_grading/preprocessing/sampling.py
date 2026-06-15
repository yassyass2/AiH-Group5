"""Sampling utilities for class imbalance handling."""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE


def apply_oversampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance the training split with SMOTE while preserving image shape."""
    n, h, w, c = X_train.shape
    X_flat = X_train.reshape(n, h * w * c).astype(np.float32)

    classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min())

    if len(classes) < 2:
        raise ValueError("SMOTE needs at least 2 classes in y_train.")

    if min_count < 2:
        raise ValueError(
            "SMOTE needs at least 2 samples in every class. "
            f"Class distribution: {dict(zip(classes.tolist(), counts.tolist()))}"
        )

    k_neighbors = min(5, min_count - 1)
    smote = SMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors,
    )

    X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y_train)
    X_resampled = X_resampled_flat.reshape(-1, h, w, c)

    if np.issubdtype(X_train.dtype, np.integer):
        X_resampled = np.clip(np.round(X_resampled), 0, 255).astype(X_train.dtype)
    else:
        X_resampled = X_resampled.astype(X_train.dtype)

    return X_resampled, y_resampled.astype(y_train.dtype)
