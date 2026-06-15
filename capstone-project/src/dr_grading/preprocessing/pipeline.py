"""Dataset-level preprocessing pipeline assembly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from dr_grading.preprocessing.sampling import apply_oversampling
from dr_grading.preprocessing.transforms import DEFAULT_IMAGE_SIZE, preprocess_image_path


def build_dataset(
    df,
    images_dir: str | Path,
    id_column: str = "id_code",
    label_column: str = "diagnosis",
    image_extension: str = ".png",
    image_size: int = DEFAULT_IMAGE_SIZE,
    skip_cut_off: bool = True,
    denoise_method: str = "median",
    **clahe_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Build model-ready image and label arrays from a split dataframe."""
    images_dir = Path(images_dir)

    images: list[np.ndarray] = []
    labels: list[int] = []
    n_skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        image_path = images_dir / f"{row[id_column]}{image_extension}"

        processed = preprocess_image_path(
            image_path,
            image_size=image_size,
            skip_cut_off=skip_cut_off,
            denoise_method=denoise_method,
            **clahe_kwargs,
        )

        if processed is None:
            n_skipped += 1
            continue

        images.append(processed)
        labels.append(int(row[label_column]))

    print(f"Kept {len(images)} images, skipped {n_skipped} unreadable/cut-off images.")

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    return X, y


def run_pipeline(
    df,
    images_dir: str | Path,
    id_column: str = "id_code",
    label_column: str = "diagnosis",
    image_extension: str = ".png",
    image_size: int = DEFAULT_IMAGE_SIZE,
    test_size: float = 0.2,
    random_state: int = 42,
    balance_train: bool = True,
    skip_cut_off: bool = True,
    denoise_method: str = "median",
    **clahe_kwargs,
) -> dict[str, np.ndarray]:
    """Create train/validation arrays for datasets without predefined splits."""
    from sklearn.model_selection import train_test_split

    X, y = build_dataset(
        df,
        images_dir,
        id_column=id_column,
        label_column=label_column,
        image_extension=image_extension,
        image_size=image_size,
        skip_cut_off=skip_cut_off,
        denoise_method=denoise_method,
        **clahe_kwargs,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if balance_train:
        X_train, y_train = apply_oversampling(
            X_train,
            y_train,
            random_state=random_state,
        )
        X_train = X_train.astype(np.float32)
        print(f"After SMOTE: training set has {len(X_train)} images.")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }
