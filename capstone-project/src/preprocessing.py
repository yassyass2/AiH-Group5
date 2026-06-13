import cv2
import numpy as np
from imblearn.over_sampling import SMOTE

DEFAULT_IMAGE_SIZE = 224  # EfficientNet-B0 input resolution; shared source of
#                           truth, imported by src/pre_proc_pipeline.py.


def crop_black_border_rgb(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Crop near-black camera background from an RGB image."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if not mask.any():
        return image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    y1, y2 = int(rows[0]), int(rows[-1])
    x1, x2 = int(cols[0]), int(cols[-1])
    return image[y1 : y2 + 1, x1 : x2 + 1]


def resize_with_padding(
    image: np.ndarray,
    target_size: int = DEFAULT_IMAGE_SIZE,
    pad_value: int = 0,
) -> np.ndarray:
    """Resize an RGB image to a square canvas without distorting aspect ratio."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    height, width = image.shape[:2]
    scale = target_size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full(
        (target_size, target_size, 3),
        pad_value,
        dtype=resized.dtype,
    )
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return canvas


def apply_green_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Extract the green channel and apply CLAHE."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")

    green = image[:, :, 1]
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(green)


def preprocess_fundus_rgb(
    image: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop_threshold: int = 10,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    median_kernel_size: int = 3,
) -> np.ndarray:
    """
    Canonical per-image preprocessing.

    Input: RGB uint8 image.
    Output: float32 tensor, shape (image_size, image_size, 1), values in [0, 1].
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    image = crop_black_border_rgb(image, threshold=crop_threshold)
    image = resize_with_padding(image, target_size=image_size)
    green = apply_green_clahe(
        image,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )
    denoised = cv2.medianBlur(green, median_kernel_size)
    normalized = denoised.astype(np.float32) / 255.0
    return normalized[..., np.newaxis]


def preprocess_image_path(
    image_path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop_threshold: int = 10,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    median_kernel_size: int = 3,
) -> np.ndarray:
    """Read an image from disk and apply canonical preprocessing."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return preprocess_fundus_rgb(
        image_rgb,
        image_size=image_size,
        crop_threshold=crop_threshold,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
        median_kernel_size=median_kernel_size,
    )


def crop_black_border(image: np.ndarray, threshold: int = 20) -> np.ndarray:
    """
    Crop the black background around a fundus image.

    Input:
        image: BGR uint8 image

    Output:
        cropped BGR uint8 image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold

    if not mask.any():
        return image

    y_indices, x_indices = np.where(mask)

    y1, y2 = y_indices.min(), y_indices.max()
    x1, x2 = x_indices.min(), x_indices.max()

    return image[y1:y2 + 1, x1:x2 + 1]


def reduce_fundus_noise(
    image: np.ndarray,
    method: str = "bilateral",
) -> np.ndarray:
    """
    Mild noise reduction for retinal fundus images.

    Keep this conservative. Diabetic retinopathy features like microaneurysms,
    hemorrhages, vessels, and small exudates can be tiny, so aggressive blur
    can remove useful information.

    Recommended default:
        bilateral

    Input:
        image: BGR image, usually uint8

    Output:
        denoised BGR uint8 image
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if method == "none":
        return image

    if method == "bilateral":
        return cv2.bilateralFilter(
            image,
            d=5,
            sigmaColor=35,
            sigmaSpace=35,
        )

    if method == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=3,
            hColor=3,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    if method == "median":
        return cv2.medianBlur(image, 3)

    if method == "gaussian":
        return cv2.GaussianBlur(image, (3, 3), 0)

    raise ValueError(f"Unknown denoising method: {method}")


def apply_oversampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance the TRAINING set with SMOTE.

    SMOTE operates on flattened pixels. We cast to float32 first because SMOTE
    interpolates between samples. Doing that directly on uint8 can corrupt
    values through integer wraparound.

    Returns:
        X_resampled: image array with the same dtype as ``X_train``,
            shape (N, H, W, C)
        y_resampled: label array
    """
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

    # SMOTE default k_neighbors=5 fails if the smallest class has <=5 samples.
    k_neighbors = min(5, min_count - 1)

    smote = SMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors,
    )

    X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y_train)

    X_resampled = X_resampled_flat.reshape(-1, h, w, c)

    # Restore the caller's dtype. SMOTE interpolates between existing samples,
    # so values stay within the original range; integer inputs (uint8 [0,255])
    # are rounded and clipped, while float inputs (e.g. normalised [0,1]) are
    # kept as-is -- clipping them to [0,255].astype(uint8) would zero them out.
    if np.issubdtype(X_train.dtype, np.integer):
        X_resampled = np.clip(np.round(X_resampled), 0, 255).astype(X_train.dtype)
    else:
        X_resampled = X_resampled.astype(X_train.dtype)

    y_resampled = y_resampled.astype(y_train.dtype)

    return X_resampled, y_resampled
