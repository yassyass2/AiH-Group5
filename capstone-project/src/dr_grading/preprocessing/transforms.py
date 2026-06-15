"""Image-level preprocessing transforms for retinal fundus images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_IMAGE_SIZE = 224


def crop_black_border(image: np.ndarray, threshold: int = 20) -> np.ndarray:
    """Crop the black background around a BGR fundus image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold

    if not mask.any():
        return image

    y_indices, x_indices = np.where(mask)
    y1, y2 = y_indices.min(), y_indices.max()
    x1, x2 = x_indices.min(), x_indices.max()

    return image[y1 : y2 + 1, x1 : x2 + 1]


def reduce_fundus_noise(
    image: np.ndarray,
    method: str = "bilateral",
) -> np.ndarray:
    """Apply conservative noise reduction without removing small DR features."""
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


def create_retina_mask(image: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Create a binary mask of the bright retina against the black background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def _border_retina_ratios(
    mask: np.ndarray,
    border_width_ratio: float = 0.02,
) -> dict[str, float]:
    height, width = mask.shape
    border_w = max(1, int(width * border_width_ratio))
    border_h = max(1, int(height * border_width_ratio))

    return {
        "left": float(mask[:, :border_w].mean()),
        "right": float(mask[:, width - border_w :].mean()),
        "top": float(mask[:border_h, :].mean()),
        "bottom": float(mask[height - border_h :, :].mean()),
    }


def is_cut_off(
    image: np.ndarray,
    threshold: int = 15,
    border_width_ratio: float = 0.01,
    max_side_ratio: float = 0.30,
    min_cut_sides: int = 3,
) -> bool:
    """Return whether a fundus image is over-cropped at the image borders."""
    mask = create_retina_mask(image, threshold=threshold)
    ratios = _border_retina_ratios(mask, border_width_ratio=border_width_ratio)
    n_cut_sides = sum(value > max_side_ratio for value in ratios.values())

    return n_cut_sides >= min_cut_sides


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to a colour BGR image via its LAB lightness channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lightness = clahe.apply(lightness)

    lab = cv2.merge([lightness, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def extract_green_channel_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    to_three_channels: bool = True,
) -> np.ndarray:
    """Extract the green channel and apply CLAHE."""
    green = image[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    green = clahe.apply(green)

    if to_three_channels:
        return cv2.merge([green, green, green]).astype(np.uint8)

    return green.astype(np.uint8)


def preprocess_image(
    image: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    denoise_method: str = "median",
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Run the paper-aligned per-image transform and return float32 [0, 1]."""
    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    image = apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    image = reduce_fundus_noise(image, method=denoise_method)

    return image.astype(np.float32) / 255.0


def preprocess_image_path(
    image_path: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    skip_cut_off: bool = True,
    denoise_method: str = "median",
    **clahe_kwargs,
) -> np.ndarray | None:
    """Read an image path and return the preprocessed image, or ``None``."""
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    if not skip_cut_off and is_cut_off(image):
        return None

    return preprocess_image(
        image,
        image_size=image_size,
        denoise_method=denoise_method,
        **clahe_kwargs,
    )
