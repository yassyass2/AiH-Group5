import cv2
import numpy as np
from imblearn.over_sampling import SMOTE

DEFAULT_IMAGE_SIZE = 224  # For EfficientNet-B0


def crop_black_border(image, threshold=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold

    if not mask.any():
        return image

    y_indices, x_indices = np.where(mask)
    y1, y2 = y_indices.min(), y_indices.max()
    x1, x2 = x_indices.min(), x_indices.max()

    return image[y1:y2 + 1, x1:x2 + 1]


def reduce_fundus_noise(image, method="bilateral"):
    """
    Mild noise reduction for fundus images.

    Use conservative denoising because DR signs like microaneurysms,
    hemorrhages and exudates can be very small.
    """

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if method == "bilateral":
        # Best default: smooths noise while preserving edges better
        # than Gaussian blur.
        return cv2.bilateralFilter(
            image,
            d=5,
            sigmaColor=35,
            sigmaSpace=35,
        )

    if method == "nlmeans":
        # Stronger but slower.
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=3,
            hColor=3,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    if method == "median":
        # Useful for salt-and-pepper noise, but can remove tiny details.
        return cv2.medianBlur(image, 3)

    if method == "none":
        return image

    raise ValueError(f"Unknown denoising method: {method}")


def apply_green_channel_clahe(image):
    """
    Extract green channel and apply CLAHE.

    Returns a 3-channel green-tinted image so EfficientNet-B0 still receives
    shape (224, 224, 3).
    """

    green = image[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    green_enhanced = clahe.apply(green)
    zeros = np.zeros_like(green_enhanced)

    return cv2.merge([zeros, green_enhanced, zeros]).astype(np.uint8)


def preprocess_image(image, image_size=DEFAULT_IMAGE_SIZE, denoise_method="bilateral"):
    """
    Full preprocessing pipeline:

    1. Crop black border
    2. Resize to EfficientNet-B0 input size
    3. Apply mild noise reduction
    4. Extract green channel
    5. Apply CLAHE
    """

    image = crop_black_border(image)

    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    image = reduce_fundus_noise(image, method=denoise_method)

    image = apply_green_channel_clahe(image)

    return image


def apply_oversampling(X_train, y_train, random_state=42):
    """
    Balance the TRAINING set with SMOTE.

    Important:
    SMOTE returns float data, so we clip and convert back to uint8 after
    reshaping. This keeps your saved .npy arrays compatible with the validator.
    """

    n, h, w, c = X_train.shape

    X_flat = X_train.reshape(n, h * w * c).astype(np.float32)

    smote = SMOTE(random_state=random_state)
    X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y_train)

    X_resampled = X_resampled_flat.reshape(-1, h, w, c)

    X_resampled = np.clip(X_resampled, 0, 255).astype(np.uint8)
    y_resampled = y_resampled.astype(y_train.dtype)

    return X_resampled, y_resampled
