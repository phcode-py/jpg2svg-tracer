import cv2
import numpy as np


def load_and_binarize(
    path: str,
    threshold: int | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an image and binarize it to pure black-and-white.

    Args:
        path: Path to the input image.
        threshold: Manual threshold value 0-255. If None, Otsu's method is used.

    Returns:
        Tuple of (binary_image, (height, width)) where binary_image is a uint8
        ndarray with values in {0, 255}, and white pixels represent foreground
        features (lines/edges) to be traced.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
        ValueError: If the image is blank after binarization.
    """
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not load image: {path!r}")

    height, width = gray.shape

    if threshold is None:
        flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        thresh_val = 0
    else:
        flags = cv2.THRESH_BINARY
        thresh_val = int(threshold)

    _, binary = cv2.threshold(gray, thresh_val, 255, flags)

    # Orientation heuristic: findContours expects features as white on black.
    # If the image has more white than black (typical light-background line art),
    # the features (lines) are dark — invert so they become white.
    white_pixels = int(np.count_nonzero(binary))
    total_pixels = binary.size
    if white_pixels > total_pixels // 2:
        binary = cv2.bitwise_not(binary)

    if np.count_nonzero(binary) == 0:
        raise ValueError(
            "Image appears to be blank after binarization. "
            "Try adjusting --threshold or check the input image."
        )

    return binary, (height, width)
