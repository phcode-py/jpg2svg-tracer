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

    # Remove isolated singular pixels (no 8-connected neighbors) — pure noise.
    fg = (binary > 0).astype(np.uint8)
    padded = np.pad(fg, 1, constant_values=0)
    neighbor_sum = (
        padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
        padded[1:-1, 0:-2]                        + padded[1:-1, 2:] +
        padded[2:,   0:-2] + padded[2:,   1:-1]   + padded[2:,   2:]
    )
    binary[fg.astype(bool) & (neighbor_sum == 0)] = 0

    return binary, (height, width)


# ---------------------------------------------------------------------------
# Skeletonization (centerline extraction)
# ---------------------------------------------------------------------------

def _zhang_suen_thin(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen (1984) iterative thinning to 1-pixel-wide skeleton.

    Reduces white foreground regions to 1-pixel-wide centerlines while
    preserving connectivity. The skeleton approximates the medial axis.
    """
    img = (binary > 0).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            pad = np.pad(img, 1, constant_values=0)
            # 8-neighbors in clockwise order starting from North: P2..P9
            nb = [
                pad[0:-2, 1:-1],  # P2 N
                pad[0:-2, 2:],    # P3 NE
                pad[1:-1, 2:],    # P4 E
                pad[2:,   2:],    # P5 SE
                pad[2:,   1:-1],  # P6 S
                pad[2:,   0:-2],  # P7 SW
                pad[1:-1, 0:-2],  # P8 W
                pad[0:-2, 0:-2],  # P9 NW
            ]
            B = sum(nb)   # number of white neighbors
            # A = number of 0→1 transitions in the cyclic neighbor sequence
            A = sum((nb[i] == 0) & (nb[(i + 1) % 8] == 1) for i in range(8))
            base = (img == 1) & (B >= 2) & (B <= 6) & (A == 1)
            if step == 0:
                to_remove = base & (nb[0] * nb[2] * nb[4] == 0) & (nb[2] * nb[4] * nb[6] == 0)
            else:
                to_remove = base & (nb[0] * nb[2] * nb[6] == 0) & (nb[0] * nb[4] * nb[6] == 0)
            if to_remove.any():
                changed = True
                img[to_remove] = 0
        if not changed:
            break
    return img * 255


def skeleton_paths(
    binary: np.ndarray,
    min_path_pixels: int = 4,
) -> list[np.ndarray]:
    """Skeletonize a binary image and extract centerline paths.

    Uses Zhang-Suen thinning to reduce thick strokes to 1-pixel-wide
    centerlines, then traces each branch into an ordered (x, y) sequence.
    Junction pixels (degree ≥ 3) act as branch endpoints so that T- and
    X-intersections produce separate, clean line segments rather than a
    single tangled path.

    Args:
        binary: uint8 (H, W) array; white (255) = foreground strokes.
        min_path_pixels: Discard paths shorter than this many raw pixels.

    Returns:
        List of (N, 2) float64 arrays with (x, y) coordinates, one per
        branch. Paths are open polylines (not closed).
    """
    skel = _zhang_suen_thin(binary) > 0
    h, w = skel.shape
    if not skel.any():
        return []

    # Degree = number of 8-connected skeleton neighbors for each pixel.
    padded = np.pad(skel.astype(np.uint8), 1, constant_values=0)
    n_count = (
        padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
        padded[1:-1, 0:-2]                        + padded[1:-1, 2:] +
        padded[2:,   0:-2] + padded[2:,   1:-1]   + padded[2:,   2:]
    )
    n_count = n_count * skel  # zero out non-skeleton pixels

    junction_mask = (n_count >= 3)               # degree ≥ 3
    branch_mask = skel & ~junction_mask           # skeleton minus junctions

    # Connected components of branch pixels (cv2 supports 8-connectivity).
    n_labels, labeled = cv2.connectedComponents(
        branch_mask.astype(np.uint8), connectivity=8
    )

    result = []
    for cid in range(1, n_labels):
        ys, xs = np.where(labeled == cid)
        if len(ys) < min_path_pixels:
            continue

        comp_set: set[tuple[int, int]] = set(zip(ys.tolist(), xs.tolist()))

        def nbrs_comp(r: int, c: int) -> list[tuple[int, int]]:
            return [
                (r + dr, c + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (r + dr, c + dc) in comp_set
            ]

        # Start tracing from a degree-≤1 pixel (endpoint); fall back to any
        # pixel for closed loops where every pixel has exactly 2 neighbors.
        start = next(
            (px for px in comp_set if len(nbrs_comp(*px)) <= 1),
            (int(ys[0]), int(xs[0])),
        )

        visited: set[tuple[int, int]] = {start}
        path: list[tuple[int, int]] = [start]
        cur = start
        while True:
            nexts = [n for n in nbrs_comp(*cur) if n not in visited]
            if not nexts:
                break
            cur = nexts[0]
            visited.add(cur)
            path.append(cur)

        # Extend both endpoints to include the nearest adjacent junction pixel
        # so that branches connect at their shared junction vertex.
        def adj_junc(r: int, c: int) -> tuple[int, int] | None:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr, dc) == (0, 0):
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and junction_mask[nr, nc]:
                        return (nr, nc)
            return None

        j = adj_junc(*path[0])
        if j:
            path = [j] + path
        j = adj_junc(*path[-1])
        if j:
            path = path + [j]

        if len(path) < min_path_pixels:
            continue

        # Convert (row, col) → (x, y) = (col, row)
        result.append(np.array([(c, r) for r, c in path], dtype=np.float64))

    return result
