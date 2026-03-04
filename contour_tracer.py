import heapq
import warnings

import cv2
import numpy as np

try:
    from scipy.spatial import cKDTree as KDTree
    from scipy.ndimage import gaussian_filter1d
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# Type alias: a simplified contour is a (N, 2) float64 array of (x, y) points.
SimplifiedContour = np.ndarray


def _smooth_contour(points: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    """Gaussian-smooth contour coordinates to remove pixel-grid staircase artifacts."""
    if sigma <= 0.0:
        return points

    x, y = points[:, 0], points[:, 1]

    if _SCIPY_AVAILABLE:
        mode = "wrap" if closed else "reflect"
        x_smooth = gaussian_filter1d(x, sigma, mode=mode)
        y_smooth = gaussian_filter1d(y, sigma, mode=mode)
    else:
        radius = max(1, int(3 * sigma))
        k = np.arange(2 * radius + 1) - radius
        kernel = np.exp(-0.5 * (k / sigma) ** 2)
        kernel /= kernel.sum()
        if closed:
            x_pad = np.concatenate([x[-radius:], x, x[:radius]])
            y_pad = np.concatenate([y[-radius:], y, y[:radius]])
            x_smooth = np.convolve(x_pad, kernel, mode="valid")
            y_smooth = np.convolve(y_pad, kernel, mode="valid")
        else:
            x_smooth = np.convolve(x, kernel, mode="same")
            y_smooth = np.convolve(y, kernel, mode="same")

    return np.stack([x_smooth, y_smooth], axis=1)


def _visvalingam_whyatt(
    points: np.ndarray,
    target_n: int,
    closed: bool = True,
) -> np.ndarray:
    """Simplify a polyline/polygon using the Visvalingam-Whyatt algorithm.

    Unlike Douglas-Peucker (which uses a single global deviation threshold),
    VW removes points greedily by the area of the triangle each point forms
    with its two neighbors — smallest area first. This naturally preserves
    high-curvature regions (large triangle area) while collapsing nearly-
    collinear runs (small triangle area), giving better fidelity on curves
    without spending points on straight edges.

    Args:
        points: (N, 2) float64 array.
        target_n: Desired number of output points.
        closed: If True, first and last points are neighbors (closed polygon).

    Returns:
        Simplified (M, 2) array where M ≈ target_n.
    """
    n = len(points)
    min_n = 3 if closed else 2
    target_n = max(target_n, min_n)
    if n <= target_n:
        return points.copy(), 0.0

    # Doubly-linked list via mutable Python lists (O(1) neighbor updates).
    prev = list(range(-1, n - 1))
    nxt  = list(range(1, n + 1))
    if closed:
        prev[0] = n - 1
        nxt[n - 1] = 0
    else:
        # Endpoints are self-referential; their area is set to inf below.
        prev[0] = 0
        nxt[n - 1] = n - 1

    def tri_area(i: int) -> float:
        ax, ay = points[prev[i]]
        bx, by = points[i]
        cx, cy = points[nxt[i]]
        return abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) * 0.5

    alive = [True] * n
    area  = [tri_area(i) for i in range(n)]

    if not closed:
        area[0] = area[n - 1] = float("inf")

    # Min-heap of (area, index).  Stale entries are skipped on pop.
    heap = [(a, i) for i, a in enumerate(area)]
    heapq.heapify(heap)

    current_n = n
    min_survived = 0.0  # monotonicity floor — area can only grow as points are removed

    while current_n > target_n and heap:
        a, i = heapq.heappop(heap)

        if not alive[i]:
            continue          # already removed
        if a < area[i] - 1e-12:
            continue          # stale entry; a newer, higher-area entry is in the heap

        # Enforce monotonicity: once we've removed a point of area X,
        # we treat all subsequent removals as having area ≥ X.
        min_survived = max(min_survived, area[i])

        alive[i] = False
        current_n -= 1

        p, nx = prev[i], nxt[i]
        nxt[p] = nx
        prev[nx] = p

        # Recompute triangle areas for the two neighbors, then re-push.
        for j in (p, nx):
            if alive[j] and (closed or (j != 0 and j != n - 1)):
                area[j] = max(tri_area(j), min_survived)
                heapq.heappush(heap, (area[j], j))

    return points[[i for i in range(n) if alive[i]]], min_survived


def find_raw_contours(
    binary: np.ndarray,
    min_contour_area: float = 10.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Find contours in a binary image and filter by area.

    Returns:
        Tuple of (cv_contours, float_contours):
          - cv_contours: OpenCV-native (N, 1, 2) int32 arrays.
          - float_contours: Corresponding (N, 2) float64 arrays.
    """
    raw, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    cv_contours, float_contours = [], []
    for c in raw:
        if cv2.contourArea(c) >= min_contour_area:
            cv_contours.append(c)
            float_contours.append(c.reshape(-1, 2).astype(np.float64))

    return cv_contours, float_contours


def _compute_loss(original: np.ndarray, simplified: np.ndarray) -> float:
    """Mean nearest-neighbor distance from original to simplified contour (one-sided)."""
    if len(simplified) == 0:
        return float("inf")

    if _SCIPY_AVAILABLE:
        tree = KDTree(simplified)
        distances, _ = tree.query(original)
        return float(np.mean(distances))

    chunk_size = 512
    total_dist = 0.0
    n = len(original)
    for start in range(0, n, chunk_size):
        chunk = original[start : start + chunk_size]
        diffs = chunk[:, None, :] - simplified[None, :, :]
        dist_sq = np.sum(diffs ** 2, axis=2)
        total_dist += float(np.sum(np.sqrt(np.min(dist_sq, axis=1))))
    return total_dist / n


def find_contours_with_budget(
    binary: np.ndarray,
    max_points: int = 500,
    min_contour_area: float = 10.0,
    contour_smooth: float = 1.5,
) -> tuple[list[SimplifiedContour], float, float]:
    """Detect, filter, and simplify contours to fit within a point budget.

    Uses Visvalingam-Whyatt simplification, which naturally preserves curved
    regions (high triangle area = high curvature) while aggressively collapsing
    straight edges. The point budget is allocated proportionally by raw contour
    length, so longer features get more resolution.

    Args:
        binary: uint8 ndarray (H, W) with foreground as white.
        max_points: Maximum total points across all output contours.
        min_contour_area: Contours with pixel area below this are discarded.
        contour_smooth: Gaussian sigma (px) applied before simplification to
            remove pixel-grid staircase. Set to 0 to disable.

    Returns:
        Tuple of (simplified_contours, min_triangle_area, loss):
          - simplified_contours: List of (N, 2) float64 arrays.
          - min_triangle_area: Smallest triangle area that survived (proxy for
            simplification aggressiveness; 0 means no simplification was needed).
          - loss: Weighted mean deviation from original contour in pixels.
    """
    cv_contours, float_contours = find_raw_contours(binary, min_contour_area)

    if not cv_contours:
        return [], 0.0, 0.0

    # Smooth to remove pixel-grid staircase.
    if contour_smooth > 0.0:
        smoothed_floats = [
            _smooth_contour(fc, contour_smooth, closed=True)
            for fc in float_contours
        ]
    else:
        smoothed_floats = float_contours

    # Allocate point budget proportionally by raw contour length (= perimeter).
    # Longer contours get more points; curved vs. straight is handled by VW itself.
    raw_lengths = [len(sf) for sf in smoothed_floats]
    total_raw = sum(raw_lengths)

    simplified_contours: list[SimplifiedContour] = []
    total_loss = 0.0
    total_original_points = 0
    min_area_global = 0.0

    for sf, orig_f, raw_len in zip(smoothed_floats, float_contours, raw_lengths):
        budget = max(3, round(max_points * raw_len / total_raw))
        simplified, min_area = _visvalingam_whyatt(sf, budget, closed=True)
        min_area_global = max(min_area_global, min_area)

        if len(simplified) < 2:
            continue

        simplified_contours.append(simplified)

        n = len(orig_f)
        loss = _compute_loss(orig_f, simplified)
        total_loss += loss * n
        total_original_points += n

    weighted_loss = total_loss / total_original_points if total_original_points > 0 else 0.0

    return simplified_contours, min_area_global, weighted_loss


def _rdp_contour(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Apply Ramer-Douglas-Peucker to a single (N, 2) contour."""
    pts_cv = points.astype(np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts_cv, epsilon, closed=True)
    return approx.reshape(-1, 2).astype(np.float64)


def _rdp_to_budget(points: np.ndarray, target_n: int) -> np.ndarray:
    """RDP-simplify a single contour to approximately target_n points.

    Binary-searches the per-contour epsilon using the contour's own bounding-box
    diagonal as the upper bound — much tighter than the image diagonal and avoids
    the global-epsilon failure mode where tiny contours collapse to <3 points.
    """
    target_n = max(3, target_n)
    if len(points) <= target_n:
        return points.copy()

    # Upper bound: bounding-box diagonal of this contour (not the whole image).
    span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    lo, hi = 0.0, max(span, 1.0)

    for _ in range(30):
        mid = (lo + hi) / 2
        if len(_rdp_contour(points, mid)) <= target_n:
            hi = mid
        else:
            lo = mid

    result = _rdp_contour(points, hi)
    # If hi collapsed too far (< 3 pts), fall back to lo which is guaranteed to
    # have enough points.
    if len(result) < 3:
        result = _rdp_contour(points, lo)
    return result


def find_contours_rdp(
    binary: np.ndarray,
    max_points: int = 500,
    min_contour_area: float = 10.0,
    contour_smooth: float = 0.0,
) -> tuple[list[SimplifiedContour], float, float]:
    """Detect and simplify contours using Ramer-Douglas-Peucker (RDP).

    Better suited than VW for images with predominantly straight lines.
    Allocates the point budget proportionally by raw contour length (identical
    strategy to find_contours_with_budget) and runs a per-contour binary search
    on epsilon — so every contour is guaranteed at least 3 points regardless of
    the global budget, eliminating the "no contours found" failure and the
    "confusing diagonal lines" artifact that occur with a single global epsilon.

    Args:
        binary: uint8 ndarray (H, W) with foreground as white.
        max_points: Maximum total points across all output contours.
        min_contour_area: Contours below this pixel area are discarded.
        contour_smooth: Gaussian sigma applied before RDP. Typically 0 for
            straight-line images (smoothing rounds corners).

    Returns:
        (simplified_contours, max_epsilon, weighted_loss) — same shape as
        find_contours_with_budget. max_epsilon is the largest per-contour
        RDP tolerance (px) used; 0 if no simplification was needed.
    """
    _, float_contours = find_raw_contours(binary, min_contour_area)

    if not float_contours:
        return [], 0.0, 0.0

    if contour_smooth > 0.0:
        float_contours = [
            _smooth_contour(fc, contour_smooth, closed=True) for fc in float_contours
        ]

    raw_lengths = [len(c) for c in float_contours]
    total_raw = sum(raw_lengths)

    simplified_contours: list[SimplifiedContour] = []
    total_loss = 0.0
    total_original_points = 0
    max_epsilon = 0.0

    for fc, raw_len in zip(float_contours, raw_lengths):
        budget = round(max_points * raw_len / total_raw)
        if budget < 3:
            continue   # contour's share of the budget is too small for a valid polygon

        if len(fc) <= budget:
            simp = fc.copy()
        else:
            simp = _rdp_to_budget(fc, budget)
            # Estimate this contour's epsilon for reporting (hi after convergence).
            span = float(np.linalg.norm(fc.max(axis=0) - fc.min(axis=0)))
            lo2, hi2 = 0.0, max(span, 1.0)
            for _ in range(30):
                mid = (lo2 + hi2) / 2
                if len(_rdp_contour(fc, mid)) <= budget:
                    hi2 = mid
                else:
                    lo2 = mid
            max_epsilon = max(max_epsilon, hi2)

        if len(simp) < 3:
            continue

        simplified_contours.append(simp)
        n = len(fc)
        total_loss += _compute_loss(fc, simp) * n
        total_original_points += n

    weighted_loss = total_loss / total_original_points if total_original_points > 0 else 0.0
    return simplified_contours, max_epsilon, weighted_loss
