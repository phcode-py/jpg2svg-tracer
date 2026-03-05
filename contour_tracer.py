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
    h, w = binary.shape[:2]
    image_area = float(h * w)
    raw, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    cv_contours, float_contours = [], []
    for c in raw:
        area = cv2.contourArea(c)
        if area < min_contour_area:
            continue
        if area > 0.9 * image_area:
            continue  # discard background boundary contour
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


def _rdp_to_budget(points: np.ndarray, target_n: int, eps_min: float = 0.0) -> np.ndarray:
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

    hi = max(hi, eps_min)
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
    eps_min: float = 0.0,
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
        if budget < 2:
            continue   # contour's share of the budget is too small

        if len(fc) <= budget:
            simp = fc.copy()
        else:
            simp = _rdp_to_budget_open(fc, budget, eps_min=eps_min)
            # Estimate this contour's epsilon for reporting (hi after convergence).
            pts_cv = fc.astype(np.float32).reshape(-1, 1, 2)
            span = float(np.linalg.norm(fc.max(axis=0) - fc.min(axis=0)))
            lo2, hi2 = 0.0, max(span, 1.0)
            for _ in range(30):
                mid = (lo2 + hi2) / 2
                if len(cv2.approxPolyDP(pts_cv, mid, closed=False)) <= budget:
                    hi2 = mid
                else:
                    lo2 = mid
            max_epsilon = max(max_epsilon, max(hi2, eps_min))

        if len(simp) < 2:
            continue

        simplified_contours.append(simp)
        n = len(fc)
        total_loss += _compute_loss(fc, simp) * n
        total_original_points += n

    weighted_loss = total_loss / total_original_points if total_original_points > 0 else 0.0
    return simplified_contours, max_epsilon, weighted_loss


def _rdp_to_budget_open(points: np.ndarray, target_n: int, eps_min: float = 0.0) -> np.ndarray:
    """RDP-simplify an open path to approximately target_n points (minimum 2)."""
    target_n = max(2, target_n)
    if len(points) <= target_n:
        return points.copy()
    span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    lo, hi = 0.0, max(span, 1.0)
    pts_cv = points.astype(np.float32).reshape(-1, 1, 2)
    for _ in range(30):
        mid = (lo + hi) / 2
        if len(cv2.approxPolyDP(pts_cv, mid, closed=False)) <= target_n:
            hi = mid
        else:
            lo = mid
    hi = max(hi, eps_min)
    result = cv2.approxPolyDP(pts_cv, hi, closed=False).reshape(-1, 2).astype(np.float64)
    if len(result) < 2:
        result = cv2.approxPolyDP(pts_cv, lo, closed=False).reshape(-1, 2).astype(np.float64)
    return result


def find_skeleton_paths(
    binary: np.ndarray,
    max_points: int = 500,
    min_contour_area: float = 10.0,
    contour_smooth: float = 0.0,
    simplify: str = "rdp",
    eps_min: float = 0.0,
) -> tuple[list[SimplifiedContour], float, float]:
    """Extract and simplify centerline skeleton paths from a binary image.

    Uses Zhang-Suen skeletonization to convert thick strokes to 1-pixel-wide
    centerlines, then simplifies them with proportional budget allocation.
    Eliminates the "webbing" artifact at intersections that occurs when tracing
    perimeter contours of thick lines with RDP.

    Args:
        binary: uint8 (H, W) array with foreground as white.
        max_points: Maximum total simplified points across all paths.
        min_contour_area: Paths with fewer than sqrt(min_contour_area) raw
            pixels are discarded (analogous to area filtering for contours).
        contour_smooth: Gaussian sigma applied to raw skeleton paths (px).
        simplify: "rdp" (straight-line segments) or "vw" (Catmull-Rom curves).

    Returns:
        (simplified_paths, metric, weighted_loss):
          - simplified_paths: List of open (N, 2) float64 path arrays.
          - metric: Largest RDP epsilon or VW triangle area used (display only).
          - weighted_loss: Weighted mean deviation from raw skeleton paths.
    """
    from image_processing import skeleton_paths as _skel_paths

    min_px = max(4, int(min_contour_area ** 0.5))
    raw_paths = _skel_paths(binary, min_path_pixels=min_px)
    if not raw_paths:
        return [], 0.0, 0.0

    if contour_smooth > 0.0:
        raw_paths = [_smooth_contour(p, contour_smooth, closed=False) for p in raw_paths]

    raw_lengths = [len(p) for p in raw_paths]
    total_raw = sum(raw_lengths)

    simplified: list[SimplifiedContour] = []
    total_loss = 0.0
    total_original = 0
    global_metric = 0.0

    for path, raw_len in zip(raw_paths, raw_lengths):
        budget = round(max_points * raw_len / total_raw)
        if budget < 2:
            continue

        if simplify == "rdp":
            if len(path) <= budget:
                simp = path.copy()
                eps = 0.0
            else:
                simp = _rdp_to_budget_open(path, budget, eps_min=eps_min)
                # Estimate per-path epsilon for reporting.
                span = float(np.linalg.norm(path.max(axis=0) - path.min(axis=0)))
                lo2, hi2 = 0.0, max(span, 1.0)
                pts_cv = path.astype(np.float32).reshape(-1, 1, 2)
                for _ in range(30):
                    mid = (lo2 + hi2) / 2
                    if len(cv2.approxPolyDP(pts_cv, mid, closed=False)) <= budget:
                        hi2 = mid
                    else:
                        lo2 = mid
                eps = max(hi2, eps_min)
            global_metric = max(global_metric, eps)
            if len(simp) < 2:
                continue
        else:  # vw
            simp, min_area = _visvalingam_whyatt(path, budget, closed=False)
            global_metric = max(global_metric, min_area)
            if len(simp) < 2:
                continue

        simplified.append(simp)
        n = len(path)
        total_loss += _compute_loss(path, simp) * n
        total_original += n

    weighted_loss = total_loss / total_original if total_original > 0 else 0.0
    return simplified, global_metric, weighted_loss


def find_arch_paths(
    binary: np.ndarray,
    max_points: int = 500,
    min_contour_area: float = 10.0,
    straight_threshold: float = 1.0,
    eps_min: float = 0.0,
) -> tuple[list[SimplifiedContour], list[SimplifiedContour], float, float]:
    """Geometry-based two-pass arch mode simplification.

    Skeletonizes the image once, then classifies each branch path by how
    straight it is: paths whose maximum orthogonal deviation from the chord
    joining their endpoints is < straight_threshold are encoded as exactly
    two points (start + end).  The remaining budget is spent on curved /
    detail paths simplified with RDP.

    No retracing: each raw path goes to exactly one pass.
    No point explosion: skeleton_paths() splits at junctions, so junction
    pixels only ever appear as path endpoints, never duplicated mid-path.

    Args:
        binary:             uint8 (H, W) foreground-white image.
        max_points:         Total point budget across both passes.
        min_contour_area:   Paths with < sqrt(min_contour_area) raw pixels
                            are discarded.
        straight_threshold: Max orthogonal deviation (px) for a path to be
                            classified as straight.  Raise for noisier images.
        eps_min:            Minimum RDP epsilon for the curved pass.

    Returns:
        (straight_paths, curved_paths, max_rdp_epsilon, weighted_loss)
    """
    from image_processing import skeleton_paths as _skel_paths

    min_px = max(4, int(min_contour_area ** 0.5))
    raw_paths = _skel_paths(binary, min_path_pixels=min_px)
    if not raw_paths:
        return [], [], 0.0, 0.0

    # Classify each raw path by maximum orthogonal deviation from its chord.
    straight_raw: list[np.ndarray] = []
    curved_raw: list[np.ndarray] = []
    thresh_sq = straight_threshold ** 2

    for path in raw_paths:
        if len(path) < 3:
            straight_raw.append(path)
            continue
        p0, p1 = path[0], path[-1]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            # Degenerate (closed loop): curved.
            curved_raw.append(path)
            continue
        seg_unit = seg / seg_len
        vecs = path - p0
        proj = (vecs * seg_unit).sum(axis=1, keepdims=True) * seg_unit
        max_dev_sq = float(np.max(np.sum((vecs - proj) ** 2, axis=1)))
        if max_dev_sq < thresh_sq:
            straight_raw.append(path)
        else:
            curved_raw.append(path)

    # Pass 1: straight paths → exactly 2 points each (start + end).
    straight_simplified: list[SimplifiedContour] = [
        np.array([p[0], p[-1]], dtype=np.float64) for p in straight_raw
    ]
    curved_budget = max(0, max_points - 2 * len(straight_simplified))

    if not curved_raw or curved_budget < 2:
        return straight_simplified, [], 0.0, 0.0

    # Pass 2: proportional RDP on curved paths with the remaining budget.
    raw_lens = [len(p) for p in curved_raw]
    total_raw = sum(raw_lens)

    curved_simplified: list[SimplifiedContour] = []
    max_eps = 0.0
    tot_loss = 0.0
    tot_orig = 0

    for path, rlen in zip(curved_raw, raw_lens):
        per_budget = max(2, round(curved_budget * rlen / total_raw))
        if len(path) <= per_budget:
            simp = path.copy()
            eps = 0.0
        else:
            simp = _rdp_to_budget_open(path, per_budget, eps_min=eps_min)
            pts_cv = path.astype(np.float32).reshape(-1, 1, 2)
            span = float(np.linalg.norm(path.max(axis=0) - path.min(axis=0)))
            lo2, hi2 = 0.0, max(span, 1.0)
            for _ in range(30):
                mid = (lo2 + hi2) / 2
                if len(cv2.approxPolyDP(pts_cv, mid, closed=False)) <= per_budget:
                    hi2 = mid
                else:
                    lo2 = mid
            eps = max(hi2, eps_min)
        if len(simp) < 2:
            continue
        curved_simplified.append(simp)
        max_eps = max(max_eps, eps)
        n = len(path)
        tot_loss += _compute_loss(path, simp) * n
        tot_orig += n

    weighted_loss = tot_loss / tot_orig if tot_orig > 0 else 0.0
    return straight_simplified, curved_simplified, max_eps, weighted_loss
