"""Detect circular arc segments in a sequence of 2D points.

Works on VW-simplified contour points. Tries to replace runs of points that
lie on a circle with a single arc description, drastically reducing the number
of SVG path commands needed for circular features.
"""

import numpy as np


def _fit_circle_kasa(points: np.ndarray):
    """Kåsa's algebraic circle fit (linear least-squares, O(N)).

    Solves for (cx, cy, r) by rewriting (x-cx)²+(y-cy)²=r² as a linear system.

    Returns (cx, cy, r) or None if the system is degenerate (e.g. collinear points).
    """
    if len(points) < 3:
        return None

    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = -(x ** 2 + y ** 2)

    try:
        coeffs, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if rank < 3:
        return None  # degenerate (collinear points give rank < 3)

    a_c, b_c, c_c = coeffs
    cx = -a_c / 2.0
    cy = -b_c / 2.0
    r_sq = cx ** 2 + cy ** 2 - c_c

    if r_sq <= 0 or not np.isfinite(r_sq):
        return None

    return float(cx), float(cy), float(np.sqrt(r_sq))


def _circle_residual(points: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Mean absolute deviation of points from a circle."""
    d = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    return float(np.mean(np.abs(d - r)))


def segment_contour(
    points: np.ndarray,
    tolerance: float,
    min_arc_points: int = 4,
    min_radius: float = 3.0,
    max_radius: float = float("inf"),
) -> list[tuple]:
    """Greedily partition a contour into arc and polyline segments.

    First checks if the *entire* contour is a circle (most common case for
    icon images where circles are separate closed contours). Then does a
    forward-greedy pass to detect partial arcs in mixed contours.

    Each segment in the returned list is one of:
      ('arc',  pts, cx, cy, r)  — pts includes start and end points
      ('poly', pts)              — to be Bezier-interpolated by the caller

    Adjacent segments share their connecting point (last point of segment N ==
    first point of segment N+1), so the full contour is covered without gaps.

    Args:
        points: (N, 2) float64 VW-simplified contour.
        tolerance: Max mean absolute residual (px) to accept a circle fit.
        min_arc_points: Minimum points required to form an arc.
        min_radius: Arcs with radius < this (px) are rejected (too tight).
        max_radius: Arcs with radius > this (px) are rejected (near-straight).
    """
    n = len(points)
    if n < min_arc_points:
        return [("poly", points)]

    def _good(pts, cx, cy, r):
        return (
            np.isfinite(r)
            and min_radius <= r <= max_radius
            and _circle_residual(pts, cx, cy, r) <= tolerance
        )

    # ---- Fast path: entire contour is a circle ----
    fit = _fit_circle_kasa(points)
    if fit is not None:
        cx, cy, r = fit
        if _good(points, cx, cy, r):
            return [("arc", points, cx, cy, r)]

    # ---- Greedy forward pass ----
    segments: list[tuple] = []
    poly_start = 0
    i = 0

    while i <= n - min_arc_points:
        # Try fitting a circle to the minimum window starting at i.
        window = points[i : i + min_arc_points]
        fit = _fit_circle_kasa(window)
        if fit is None or not _good(window, *fit):
            i += 1
            continue

        cx, cy, r = fit
        best_j = i + min_arc_points - 1  # inclusive end index

        # Extend the arc window as far as the fit stays within tolerance.
        for j in range(i + min_arc_points, n):
            extended = points[i : j + 1]
            fit_ext = _fit_circle_kasa(extended)
            if fit_ext is not None and _good(extended, *fit_ext):
                cx, cy, r = fit_ext
                best_j = j
            else:
                break

        # Flush any pending polyline up to (and including) the arc start.
        if i > poly_start:
            segments.append(("poly", points[poly_start : i + 1]))

        segments.append(("arc", points[i : best_j + 1], cx, cy, r))
        i = best_j
        poly_start = best_j

    # Flush the remaining polyline tail.
    if poly_start < n:
        segments.append(("poly", points[poly_start:]))

    return segments
