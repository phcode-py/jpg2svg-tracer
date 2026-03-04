import numpy as np


def _ctrl_pt_deviation(c1: np.ndarray, c2: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    """Max perpendicular distance of Bezier control points c1, c2 from chord p0→p1.

    Used to decide whether a cubic Bezier segment is visually straight enough
    to be replaced with a line command.
    """
    chord = p1 - p0
    length_sq = float(np.dot(chord, chord))
    if length_sq < 1e-10:
        # Degenerate zero-length chord: use direct distance to p0.
        return float(max(np.linalg.norm(c1 - p0), np.linalg.norm(c2 - p0)))

    def perp_dist(pt: np.ndarray) -> float:
        t = float(np.dot(pt - p0, chord) / length_sq)
        t = max(0.0, min(1.0, t))
        return float(np.linalg.norm(pt - (p0 + t * chord)))

    return max(perp_dist(c1), perp_dist(c2))


def _catmull_rom_controls(
    points: np.ndarray,
    tension: float,
    closed: bool,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Compute cubic Bezier control points via Catmull-Rom interpolation.

    For each segment from points[i] to points[i+1], derives two control points
    using the Catmull-Rom formula:
        C1 = P[i]   + (P[i+1] - P[i-1]) * tension / 2
        C2 = P[i+1] - (P[i+2] - P[i])   * tension / 2

    At tension=0.5 this matches the classic Catmull-Rom spline.
    At tension=0 the control points collapse to the anchors (polyline).

    Args:
        points: (N, 2) float64 array of (x, y) coordinates.
        tension: Smoothness parameter in [0, 1].
        closed: If True, wrap indices (last segment connects back to first).

    Returns:
        List of (P_i, C1, C2, P_next) tuples, one per segment.
    """
    n = len(points)
    segments = []
    num_segments = n if closed else n - 1

    for i in range(num_segments):
        if closed:
            p_prev = points[(i - 1) % n]
            p_cur  = points[i % n]
            p_next = points[(i + 1) % n]
            p_next2 = points[(i + 2) % n]
        else:
            # Duplicate endpoints so the curve is tangent to the endpoint direction.
            p_prev  = points[0]      if i == 0     else points[i - 1]
            p_cur   = points[i]
            p_next  = points[i + 1]
            p_next2 = points[n - 1]  if i >= n - 2 else points[i + 2]

        c1 = p_cur  + (p_next  - p_prev)  * (tension / 2.0)
        c2 = p_next - (p_next2 - p_cur)   * (tension / 2.0)

        segments.append((p_cur, c1, c2, p_next))

    return segments


def points_to_svg_path(
    points: np.ndarray,
    tension: float = 0.5,
    closed: bool = True,
    precision: int = 3,
    straight_threshold: float = 1.0,
) -> str:
    """Convert a sequence of 2D points to an SVG path `d` attribute string.

    Uses Catmull-Rom interpolation for curved segments and falls back to straight
    line commands (L) where the control points deviate less than `straight_threshold`
    pixels from the chord. This prevents artificial bending of straight or nearly-
    straight sections when the point count is low.

    Args:
        points: (N, 2) float64 array.
        tension: Catmull-Rom tension; higher = tighter to the control polygon.
        closed: If True, the path ends with Z and wraps around.
        precision: Number of decimal places in coordinate output.
        straight_threshold: Max control-point deviation (px) below which a segment
            is emitted as a straight line. Set to 0 to always use cubic Bezier.

    Returns:
        SVG path `d` string, or empty string for zero-point input.
    """
    n = len(points)

    if n == 0:
        return ""

    fmt = f".{precision}f"

    if n == 1:
        x, y = points[0]
        return f"M {x:{fmt}},{y:{fmt}}"

    if n == 2:
        x0, y0 = points[0]
        x1, y1 = points[1]
        suffix = " Z" if closed else ""
        return f"M {x0:{fmt}},{y0:{fmt}} L {x1:{fmt}},{y1:{fmt}}{suffix}"

    segments = _catmull_rom_controls(points, tension=tension, closed=closed)

    x0, y0 = segments[0][0]
    parts = [f"M {x0:{fmt}},{y0:{fmt}}"]

    for p_cur, c1, c2, p_next in segments:
        px, py = p_next
        if straight_threshold > 0 and _ctrl_pt_deviation(c1, c2, p_cur, p_next) < straight_threshold:
            parts.append(f"L {px:{fmt}},{py:{fmt}}")
        else:
            cx1, cy1 = c1
            cx2, cy2 = c2
            parts.append(
                f"C {cx1:{fmt}},{cy1:{fmt}} {cx2:{fmt}},{cy2:{fmt}} {px:{fmt}},{py:{fmt}}"
            )

    if closed:
        parts.append("Z")

    return " ".join(parts)


def _arc_svg_commands(
    arc_points: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    precision: int,
) -> list[str]:
    """Generate SVG 'A' command string(s) for a circular arc segment.

    The caller is assumed to have already moved to arc_points[0].
    The A command(s) drive the path to arc_points[-1].

    The total angular sweep is computed by summing arctan2(cross, dot) for each
    consecutive pair of vectors from the centre. This gives the true cumulative
    sweep even when start and end are almost at the same angle (near-full circles),
    unlike comparing atan2(start) vs atan2(end) which collapses to ~0° for a 359° arc.

    Arcs spanning more than ~335° are split into two semicircles because SVG arc
    commands become numerically degenerate when start ≈ end.
    """
    fmt = f".{precision}f"
    start = arc_points[0]
    end   = arc_points[-1]

    # Vectors from centre to each point on the arc.
    vecs  = arc_points - np.array([cx, cy])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit  = vecs / np.maximum(norms, 1e-10)

    # Angle change at each consecutive step — exact via arctan2, no wrapping issues.
    cos_steps = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
    sin_steps = unit[:-1, 0] * unit[1:, 1] - unit[:-1, 1] * unit[1:, 0]
    total_angle = float(np.sum(np.arctan2(sin_steps, cos_steps)))

    # In SVG Y-down coords: positive total_angle → CW visually → sweep=1.
    sweep     = 1 if total_angle > 0 else 0
    abs_angle = abs(total_angle)

    # Near-full-circle: SVG A is degenerate when start ≈ end.  Split at midpoint.
    if abs_angle > 5.85:   # > ~335°
        theta0 = np.arctan2(start[1] - cy, start[0] - cx)
        sign   = 1.0 if total_angle > 0 else -1.0
        mid_x  = float(cx + r * np.cos(theta0 + sign * np.pi))
        mid_y  = float(cy + r * np.sin(theta0 + sign * np.pi))
        return [
            f"A {r:{fmt}},{r:{fmt}} 0 1,{sweep} {mid_x:{fmt}},{mid_y:{fmt}}",
            f"A {r:{fmt}},{r:{fmt}} 0 1,{sweep} {end[0]:{fmt}},{end[1]:{fmt}}",
        ]

    large_arc = 1 if abs_angle > np.pi else 0
    ex, ey = end
    return [f"A {r:{fmt}},{r:{fmt}} 0 {large_arc},{sweep} {ex:{fmt}},{ey:{fmt}}"]


def _segments_to_svg_path(
    segments: list[tuple],
    tension: float,
    precision: int,
    straight_threshold: float,
) -> str:
    """Build a closed SVG path d string from mixed arc/poly segments.

    Each segment is either:
      ('arc',  pts, cx, cy, r)  → one or two A commands
      ('poly', pts)              → Catmull-Rom C/L commands

    Adjacent segments share their connecting point, so the path is gapless.
    The path is always closed with Z.
    """
    if not segments:
        return ""

    fmt = f".{precision}f"
    x0, y0 = segments[0][1][0]   # first point of first segment
    parts   = [f"M {x0:{fmt}},{y0:{fmt}}"]

    for seg in segments:
        if seg[0] == "arc":
            _, arc_pts, cx, cy, r = seg
            parts.extend(_arc_svg_commands(arc_pts, cx, cy, r, precision))

        else:  # 'poly'
            poly_pts = seg[1]
            if len(poly_pts) < 2:
                continue

            if len(poly_pts) == 2:
                x, y = poly_pts[1]
                parts.append(f"L {x:{fmt}},{y:{fmt}}")
                continue

            # Catmull-Rom on the open sub-segment.
            # closed=False: the sub-segment is a piece of the full closed path,
            # not a closed loop itself.
            sub_segs = _catmull_rom_controls(poly_pts, tension=tension, closed=False)
            for p_cur, c1, c2, p_next in sub_segs:
                px, py = p_next
                if (straight_threshold > 0
                        and _ctrl_pt_deviation(c1, c2, p_cur, p_next) < straight_threshold):
                    parts.append(f"L {px:{fmt}},{py:{fmt}}")
                else:
                    cx1, cy1 = c1
                    cx2, cy2 = c2
                    parts.append(
                        f"C {cx1:{fmt}},{cy1:{fmt}} {cx2:{fmt}},{cy2:{fmt}} {px:{fmt}},{py:{fmt}}"
                    )

    parts.append("Z")
    return " ".join(parts)


def contours_to_svg_paths(
    contours: list[np.ndarray],
    tension: float = 0.5,
    precision: int = 3,
    straight_threshold: float = 1.0,
    arc_tolerance: float | None = None,
    arc_min_points: int = 4,
    arc_min_radius: float = 3.0,
) -> list[str]:
    """Convert a list of simplified contours to SVG path strings.

    When arc_tolerance is given, circular arc segments in each contour are
    replaced with exact SVG 'A' commands instead of Catmull-Rom approximations.
    Pass arc_tolerance=None (or use --no-arcs) to use the pure Catmull-Rom path.

    Args:
        contours: List of (N, 2) float64 arrays.
        tension: Catmull-Rom tension for non-arc segments.
        precision: Decimal places for coordinate output.
        straight_threshold: Max control-point deviation (px) to use L instead of C.
        arc_tolerance: Max mean residual (px) to accept a circle fit. None = disabled.
        arc_min_points: Minimum VW points needed to trigger arc detection.
        arc_min_radius: Circles below this radius (px) are not replaced with arcs.

    Returns:
        List of non-empty SVG `d` strings.
    """
    if arc_tolerance is not None:
        from arc_detector import segment_contour

    paths = []
    for contour in contours:
        if arc_tolerance is not None and len(contour) >= arc_min_points:
            segs = segment_contour(
                contour,
                tolerance=arc_tolerance,
                min_arc_points=arc_min_points,
                min_radius=arc_min_radius,
            )
            d = _segments_to_svg_path(segs, tension, precision, straight_threshold)
        else:
            d = points_to_svg_path(
                contour,
                tension=tension,
                closed=True,
                precision=precision,
                straight_threshold=straight_threshold,
            )
        if d:
            paths.append(d)
    return paths
