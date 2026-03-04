"""Core tracing pipeline shared by CLI (main.py) and web server (server.py)."""

import warnings

from image_processing import load_and_binarize
from contour_tracer import find_contours_with_budget, find_contours_rdp
from bezier import contours_to_svg_paths
from svg_writer import build_svg


def _count_arc_savings(
    contours: list,
    arc_tolerance: float,
    arc_min_points: int = 4,
    arc_min_radius: float = 3.0,
) -> int:
    """Count VW points consumed by arc segments that are freed by SVG A commands."""
    from arc_detector import segment_contour

    freed = 0
    for contour in contours:
        if len(contour) < arc_min_points:
            continue
        segs = segment_contour(
            contour,
            tolerance=arc_tolerance,
            min_arc_points=arc_min_points,
            min_radius=arc_min_radius,
        )
        for seg in segs:
            if seg[0] == "arc":
                freed += max(0, len(seg[1]) - 1)
    return freed


def trace(
    image_path: str,
    max_points: int = 500,
    threshold: int | None = None,
    min_contour_area: float = 10.0,
    stroke_width: float = 2.0,
    tension: float = 0.5,
    contour_smooth: float = 1.5,
    straight_threshold: float = 1.0,
    arc_tolerance: float | None = 1.5,
    min_vw_points: int = 0,
    simplify: str = "vw",
) -> tuple[str, dict]:
    """Run the full tracing pipeline.

    Args:
        simplify: "vw" (Visvalingam-Whyatt + Catmull-Rom, default) or
                  "rdp" (Ramer-Douglas-Peucker + straight lines).
                  In RDP mode contour_smooth defaults to 0 unless explicitly set,
                  and straight_threshold is forced to infinity (all L commands).

    Returns:
        (svg_string, stats_dict)

    Raises:
        FileNotFoundError: Image not found or unreadable.
        ValueError: Image is blank after binarization.
    """
    if simplify not in ("vw", "rdp"):
        raise ValueError(f"simplify must be 'vw' or 'rdp', got {simplify!r}")

    binary, (height, width) = load_and_binarize(image_path, threshold=threshold)

    collected_warnings: list[str] = []

    # In RDP mode: force straight-line output regardless of straight_threshold arg.
    effective_straight_threshold = float("inf") if simplify == "rdp" else straight_threshold

    def _run(budget: int):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if simplify == "rdp":
                result = find_contours_rdp(
                    binary,
                    max_points=budget,
                    min_contour_area=min_contour_area,
                    contour_smooth=contour_smooth,
                )
            else:
                result = find_contours_with_budget(
                    binary,
                    max_points=budget,
                    min_contour_area=min_contour_area,
                    contour_smooth=contour_smooth,
                )
        collected_warnings.extend(str(w.message) for w in caught)
        return result

    simplified_contours, epsilon, loss = _run(max_points)

    freed = 0
    if arc_tolerance is not None and simplified_contours:
        freed = _count_arc_savings(simplified_contours, arc_tolerance)
        if freed > 0:
            simplified_contours, epsilon, loss = _run(max_points + freed)

    if min_vw_points > 0:
        simplified_contours = [c for c in simplified_contours if len(c) > min_vw_points]

    path_strings = contours_to_svg_paths(
        simplified_contours,
        tension=tension,
        straight_threshold=effective_straight_threshold,
        arc_tolerance=arc_tolerance,
    )

    svg_content = build_svg(
        path_strings,
        width=width,
        height=height,
        stroke_width=stroke_width,
    )

    arc_count = 0
    if arc_tolerance is not None:
        arc_count = sum(1 for p in path_strings if " A " in p or p.startswith("A "))

    stats = {
        "width": width,
        "height": height,
        "contours": len(simplified_contours),
        "points": sum(len(c) for c in simplified_contours),
        "epsilon": round(epsilon, 4),
        "loss": round(loss, 4),
        "freed": freed,
        "arc_count": arc_count,
        "simplify": simplify,
        "warnings": collected_warnings,
    }

    return svg_content, stats
