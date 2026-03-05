"""Core tracing pipeline shared by CLI (main.py) and web server (server.py)."""

import warnings

import cv2

from image_processing import load_and_binarize, _zhang_suen_thin
from contour_tracer import find_contours_with_budget, find_contours_rdp, find_skeleton_paths, find_arch_paths
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
    skeletonize: bool | None = None,
    thick_threshold: int | None = None,
    testing_prefix: str | None = None,
    eps_min: float = 0.0,
) -> tuple[str, dict]:
    """Run the full tracing pipeline.

    Args:
        simplify: "vw" (Visvalingam-Whyatt + Catmull-Rom, default),
                  "rdp" (Ramer-Douglas-Peucker + straight lines), or
                  "arch" (Architectural Plan: geometry-based two-pass —
                  skeletonize once, classify paths by straightness, encode
                  straight paths as 2-point lines, spend remaining budget on
                  curved/detail paths with RDP).
        skeletonize: If True, extract skeleton centerlines instead of perimeter
                  contours (eliminates "webbing" at thick-line intersections).
                  None (default) = auto: True for rdp, False for vw. Ignored
                  for arch mode.
        thick_threshold: Unused in the current arch mode (kept for API compat).

    Returns:
        (svg_string, stats_dict)

    Raises:
        FileNotFoundError: Image not found or unreadable.
        ValueError: Image is blank after binarization.
    """
    if simplify not in ("vw", "rdp", "arch"):
        raise ValueError(f"simplify must be 'vw', 'rdp', or 'arch', got {simplify!r}")

    binary, (height, width) = load_and_binarize(image_path, threshold=threshold)

    collected_warnings: list[str] = []

    # In RDP/arch mode: force straight-line output regardless of straight_threshold arg.
    effective_straight_threshold = float("inf") if simplify in ("rdp", "arch") else straight_threshold

    # Skeleton mode: off by default for all modes. Ignored for arch.
    use_skel = False if skeletonize is None else skeletonize

    freed = 0
    arc_count = 0

    if testing_prefix is not None and simplify != "arch":
        bitmap = _zhang_suen_thin(binary) if use_skel else binary
        suffix = "_skeleton" if use_skel else "_binary"
        cv2.imwrite(f"{testing_prefix}{suffix}.bmp", bitmap)

    if simplify == "arch":
        if testing_prefix is not None:
            cv2.imwrite(f"{testing_prefix}_skeleton.bmp", _zhang_suen_thin(binary))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            straight_paths, curved_paths, epsilon, loss = find_arch_paths(
                binary,
                max_points=max_points,
                min_contour_area=min_contour_area,
                straight_threshold=straight_threshold,
                eps_min=eps_min,
            )
        collected_warnings.extend(str(w.message) for w in caught)

        if testing_prefix is not None:
            # Testing mode: pass 1 only (straight paths).
            curved_paths = []

        if min_vw_points > 0:
            straight_paths = [c for c in straight_paths if len(c) > min_vw_points]
            curved_paths = [c for c in curved_paths if len(c) > min_vw_points]

        simplified_contours = straight_paths + curved_paths
        path_strings = contours_to_svg_paths(
            simplified_contours,
            tension=tension,
            straight_threshold=float("inf"),
            arc_tolerance=None,
            closed=False,
        )

    elif use_skel:
        # Skeleton centerline path: bypass perimeter contours entirely.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simplified_contours, epsilon, loss = find_skeleton_paths(
                binary,
                max_points=max_points,
                min_contour_area=min_contour_area,
                contour_smooth=0.0,
                simplify=simplify,
                eps_min=eps_min,
            )
        collected_warnings.extend(str(w.message) for w in caught)

        if min_vw_points > 0:
            simplified_contours = [c for c in simplified_contours if len(c) > min_vw_points]

        path_strings = contours_to_svg_paths(
            simplified_contours,
            tension=tension,
            straight_threshold=effective_straight_threshold,
            arc_tolerance=None,   # centerlines are not circular regions
            closed=False,         # skeleton paths are open polylines
        )

    else:
        def _run(budget: int):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if simplify == "rdp":
                    result = find_contours_rdp(
                        binary,
                        max_points=budget,
                        min_contour_area=min_contour_area,
                        contour_smooth=contour_smooth,
                        eps_min=eps_min,
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
            closed=True,
        )

        if arc_tolerance is not None:
            arc_count = sum(1 for p in path_strings if " A " in p or p.startswith("A "))

    svg_content = build_svg(
        path_strings,
        width=width,
        height=height,
        stroke_width=stroke_width,
    )

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
        "skeleton": use_skel or simplify == "arch",
        "warnings": collected_warnings,
    }

    return svg_content, stats
