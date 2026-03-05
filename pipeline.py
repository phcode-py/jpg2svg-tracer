"""Core tracing pipeline shared by CLI (main.py) and web server (server.py)."""

import warnings

import cv2

from image_processing import load_and_binarize, _zhang_suen_thin
from contour_tracer import find_contours_with_budget, find_contours_rdp, find_skeleton_paths
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
) -> tuple[str, dict]:
    """Run the full tracing pipeline.

    Args:
        simplify: "vw" (Visvalingam-Whyatt + Catmull-Rom, default),
                  "rdp" (Ramer-Douglas-Peucker + straight lines), or
                  "arch" (Architectural Plan: two-pass — skeleton on thick lines
                  at thick_threshold, then RDP on full image).
        skeletonize: If True, extract skeleton centerlines instead of perimeter
                  contours (eliminates "webbing" at thick-line intersections).
                  None (default) = auto: True for rdp, False for vw. Ignored
                  for arch mode.
        thick_threshold: Binarization threshold (0-255) used in the first pass
                  of arch mode to isolate only thick black lines. Required when
                  simplify="arch".

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
        if thick_threshold is None:
            raise ValueError("simplify='arch' requires thick_threshold to be set")

        skel_budget = max_points // 2
        rdp_budget = max_points - skel_budget

        # Pass 1: skeletonize the thick-lines-only image.
        binary_thick, _ = load_and_binarize(image_path, threshold=thick_threshold)

        if testing_prefix is not None:
            cv2.imwrite(f"{testing_prefix}_binary_thick.bmp", binary_thick)
            cv2.imwrite(f"{testing_prefix}_skeleton_thick.bmp", _zhang_suen_thin(binary_thick))
            cv2.imwrite(f"{testing_prefix}_binary.bmp", binary)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            skel_paths, skel_metric, skel_loss = find_skeleton_paths(
                binary_thick,
                max_points=skel_budget,
                min_contour_area=min_contour_area,
                contour_smooth=0.0,
                simplify="rdp",
            )
        collected_warnings.extend(str(w.message) for w in caught)

        if min_vw_points > 0:
            skel_paths = [c for c in skel_paths if len(c) > min_vw_points]

        skel_path_strings = contours_to_svg_paths(
            skel_paths,
            tension=tension,
            straight_threshold=float("inf"),
            arc_tolerance=None,
            closed=False,
        )

        # Pass 2: RDP on the full image (regular threshold).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rdp_contours, rdp_epsilon, rdp_loss = find_contours_rdp(
                binary,
                max_points=rdp_budget,
                min_contour_area=min_contour_area,
                contour_smooth=contour_smooth,
            )
        collected_warnings.extend(str(w.message) for w in caught)

        if min_vw_points > 0:
            rdp_contours = [c for c in rdp_contours if len(c) > min_vw_points]

        rdp_path_strings = contours_to_svg_paths(
            rdp_contours,
            tension=tension,
            straight_threshold=float("inf"),
            arc_tolerance=None,
            closed=True,
        )

        path_strings = skel_path_strings + rdp_path_strings
        simplified_contours = skel_paths + rdp_contours
        epsilon = max(skel_metric, rdp_epsilon)
        total_n = len(skel_paths) + len(rdp_contours)
        loss = (skel_loss * len(skel_paths) + rdp_loss * len(rdp_contours)) / total_n if total_n > 0 else 0.0

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
