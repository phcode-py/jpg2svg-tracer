"""jpgtracer — trace lines in a JPG and output an SVG vector graphic.

Usage:
    python main.py input.jpg [options]

The pipeline:
  1. Load and binarize the image (Otsu's thresholding by default).
  2. Gaussian-smooth raw contour coordinates to remove pixel-grid staircases.
  3. Simplify with Douglas-Peucker, binary-searching epsilon to stay within --max-points.
  4. Fit Catmull-Rom cubic Bezier curves; straight segments become L commands.
  5. Write an SVG file with a white background and black stroked paths.
"""

import argparse
import os
import sys
import warnings

from image_processing import load_and_binarize


from contour_tracer import find_contours_with_budget
from bezier import contours_to_svg_paths
from svg_writer import build_svg, write_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jpgtracer",
        description="Trace lines in a JPG image and output an SVG vector graphic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to input JPG (or PNG/BMP) image.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output SVG path. Defaults to the input filename with .svg extension.",
    )
    parser.add_argument(
        "--max-points", "-n",
        type=int,
        default=500,
        dest="max_points",
        help="Maximum total control points across all paths in the output SVG.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=None,
        help="Manual binarization threshold 0-255. If omitted, Otsu's method is used.",
    )
    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=10.0,
        dest="min_contour_area",
        help="Ignore contours with a pixel area smaller than this value (noise filter).",
    )
    parser.add_argument(
        "--stroke-width",
        type=float,
        default=2.0,
        dest="stroke_width",
        help="SVG stroke width in pixels.",
    )
    parser.add_argument(
        "--tension",
        type=float,
        default=0.5,
        help=(
            "Catmull-Rom tension for curve smoothness. "
            "0 = straight line segments, 1 = tighter curves. "
            "0.5 is the classic Catmull-Rom value."
        ),
    )
    parser.add_argument(
        "--contour-smooth",
        type=float,
        default=1.5,
        dest="contour_smooth",
        help=(
            "Gaussian sigma (px) applied to raw contour coordinates before simplification. "
            "Eliminates pixel-grid staircase artifacts on diagonal edges. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--straight-threshold",
        type=float,
        default=1.0,
        dest="straight_threshold",
        help=(
            "Max Bezier control-point deviation (px) below which a segment is emitted "
            "as a straight line (L) instead of a cubic curve (C). "
            "Prevents artificial bending of straight sections. Set to 0 to always use curves."
        ),
    )
    parser.add_argument(
        "--arc-tolerance",
        type=float,
        default=1.5,
        dest="arc_tolerance",
        help=(
            "Max mean residual (px) to accept a circle fit and emit an SVG arc (A command). "
            "Lower = stricter, fewer arcs detected. Increase for noisy/JPEG inputs. "
            "Has no effect when --no-arcs is set. (default: 1.5)"
        ),
    )
    parser.add_argument(
        "--no-arcs",
        action="store_true",
        default=False,
        dest="no_arcs",
        help=(
            "Disable arc detection entirely. All curves are rendered with Catmull-Rom "
            "Bezier approximations (the original behavior before arc support was added)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Derive output path.
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = base + ".svg"

    # --- Step 1: Load and binarize ---
    print(f"Loading:  {args.input}")
    try:
        binary, (height, width) = load_and_binarize(
            args.input,
            threshold=args.threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Size:     {width} x {height} px")

    # --- Step 2: Detect and simplify contours ---
    print("Tracing contours...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simplified_contours, epsilon, loss = find_contours_with_budget(
            binary,
            max_points=args.max_points,
            min_contour_area=args.min_contour_area,
            contour_smooth=args.contour_smooth,
        )
    for w in caught:
        print(f"Warning: {w.message}", file=sys.stderr)

    if not simplified_contours:
        print(
            "Warning: no contours found after filtering. "
            "The output SVG will contain only the background.",
            file=sys.stderr,
        )

    total_points = sum(len(c) for c in simplified_contours)
    print(f"Contours: {len(simplified_contours)}")
    print(f"Points:   {total_points} (budget: {args.max_points})")
    print(f"VW floor: {epsilon:.4f} px² (min triangle area surviving simplification)")
    print(f"Loss:     {loss:.4f} px (mean deviation from original)")

    # --- Step 3: Convert to SVG path strings ---
    arc_tol = None if args.no_arcs else args.arc_tolerance
    path_strings = contours_to_svg_paths(
        simplified_contours,
        tension=args.tension,
        straight_threshold=args.straight_threshold,
        arc_tolerance=arc_tol,
    )
    if arc_tol is not None:
        arc_count = sum(1 for p in path_strings if " A " in p or p.startswith("A "))
        print(f"Arc segs: arcs detected in {arc_count}/{len(path_strings)} paths")

    # --- Step 4: Assemble and write SVG ---
    svg_content = build_svg(
        path_strings,
        width=width,
        height=height,
        stroke_width=args.stroke_width,
    )
    try:
        write_svg(svg_content, args.output)
    except IOError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Output:   {args.output}")


if __name__ == "__main__":
    main()
