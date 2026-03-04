"""jpgtracer — trace lines in a JPG and output an SVG vector graphic.

Usage:
    python main.py input.jpg [options]

The pipeline:
  1. Load and binarize the image (Otsu's thresholding by default).
  2. Gaussian-smooth raw contour coordinates to remove pixel-grid staircases.
  3. Simplify with Visvalingam-Whyatt to stay within --max-points.
  4. Fit Catmull-Rom cubic Bezier curves; straight segments become L commands.
  5. Detect circular arcs and emit exact SVG A commands (unless --no-arcs).
  6. Write an SVG file with a white background and black stroked paths.
"""

import argparse
import os
import sys

from pipeline import trace
from svg_writer import write_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jpgtracer",
        description="Trace lines in a JPG image and output an SVG vector graphic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Path to input JPG (or PNG/BMP) image.")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output SVG path. Defaults to the input filename with .svg extension.",
    )
    parser.add_argument(
        "--max-points", "-n",
        type=int, default=500, dest="max_points",
        help="Maximum total control points across all paths in the output SVG.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int, default=None,
        help="Manual binarization threshold 0-255. If omitted, Otsu's method is used.",
    )
    parser.add_argument(
        "--min-contour-area",
        type=float, default=10.0, dest="min_contour_area",
        help="Ignore contours with a pixel area smaller than this value (noise filter).",
    )
    parser.add_argument(
        "--stroke-width",
        type=float, default=2.0, dest="stroke_width",
        help="SVG stroke width in pixels.",
    )
    parser.add_argument(
        "--tension",
        type=float, default=0.5,
        help="Catmull-Rom tension [0=polyline, 1=tight]. 0.5 is the classic value.",
    )
    parser.add_argument(
        "--contour-smooth",
        type=float, default=1.5, dest="contour_smooth",
        help=(
            "Gaussian sigma (px) applied to raw contour coordinates before simplification. "
            "Eliminates pixel-grid staircase artifacts. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--straight-threshold",
        type=float, default=1.0, dest="straight_threshold",
        help=(
            "Max Bezier control-point deviation (px) below which a segment is emitted "
            "as L instead of C. Set to 0 to always use curves."
        ),
    )
    parser.add_argument(
        "--arc-tolerance",
        type=float, default=1.5, dest="arc_tolerance",
        help=(
            "Max mean residual (px) to accept a circle fit and emit an SVG arc (A). "
            "Has no effect when --no-arcs is set."
        ),
    )
    parser.add_argument(
        "--no-arcs",
        action="store_true", default=False, dest="no_arcs",
        help="Disable arc detection. All curves use Catmull-Rom Bezier approximations.",
    )
    parser.add_argument(
        "--declutter",
        type=int, default=0, dest="min_vw_points",
        help="Remove contours with this many simplified points or fewer. 0 = off.",
    )
    parser.add_argument(
        "--simplify",
        choices=["vw", "rdp", "arch"], default="vw",
        help=(
            "Simplification algorithm. "
            "'vw' (default): Visvalingam-Whyatt + Catmull-Rom curves, best for organic shapes. "
            "'rdp': Ramer-Douglas-Peucker + straight lines, best for architectural/technical drawings. "
            "'arch': Architectural Plan mode — two-pass: skeleton on thick lines (--thick-threshold) "
            "with half the point budget, then RDP on the full image for details."
        ),
    )
    parser.add_argument(
        "--thick-threshold",
        type=int, default=None, dest="thick_threshold",
        help=(
            "Binarization threshold (0-255) for the first pass of --simplify arch. "
            "Set high enough that only thick black lines are visible (e.g. 50-100). Required for arch mode."
        ),
    )
    parser.add_argument(
        "--skeleton",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="skeletonize",
        help=(
            "Extract skeleton centerlines instead of perimeter contours. "
            "Eliminates webbing at intersections of thick lines. "
            "Default: off for all modes (use --simplify arch for the combined approach)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = base + ".svg"

    arc_tol = None if args.no_arcs else args.arc_tolerance

    print(f"Loading:  {args.input}")
    try:
        svg, stats = trace(
            args.input,
            max_points=args.max_points,
            threshold=args.threshold,
            min_contour_area=args.min_contour_area,
            stroke_width=args.stroke_width,
            tension=args.tension,
            contour_smooth=args.contour_smooth,
            straight_threshold=args.straight_threshold,
            arc_tolerance=arc_tol,
            min_vw_points=args.min_vw_points,
            simplify=args.simplify,
            skeletonize=args.skeletonize,
            thick_threshold=args.thick_threshold,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    for w in stats["warnings"]:
        print(f"Warning: {w}", file=sys.stderr)

    if stats["contours"] == 0:
        print(
            "Warning: no contours found after filtering. "
            "The output SVG will contain only the background.",
            file=sys.stderr,
        )

    skel_label = "+Skeleton" if stats.get("skeleton") and stats["simplify"] != "arch" else ""
    mode_label = {"vw": "VW+Bezier", "rdp": "RDP+lines", "arch": "Arch (skel+RDP)"}[stats["simplify"]] + skel_label
    print(f"Size:     {stats['width']} x {stats['height']} px")
    print(f"Mode:     {mode_label}")
    print(f"Contours: {stats['contours']}")
    print(f"Points:   {stats['points']} (budget: {args.max_points})")
    if stats["freed"] > 0:
        print(f"Arc freed: {stats['freed']} pts reallocated to non-arc segments")
    if stats["simplify"] == "vw":
        print(f"VW floor: {stats['epsilon']:.4f} px² (min triangle area surviving)")
    else:
        print(f"RDP eps:  {stats['epsilon']:.4f} px (max point-to-chord deviation)")
    print(f"Loss:     {stats['loss']:.4f} px (mean deviation from original)")
    if arc_tol is not None and not stats.get("skeleton"):
        print(f"Arc segs: arcs detected in {stats['arc_count']}/{stats['contours']} paths")

    try:
        write_svg(svg, args.output)
    except IOError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Output:   {args.output}")


if __name__ == "__main__":
    main()
