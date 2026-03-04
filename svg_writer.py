import os


def build_svg(
    path_strings: list[str],
    width: int,
    height: int,
    stroke_width: float = 2.0,
    background_color: str = "white",
    stroke_color: str = "black",
) -> str:
    """Assemble an SVG document from a list of path `d` strings.

    Args:
        path_strings: List of SVG path `d` attribute values.
        width: Image width in pixels (used for viewBox and width attribute).
        height: Image height in pixels.
        stroke_width: Stroke width for all paths.
        background_color: Fill color of the background rectangle.
        stroke_color: Stroke color for all paths.

    Returns:
        Complete SVG document as a string.
    """
    path_elements = "\n    ".join(
        f'<path d="{d}"/>' for d in path_strings
    )

    if path_elements:
        group = (
            f'  <g fill="none" stroke="{stroke_color}" stroke-width="{stroke_width}">\n'
            f"    {path_elements}\n"
            f"  </g>"
        )
    else:
        group = ""

    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg"\n'
        f'     width="{width}" height="{height}"\n'
        f'     viewBox="0 0 {width} {height}">\n'
        f'  <rect width="100%" height="100%" fill="{background_color}"/>\n'
        f"{group}\n"
        f"</svg>\n"
    )


def write_svg(svg_content: str, output_path: str) -> None:
    """Write an SVG string to disk.

    Args:
        svg_content: Complete SVG document string.
        output_path: Destination file path.

    Raises:
        IOError: If the parent directory does not exist.
    """
    parent = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(parent):
        raise IOError(f"Output directory does not exist: {parent!r}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
