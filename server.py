"""jpgtracer web server.

Install dependencies (one-time):
    pip install flask opencv-python numpy scipy

Run:
    python server.py

Then open http://localhost:5001 in your browser.
"""

import json
import os
import tempfile

from flask import Flask, Response, request, send_from_directory

from pipeline import trace

app = Flask(__name__)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/trace", methods=["POST"])
def trace_endpoint():
    file = request.files.get("image")
    if file is None:
        return Response("No image uploaded", status=400)

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        file.save(tmp.name)
        tmp.close()

        no_arcs = request.form.get("no_arcs") == "true"
        threshold_raw = request.form.get("threshold")
        threshold = int(threshold_raw) if threshold_raw else None

        svg, stats = trace(
            tmp.name,
            max_points=int(request.form.get("max_points", 500)),
            threshold=threshold,
            min_contour_area=float(request.form.get("min_contour_area", 10.0)),
            stroke_width=float(request.form.get("stroke_width", 2.0)),
            tension=float(request.form.get("tension", 0.5)),
            contour_smooth=float(request.form.get("contour_smooth", 1.5)),
            straight_threshold=float(request.form.get("straight_threshold", 1.0)),
            arc_tolerance=None if no_arcs else float(request.form.get("arc_tolerance", 1.5)),
            min_vw_points=int(request.form.get("min_vw_points", 0)),
        )
    except (FileNotFoundError, ValueError) as exc:
        return Response(str(exc), status=400)
    finally:
        os.unlink(tmp.name)

    return Response(
        json.dumps({"svg": svg, "stats": stats}),
        mimetype="application/json",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
