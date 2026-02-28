import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Iterable, Tuple, Optional

# ---- Constants (kg only) ----
PX2_TO_KG = 0.00012  # px^2 -> kg (given GSD=0.5 cm/px and 0.48 g/cm^2)
LOW_MULT = 1.0
MOD_MULT = 1.5
HIGH_MULT = 2.0


def shoelace_area_px2(points: List[Dict[str, float]]) -> float:
    """
    Gauss / Shoelace area formula.
    points: [{"x":..., "y":...}, ...] in pixel coordinates
    returns area in px^2
    """
    n = len(points)
    if n < 3:
        return 0.0

    s = 0.0
    for i in range(n):
        x1, y1 = points[i]["x"], points[i]["y"]
        x2, y2 = points[(i + 1) % n]["x"], points[(i + 1) % n]["y"]
        s += x1 * y2 - y1 * x2
    return abs(s) / 2.0


def iter_prediction_objects(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Supports either:
      - {"predictions": [...]}  (single object)
      - [ {"predictions": [...]}, {"predictions": [...]} ] (list of objects)
      - {"outputs": [ {"predictions":[...]} , ... ]} (common wrapper)
    Yields individual prediction dicts.
    """
    if isinstance(obj, dict):
        if "predictions" in obj and isinstance(obj["predictions"], list):
            yield from obj["predictions"]
            return
        if "outputs" in obj and isinstance(obj["outputs"], list):
            for out in obj["outputs"]:
                if isinstance(out, dict) and "predictions" in out:
                    yield from out["predictions"]
            return
        # fallback: maybe nested one level
        for v in obj.values():
            if isinstance(v, (dict, list)):
                yield from iter_prediction_objects(v)
        return

    if isinstance(obj, list):
        for item in obj:
            yield from iter_prediction_objects(item)


def prediction_to_row(pred: Dict[str, Any]) -> Dict[str, Any]:
    pts = pred.get("points", [])
    area_px2 = shoelace_area_px2(pts) if isinstance(pts, list) else 0.0
    surface_kg = area_px2 * PX2_TO_KG

    return {
        "detection_id": pred.get("detection_id"),
        "class": pred.get("class"),
        "class_id": pred.get("class_id"),
        "confidence": pred.get("confidence"),
        "bbox_x": pred.get("x"),
        "bbox_y": pred.get("y"),
        "bbox_width": pred.get("width"),
        "bbox_height": pred.get("height"),
        "num_points": len(pts) if isinstance(pts, list) else 0,
        "area_px2": area_px2,
        "mass_low_kg": surface_kg * LOW_MULT,
        "mass_mod_kg": surface_kg * MOD_MULT,
        "mass_high_kg": surface_kg * HIGH_MULT,
    }


def json_predictions_to_csv(
    input_json_path: str,
    output_csv_path: str,
    write_totals_row: bool = True,
) -> None:
    input_path = Path(input_json_path)
    output_path = Path(output_csv_path)

    data = json.loads(input_path.read_text())

    rows = [prediction_to_row(p) for p in iter_prediction_objects(data)]

    fieldnames = [
        "detection_id",
        "class",
        "class_id",
        "confidence",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "num_points",
        "area_px2",
        "mass_low_kg",
        "mass_mod_kg",
        "mass_high_kg",
    ]

    with output_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

        if write_totals_row:
            total_low = sum(r["mass_low_kg"] for r in rows)
            total_mod = sum(r["mass_mod_kg"] for r in rows)
            total_high = sum(r["mass_high_kg"] for r in rows)
            total_area = sum(r["area_px2"] for r in rows)

            w.writerow({
                "detection_id": "TOTAL",
                "class": "",
                "class_id": "",
                "confidence": "",
                "bbox_x": "",
                "bbox_y": "",
                "bbox_width": "",
                "bbox_height": "",
                "num_points": "",
                "area_px2": total_area,
                "mass_low_kg": total_low,
                "mass_mod_kg": total_mod,
                "mass_high_kg": total_high,
            })


if __name__ == "__main__":
    # Example usage:
    # json_predictions_to_csv("predictions.json", "predictions_mass.csv")
    #
    # If you're running from terminal:
    #   python script.py predictions.json predictions_mass.csv
    import sys

    if len(sys.argv) < 3:
        print("Usage: python script.py <input.json> <output.csv>")
        sys.exit(1)

    json_predictions_to_csv(sys.argv[1], sys.argv[2])
    print(f"Wrote CSV to {sys.argv[2]}")