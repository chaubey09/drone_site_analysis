"""
07_structure_tracker.py
───────────────────────
Individual Structure Tracker

Lets you define named structures (pillars, spans, stations, walls) as points
or polygons on the orthomosaic. Each structure is tracked every period:
  - Not Started / Earthwork / Rebar / Formwork / Concrete / Complete
  - % progress estimated from pixel classification inside its boundary
  - Status change detected vs previous period

Two ways to define structures:
  A) JSON file (structures.json) — define once, reused every period
  B) Interactive marker tool — click on orthomosaic to place structures (run separately)

Output:
  - structure_status_{period}.json   — per-structure status this period
  - structure_map_{period}.png       — orthomosaic with all structures labeled
  - structure_delta_{prev}_{curr}.json — what changed between periods
"""

import cv2
import json
import numpy as np
import os
from pathlib import Path
import logging
import yaml
from dataclasses import dataclass, asdict
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── STRUCTURE DEFINITION ─────────────────────────────────────────────────────

# Construction stage order (low → high = more complete)
STAGE_ORDER = [
    "not_started",
    "earthwork",
    "rebar",
    "formwork",
    "concrete",
    "complete",
]

STAGE_COLORS = {
    "not_started": (80,  80,  80),    # dark gray
    "earthwork":   (30, 140, 180),    # amber
    "rebar":       (0,  140, 220),    # orange
    "formwork":    (0,  165, 255),    # light orange
    "concrete":    (180,180,180),     # light gray
    "complete":    (50, 200, 50),     # green
}

STAGE_LABELS = {
    "not_started": "NOT STARTED",
    "earthwork":   "EARTHWORK",
    "rebar":       "REBAR",
    "formwork":    "FORMWORK",
    "concrete":    "CONCRETE",
    "complete":    "COMPLETE",
}

# Segmentation class IDs (must match 03_segment.py CLASS_NAMES)
CLASS_IDS = {
    "background":         0,
    "concrete_structure": 1,
    "steel_rebar":        2,
    "formwork":           3,
    "earthwork":          4,
    "road_surface":       5,
    "equipment":          6,
    "material_stockpile": 7,
    "water_body":         8,
    "vegetation":         9,
    "workers":            10,
}


@dataclass
class Structure:
    """A single named structure on the construction site."""
    id: str                      # e.g. "P001"
    name: str                    # e.g. "Pillar 1"
    type: str                    # "pillar" | "span" | "station" | "wall" | "other"
    # Location: either a center point + radius (circle) or polygon (list of [x,y])
    center_x: int                # pixel x in reference orthomosaic
    center_y: int                # pixel y in reference orthomosaic
    radius_px: int = 20          # for circular markers
    polygon: Optional[list] = None  # [[x1,y1],[x2,y2],...] for polygon markers
    planned_completion: Optional[str] = None  # e.g. "2024_05"
    notes: str = ""


def load_structures(structures_path: str) -> list[Structure]:
    """Load structure definitions from JSON file."""
    if not os.path.exists(structures_path):
        log.warning(f"Structures file not found: {structures_path}")
        log.warning("Run: python 07_structure_tracker.py --create-template")
        return []
    with open(structures_path, encoding="utf-8") as f:
        data = json.load(f)
    structures = []
    for d in data.get("structures", []):
        s = Structure(
            id=d["id"],
            name=d["name"],
            type=d.get("type", "pillar"),
            center_x=d["center_x"],
            center_y=d["center_y"],
            radius_px=d.get("radius_px", 20),
            polygon=d.get("polygon"),
            planned_completion=d.get("planned_completion"),
            notes=d.get("notes", ""),
        )
        structures.append(s)
    log.info(f"Loaded {len(structures)} structures from {structures_path}")
    return structures


def save_structures(structures: list[Structure], structures_path: str):
    """Save structure definitions to JSON."""
    Path(os.path.dirname(structures_path)).mkdir(parents=True, exist_ok=True)
    data = {"structures": [asdict(s) for s in structures]}
    with open(structures_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved {len(structures)} structures to {structures_path}")


# ─── STAGE DETECTION ──────────────────────────────────────────────────────────

def get_structure_mask(structure: Structure, ortho_shape: tuple) -> np.ndarray:
    """
    Return a boolean mask (H, W) covering the structure's area on the orthomosaic.
    """
    H, W = ortho_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    if structure.polygon:
        pts = np.array(structure.polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    else:
        cv2.circle(mask, (structure.center_x, structure.center_y),
                   structure.radius_px, 1, -1)
    return mask.astype(bool)


def classify_structure_stage(seg_mask: np.ndarray,
                               structure_area_mask: np.ndarray) -> tuple[str, dict]:
    """
    Determine the construction stage of a structure based on what's detected
    inside its boundary in the segmentation mask.

    Returns (stage_name, class_pixel_counts)
    """
    pixels_in_structure = seg_mask[structure_area_mask]
    total = len(pixels_in_structure)

    if total == 0:
        return "not_started", {}

    counts = {}
    for name, cls_id in CLASS_IDS.items():
        counts[name] = int(np.sum(pixels_in_structure == cls_id))

    pct = {name: counts[name] / total for name in counts}

    # Stage decision logic (priority order)
    if pct.get("concrete_structure", 0) > 0.30:
        # More than 30% concrete → concrete stage or complete
        if pct.get("concrete_structure", 0) > 0.70:
            return "complete", counts
        return "concrete", counts

    elif pct.get("formwork", 0) > 0.15:
        return "formwork", counts

    elif pct.get("steel_rebar", 0) > 0.10:
        return "rebar", counts

    elif pct.get("earthwork", 0) > 0.20:
        return "earthwork", counts

    elif pct.get("background", 0) > 0.80:
        return "not_started", counts

    else:
        # Mixed — pick the most dominant construction class
        construction_classes = ["concrete_structure", "steel_rebar",
                                  "formwork", "earthwork"]
        dominant = max(construction_classes, key=lambda c: pct.get(c, 0))
        stage_map = {
            "concrete_structure": "concrete",
            "steel_rebar":        "rebar",
            "formwork":           "formwork",
            "earthwork":          "earthwork",
        }
        return stage_map.get(dominant, "not_started"), counts


def stage_progress_pct(stage: str) -> int:
    """Convert stage name to 0-100 progress percentage."""
    mapping = {
        "not_started": 0,
        "earthwork":   20,
        "rebar":       40,
        "formwork":    60,
        "concrete":    80,
        "complete":    100,
    }
    return mapping.get(stage, 0)


# ─── ANNOTATED MAP GENERATOR ──────────────────────────────────────────────────

def draw_structure_map(ortho: np.ndarray,
                        structures: list[Structure],
                        statuses: dict,
                        period_id: str) -> np.ndarray:
    """
    Draw all structures on the orthomosaic with color-coded status labels.
    Returns annotated image.
    """
    img = ortho.copy()
    H, W = img.shape[:2]

    # Semi-transparent overlay for structure areas
    overlay = img.copy()

    for s in structures:
        status = statuses.get(s.id, {})
        stage = status.get("stage", "not_started")
        color = STAGE_COLORS[stage]
        bgr = color[::-1]  # RGB → BGR

        # Draw structure boundary
        if s.polygon:
            pts = np.array(s.polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], bgr)
            cv2.polylines(img, [pts], True, bgr, 2)
        else:
            cv2.circle(overlay, (s.center_x, s.center_y), s.radius_px, bgr, -1)
            cv2.circle(img, (s.center_x, s.center_y), s.radius_px, bgr, 2)

    # Blend overlay
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # Draw labels on top
    for s in structures:
        status = statuses.get(s.id, {})
        stage = status.get("stage", "not_started")
        pct = stage_progress_pct(stage)
        color = STAGE_COLORS[stage]
        bgr = color[::-1]

        label_x = s.center_x
        label_y = s.center_y - s.radius_px - 8 if not s.polygon else s.center_y

        # Structure ID
        cv2.putText(img, s.id, (label_x - 15, label_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, s.id, (label_x - 15, label_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Progress %
        pct_label = f"{pct}%"
        cv2.putText(img, pct_label, (label_x - 12, label_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, pct_label, (label_x - 12, label_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, bgr, 1, cv2.LINE_AA)

    # Legend
    legend_x, legend_y = 10, H - (len(STAGE_ORDER) * 22 + 10)
    cv2.rectangle(img, (legend_x - 4, legend_y - 18),
                  (legend_x + 140, H - 4), (20, 20, 20), -1)
    for i, stage in enumerate(STAGE_ORDER):
        color = STAGE_COLORS[stage]
        bgr = color[::-1]
        y = legend_y + i * 22
        cv2.rectangle(img, (legend_x, y - 10), (legend_x + 14, y + 4), bgr, -1)
        cv2.putText(img, STAGE_LABELS[stage],
                    (legend_x + 18, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # Title
    title = f"Structure Map — {period_id}"
    cv2.rectangle(img, (0, 0), (len(title) * 10 + 20, 28), (10, 10, 10), -1)
    cv2.putText(img, title, (10, 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 200, 255), 1)

    return img


# ─── DELTA COMPUTATION ────────────────────────────────────────────────────────

def compute_structure_delta(prev_statuses: dict, curr_statuses: dict,
                             structures: list[Structure]) -> list[dict]:
    """
    Compare structure statuses between two periods.
    Returns list of changes.
    """
    deltas = []
    for s in structures:
        sid = s.id
        prev = prev_statuses.get(sid, {})
        curr = curr_statuses.get(sid, {})

        prev_stage = prev.get("stage", "not_started")
        curr_stage = curr.get("stage", "not_started")
        prev_pct   = stage_progress_pct(prev_stage)
        curr_pct   = stage_progress_pct(curr_stage)
        delta_pct  = curr_pct - prev_pct

        # Status determination
        if delta_pct > 0:
            change_status = "progressed"
        elif delta_pct < 0:
            change_status = "regressed"
        else:
            change_status = "no_change"

        # Alert for PM
        alert = None
        if change_status == "progressed":
            alert = f"Advanced: {prev_stage} → {curr_stage}"
        elif change_status == "regressed":
            alert = f"REGRESSION: {prev_stage} → {curr_stage}"
        elif curr_stage == "not_started" and s.planned_completion:
            alert = f"Not started yet (planned: {s.planned_completion})"

        deltas.append({
            "structure_id":   sid,
            "structure_name": s.name,
            "type":           s.type,
            "prev_stage":     prev_stage,
            "curr_stage":     curr_stage,
            "prev_pct":       prev_pct,
            "curr_pct":       curr_pct,
            "delta_pct":      delta_pct,
            "change_status":  change_status,
            "alert":          alert,
            "planned_completion": s.planned_completion,
        })

    return sorted(deltas, key=lambda x: x["structure_id"])


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def track_structures(period_id: str, prev_period: Optional[str],
                     processed_dir: str, config: dict,
                     structures_path: str = "config/structures.json") -> dict:
    """
    Main structure tracking function for one period.
    Returns dict with statuses and delta.
    """
    out_dir = os.path.join(processed_dir, period_id)

    # Load orthomosaic and segmentation mask
    ortho_path = os.path.join(out_dir, f"orthomosaic_{period_id}.png")
    mask_path  = os.path.join(out_dir, f"segmentation_mask_{period_id}.png")

    ortho = cv2.imread(ortho_path)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if ortho is None or mask is None:
        raise FileNotFoundError(
            f"Orthomosaic or segmentation mask not found for {period_id}. "
            f"Run steps 2 and 3 first."
        )

    H, W = ortho.shape[:2]
    log.info(f"Tracking structures on {W}x{H} orthomosaic for {period_id}")

    # Load structure definitions
    structures = load_structures(structures_path)
    if not structures:
        log.warning("No structures defined. Create config/structures.json first.")
        log.warning("Run: python 07_structure_tracker.py --create-template --ortho-size 1218 639")
        return {"period_id": period_id, "structures": [], "delta": []}

    # Classify each structure
    statuses = {}
    for s in structures:
        area_mask = get_structure_mask(s, ortho.shape)
        stage, class_counts = classify_structure_stage(mask, area_mask)
        pct = stage_progress_pct(stage)
        statuses[s.id] = {
            "structure_id":   s.id,
            "structure_name": s.name,
            "type":           s.type,
            "stage":          stage,
            "progress_pct":   pct,
            "class_counts":   class_counts,
            "planned_completion": s.planned_completion,
        }
        log.info(f"  [{s.id}] {s.name:20s} → {STAGE_LABELS[stage]:12s} ({pct}%)")

    # Draw annotated map
    structure_map = draw_structure_map(ortho, structures, statuses, period_id)
    map_path = os.path.join(out_dir, f"structure_map_{period_id}.jpg")
    cv2.imwrite(map_path, structure_map, [cv2.IMWRITE_JPEG_QUALITY, 92])
    log.info(f"Structure map saved: {map_path}")

    # Compute delta vs previous period
    delta = []
    if prev_period:
        prev_status_path = os.path.join(processed_dir, prev_period,
                                         f"structure_status_{prev_period}.json")
        if os.path.exists(prev_status_path):
            with open(prev_status_path, encoding="utf-8") as f:
                prev_data = json.load(f)
            prev_statuses = {s["structure_id"]: s
                             for s in prev_data.get("structures", {}).values()}
            delta = compute_structure_delta(prev_statuses, statuses, structures)

            progressed = sum(1 for d in delta if d["change_status"] == "progressed")
            stalled    = sum(1 for d in delta if d["change_status"] == "no_change")
            regressed  = sum(1 for d in delta if d["change_status"] == "regressed")
            log.info(f"  Delta: {progressed} progressed, {stalled} no change, {regressed} regressed")

    # Save results
    result = {
        "period_id":   period_id,
        "prev_period": prev_period,
        "map_path":    map_path,
        "structures":  statuses,
        "delta":       delta,
        "summary": {
            "total":        len(structures),
            "complete":     sum(1 for s in statuses.values() if s["stage"] == "complete"),
            "concrete":     sum(1 for s in statuses.values() if s["stage"] == "concrete"),
            "formwork":     sum(1 for s in statuses.values() if s["stage"] == "formwork"),
            "rebar":        sum(1 for s in statuses.values() if s["stage"] == "rebar"),
            "earthwork":    sum(1 for s in statuses.values() if s["stage"] == "earthwork"),
            "not_started":  sum(1 for s in statuses.values() if s["stage"] == "not_started"),
        }
    }

    status_path = os.path.join(out_dir, f"structure_status_{period_id}.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    log.info(f"Structure tracking complete for {period_id}")
    log.info(f"  Summary: {result['summary']}")
    return result


# ─── INTERACTIVE MARKER TOOL ──────────────────────────────────────────────────

def run_interactive_marker(ortho_path: str, structures_path: str):
    """
    Click on the orthomosaic to place structure markers.
    Left click = add pillar
    Right click = remove last marker
    Press 's' = save and exit
    Press 'q' = quit without saving
    """
    ortho = cv2.imread(ortho_path)
    if ortho is None:
        print(f"ERROR: Cannot open {ortho_path}")
        return

    # Load existing structures
    if os.path.exists(structures_path):
        with open(structures_path, encoding="utf-8") as f:
            existing = json.load(f).get("structures", [])
    else:
        existing = []

    structures = list(existing)
    click_points = [(s["center_x"], s["center_y"]) for s in structures]
    counter = len(structures) + 1
    structure_type = "pillar"

    print("\n=== INTERACTIVE STRUCTURE MARKER ===")
    print("LEFT CLICK  → Place a structure marker")
    print("RIGHT CLICK → Remove last marker")
    print("'p'         → Switch type to: pillar")
    print("'n'         → Switch type to: span")
    print("'t'         → Switch type to: station")
    print("'s'         → Save and exit")
    print("'q'         → Quit without saving")
    print(f"\nOrthomosaic: {ortho.shape[1]}x{ortho.shape[0]}")
    print(f"Existing structures: {len(existing)}")

    def draw_markers(img):
        canvas = img.copy()
        for i, s in enumerate(structures):
            color = {"pillar": (0,200,255), "span": (0,255,100),
                     "station": (255,100,0)}.get(s.get("type","pillar"), (200,200,200))
            x, y = s["center_x"], s["center_y"]
            cv2.circle(canvas, (x, y), 18, color, 2)
            cv2.circle(canvas, (x, y), 3,  color, -1)
            cv2.putText(canvas, s["id"], (x - 15, y - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
            cv2.putText(canvas, s["id"], (x - 15, y - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        # Status bar
        status_text = (f"Markers: {len(structures)} | "
                       f"Type: {structure_type.upper()} | "
                       f"S=Save  Q=Quit")
        cv2.rectangle(canvas, (0, canvas.shape[0]-28), (canvas.shape[1], canvas.shape[0]),
                      (20,20,20), -1)
        cv2.putText(canvas, status_text, (8, canvas.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,220,255), 1)
        return canvas

    def on_mouse(event, x, y, flags, param):
        nonlocal counter
        if event == cv2.EVENT_LBUTTONDOWN:
            sid = f"P{counter:03d}"
            sname = f"{structure_type.capitalize()} {counter}"
            structures.append({
                "id": sid, "name": sname, "type": structure_type,
                "center_x": x, "center_y": y, "radius_px": 18,
                "polygon": None, "planned_completion": None, "notes": ""
            })
            click_points.append((x, y))
            counter += 1
            print(f"  Added [{sid}] {sname} at ({x}, {y})")
            cv2.imshow("Structure Marker", draw_markers(ortho))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if structures:
                removed = structures.pop()
                if click_points:
                    click_points.pop()
                counter -= 1
                print(f"  Removed [{removed['id']}] {removed['name']}")
                cv2.imshow("Structure Marker", draw_markers(ortho))

    cv2.namedWindow("Structure Marker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Structure Marker", min(1400, ortho.shape[1]),
                      min(800, ortho.shape[0]))
    cv2.setMouseCallback("Structure Marker", on_mouse)
    cv2.imshow("Structure Marker", draw_markers(ortho))

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            save_structures([Structure(**{k: v for k, v in s.items()
                                          if k in Structure.__dataclass_fields__})
                              for s in structures], structures_path)
            print(f"\nSaved {len(structures)} structures to {structures_path}")
            break
        elif key == ord('q'):
            print("Quit without saving.")
            break
        elif key == ord('p'):
            structure_type = "pillar"
            print(f"Type: PILLAR")
            cv2.imshow("Structure Marker", draw_markers(ortho))
        elif key == ord('n'):
            structure_type = "span"
            print(f"Type: SPAN")
            cv2.imshow("Structure Marker", draw_markers(ortho))
        elif key == ord('t'):
            structure_type = "station"
            print(f"Type: STATION")
            cv2.imshow("Structure Marker", draw_markers(ortho))

    cv2.destroyAllWindows()


# ─── TEMPLATE GENERATOR ───────────────────────────────────────────────────────

def create_template(structures_path: str, ortho_w: int = 1218, ortho_h: int = 639,
                    n_pillars: int = 10):
    """
    Create a sample structures.json template with evenly spaced pillars.
    Edit this file with your actual pillar positions.
    """
    structures = []
    spacing = ortho_w // (n_pillars + 1)
    for i in range(1, n_pillars + 1):
        structures.append({
            "id": f"P{i:03d}",
            "name": f"Pillar {i}",
            "type": "pillar",
            "center_x": spacing * i,
            "center_y": ortho_h // 2,
            "radius_px": 20,
            "polygon": None,
            "planned_completion": None,
            "notes": f"Replace center_x/center_y with actual pixel position of Pillar {i}"
        })

    Path(os.path.dirname(structures_path)).mkdir(parents=True, exist_ok=True)
    with open(structures_path, "w", encoding="utf-8") as f:
        json.dump({"structures": structures}, f, indent=2)

    print(f"\nTemplate created: {structures_path}")
    print(f"  {n_pillars} sample pillars placed across {ortho_w}x{ortho_h} orthomosaic")
    print(f"\nNext steps:")
    print(f"  1. Open {structures_path} and edit center_x/center_y for each pillar")
    print(f"     OR use the interactive marker tool:")
    print(f"  2. python 07_structure_tracker.py --mark --ortho data/processed/2024_01/orthomosaic_2024_01.png")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 7: Individual structure tracker")
    parser.add_argument("--period",    help="Period ID to track (e.g. 2024_01)")
    parser.add_argument("--prev",      default=None)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--config",    default="config/pipeline_config.yaml")
    parser.add_argument("--structures",default="config/structures.json")

    parser.add_argument("--mark",      action="store_true",
                        help="Open interactive marker tool to place structures")
    parser.add_argument("--ortho",     default=None,
                        help="Orthomosaic path for --mark mode")

    parser.add_argument("--create-template", action="store_true",
                        help="Create a template structures.json")
    parser.add_argument("--ortho-size", nargs=2, type=int, default=[1218, 639],
                        metavar=("W", "H"),
                        help="Orthomosaic width and height for template")
    parser.add_argument("--n-pillars", type=int, default=10,
                        help="Number of sample pillars in template")

    args = parser.parse_args()

    if args.create_template:
        create_template(args.structures, args.ortho_size[0], args.ortho_size[1],
                        args.n_pillars)

    elif args.mark:
        ortho_path = args.ortho
        if not ortho_path and args.period:
            ortho_path = f"data/processed/{args.period}/orthomosaic_{args.period}.png"
        if not ortho_path:
            print("ERROR: Provide --ortho <path> or --period <id>")
        else:
            run_interactive_marker(ortho_path, args.structures)

    elif args.period:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        track_structures(args.period, args.prev, args.processed, cfg, args.structures)

    else:
        parser.print_help()
