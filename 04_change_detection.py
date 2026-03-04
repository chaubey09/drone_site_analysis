"""
04_change_detection.py
──────────────────────
Step 4: Change Detection Between Periods

Compares current period vs previous period:
  - Aligns orthomosaics using feature homography
  - Diffs segmentation masks → per-class change maps
  - Detects: new construction, demolition, equipment movement
  - Produces: change heatmap, zone-level delta table

Output:
  - change_map_{prev}_{curr}.png      (color-coded changes)
  - change_heatmap_{prev}_{curr}.png  (density heatmap)
  - change_stats_{prev}_{curr}.json   (numerical deltas)
"""

import cv2
import json
import numpy as np
import os
from pathlib import Path
import logging
import yaml

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Change type colors (BGR)
CHANGE_COLORS = {
    "new_construction":   (0,   200, 0),    # Green
    "demolition_removal": (0,   0,   220),  # Red
    "equipment_movement": (0,   165, 255),  # Orange
    "earthwork_progress": (0,   200, 200),  # Yellow
    "no_change":          (50,  50,  50),   # Dark gray
}


# ─── IMAGE ALIGNMENT ──────────────────────────────────────────────────────────

def align_images(src: np.ndarray, dst: np.ndarray,
                 max_features: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """
    Align `src` to `dst` using SIFT + homography.
    Returns (aligned_src, homography_matrix).
    """
    log.info("Aligning orthomosaics via SIFT homography...")

    src_gray = cv2.cvtColor(cv2.resize(src, (2000, 2000)), cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(cv2.resize(dst, (2000, 2000)), cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(src_gray, None)
    kp2, des2 = sift.detectAndCompute(dst_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        log.warning("Not enough features for alignment. Using identity transform.")
        return cv2.resize(src, (dst.shape[1], dst.shape[0])), np.eye(3)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    log.info(f"Feature matches: {len(good)} good / {len(matches)} total")

    if len(good) < 10:
        log.warning("Too few good matches. Using identity transform.")
        return cv2.resize(src, (dst.shape[1], dst.shape[0])), np.eye(3)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum())
    log.info(f"Homography inliers: {inliers}/{len(good)}")

    if H is None:
        return cv2.resize(src, (dst.shape[1], dst.shape[0])), np.eye(3)

    # Scale H for full-resolution images
    scale_x = dst.shape[1] / 2000
    scale_y = dst.shape[0] / 2000
    S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
    S_inv = np.linalg.inv(np.array([[src.shape[1]/2000, 0, 0],
                                     [0, src.shape[0]/2000, 0],
                                     [0, 0, 1]]))
    H_full = S @ H @ S_inv

    aligned = cv2.warpPerspective(src, H_full, (dst.shape[1], dst.shape[0]))
    return aligned, H_full


def align_masks(src_mask: np.ndarray, dst_mask: np.ndarray,
                H: np.ndarray) -> np.ndarray:
    """Warp source mask using precomputed homography."""
    aligned = cv2.warpPerspective(src_mask, H,
                                   (dst_mask.shape[1], dst_mask.shape[0]),
                                   flags=cv2.INTER_NEAREST)
    return aligned


# ─── CHANGE DETECTION LOGIC ───────────────────────────────────────────────────

# Classes that indicate active construction progress
CONSTRUCTION_CLASSES = {1, 2, 3}   # concrete, rebar, formwork
EARTHWORK_CLASSES    = {4}          # earthwork
EQUIPMENT_CLASSES    = {6}          # equipment
MATERIAL_CLASSES     = {7}          # stockpiles
STABLE_CLASSES       = {5, 9}       # road, vegetation (shouldn't change)


def classify_change(prev_class: int, curr_class: int) -> str:
    """Determine change type from class transition."""
    if prev_class == curr_class:
        return "no_change"

    prev_is_construction = prev_class in CONSTRUCTION_CLASSES
    curr_is_construction = curr_class in CONSTRUCTION_CLASSES
    prev_is_earth = prev_class in EARTHWORK_CLASSES
    curr_is_earth = curr_class in EARTHWORK_CLASSES

    if not prev_is_construction and curr_is_construction:
        return "new_construction"
    if prev_is_construction and not curr_is_construction:
        return "demolition_removal"
    if (prev_class in EQUIPMENT_CLASSES) != (curr_class in EQUIPMENT_CLASSES):
        return "equipment_movement"
    if prev_is_earth or curr_is_earth:
        return "earthwork_progress"
    return "no_change"


def compute_change_map(prev_mask: np.ndarray, curr_mask: np.ndarray,
                        min_area_px: int = 50) -> tuple[np.ndarray, dict]:
    """
    Compute pixel-level change map and change statistics.

    Returns:
        change_viz: BGR image with colored change regions
        stats: dict with change counts and areas
    """
    assert prev_mask.shape == curr_mask.shape, "Masks must be same shape after alignment"

    H, W = prev_mask.shape
    change_viz = np.zeros((H, W, 3), dtype=np.uint8)
    change_type_map = np.full((H, W), "no_change", dtype=object)

    # Vectorized change classification
    changed_mask = (prev_mask != curr_mask)

    for change_type, color in CHANGE_COLORS.items():
        if change_type == "no_change":
            continue
        type_mask = np.zeros((H, W), dtype=bool)

        for py in range(max(1, H // 50)):    # Sample-based for speed
            for px in range(max(1, W // 50)):
                # Block-wise classification
                y1 = py * (H // max(1, H // 50))
                x1 = px * (W // max(1, W // 50))
                y2 = min(H, y1 + H // max(1, H // 50))
                x2 = min(W, x1 + W // max(1, W // 50))

        # Pixel-wise (manageable since mask is small)
        for r in range(H):
            for c in range(W):
                if changed_mask[r, c]:
                    ct = classify_change(int(prev_mask[r, c]), int(curr_mask[r, c]))
                    change_type_map[r, c] = ct

        break  # Reassign properly below

    # Efficient vectorized approach
    change_viz = np.full((H, W, 3), CHANGE_COLORS["no_change"], dtype=np.uint8)

    for change_type, color in CHANGE_COLORS.items():
        if change_type == "no_change":
            continue
        # Identify pixel pairs for this change type
        type_pixel_mask = np.zeros((H, W), dtype=bool)

        if change_type == "new_construction":
            prev_not_c = ~np.isin(prev_mask, list(CONSTRUCTION_CLASSES))
            curr_is_c  =  np.isin(curr_mask, list(CONSTRUCTION_CLASSES))
            type_pixel_mask = prev_not_c & curr_is_c

        elif change_type == "demolition_removal":
            prev_is_c  =  np.isin(prev_mask, list(CONSTRUCTION_CLASSES))
            curr_not_c = ~np.isin(curr_mask, list(CONSTRUCTION_CLASSES))
            type_pixel_mask = prev_is_c & curr_not_c

        elif change_type == "equipment_movement":
            prev_eq = np.isin(prev_mask, list(EQUIPMENT_CLASSES))
            curr_eq = np.isin(curr_mask, list(EQUIPMENT_CLASSES))
            type_pixel_mask = (prev_eq | curr_eq) & (prev_mask != curr_mask)

        elif change_type == "earthwork_progress":
            prev_e = np.isin(prev_mask, list(EARTHWORK_CLASSES))
            curr_e = np.isin(curr_mask, list(EARTHWORK_CLASSES))
            type_pixel_mask = (prev_e | curr_e) & (prev_mask != curr_mask)

        # Remove small noise patches
        type_pixel_mask = type_pixel_mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        type_pixel_mask = cv2.morphologyEx(type_pixel_mask, cv2.MORPH_OPEN, kernel)

        # Count connected components and filter small ones
        num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(type_pixel_mask)
        for label in range(1, num_labels):
            if stats_cc[label, cv2.CC_STAT_AREA] >= min_area_px:
                change_viz[labels == label] = color

    # Compute area statistics
    total_px = H * W
    stats = {}
    for change_type in CHANGE_COLORS:
        if change_type == "no_change":
            continue
        color = CHANGE_COLORS[change_type]
        # Find pixels of this color
        match = np.all(change_viz == np.array(color, dtype=np.uint8), axis=2)
        px_count = int(match.sum())
        stats[change_type] = {
            "pixels": px_count,
            "percent_of_site": round(100 * px_count / total_px, 3),
        }

    unchanged_px = int(np.all(change_viz == np.array(CHANGE_COLORS["no_change"]), axis=2).sum())
    stats["no_change"] = {
        "pixels": unchanged_px,
        "percent_of_site": round(100 * unchanged_px / total_px, 2),
    }

    return change_viz, stats


# ─── CHANGE HEATMAP ───────────────────────────────────────────────────────────

def generate_change_heatmap(change_viz: np.ndarray) -> np.ndarray:
    """Generate smooth density heatmap of all changed areas."""
    # Any non-background pixel = changed
    bg_color = np.array(CHANGE_COLORS["no_change"], dtype=np.uint8)
    changed = (~np.all(change_viz == bg_color, axis=2)).astype(np.float32)

    # Gaussian blur for density estimation
    density = cv2.GaussianBlur(changed, (0, 0), sigmaX=30)
    density = (density / density.max() * 255).astype(np.uint8) if density.max() > 0 else density.astype(np.uint8)

    heatmap = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    return heatmap


# ─── SIDE-BY-SIDE COMPARISON ─────────────────────────────────────────────────

def create_comparison_image(prev_ortho: np.ndarray, curr_ortho: np.ndarray,
                             change_viz: np.ndarray,
                             prev_id: str, curr_id: str) -> np.ndarray:
    """Create a 3-panel side-by-side comparison image."""
    H = 800
    def resize_h(img, h):
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h))

    p = resize_h(prev_ortho, H)
    c = resize_h(curr_ortho, H)
    d = resize_h(change_viz, H)

    # Add labels
    for img, label in [(p, prev_id), (c, curr_id), (d, "Changes")]:
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                    (0, 0, 0), 1, cv2.LINE_AA)

    # Add color legend to change image
    legend_items = [(k, v) for k, v in CHANGE_COLORS.items() if k != "no_change"]
    for i, (name, color) in enumerate(legend_items):
        y = 60 + i * 25
        cv2.rectangle(d, (10, y), (25, y + 18), color, -1)
        cv2.putText(d, name.replace("_", " "), (30, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    comparison = np.hstack([p, c, d])
    return comparison


# ─── ZONE-LEVEL DELTA ─────────────────────────────────────────────────────────

def compute_zone_deltas(prev_stats: dict, curr_stats: dict, zones: list[dict]) -> list[dict]:
    """
    Compute per-zone construction progress delta.
    Returns list of zone delta dicts.
    """
    deltas = []
    for zone in zones:
        zid = zone["id"]
        prev_zone = prev_stats.get("zone_areas", {}).get(zid, {})
        curr_zone = curr_stats.get("zone_areas", {}).get(zid, {})

        prev_areas = prev_zone.get("areas", {})
        curr_areas = curr_zone.get("areas", {})

        # Construction progress = concrete + rebar + formwork area
        def construction_m2(areas):
            return sum(areas.get(c, {}).get("area_m2", 0)
                       for c in ["concrete_structure", "steel_rebar", "formwork"])

        prev_c = construction_m2(prev_areas)
        curr_c = construction_m2(curr_areas)
        delta_c = curr_c - prev_c
        pct_change = round(100 * delta_c / max(prev_c, 1), 1)

        deltas.append({
            "zone_id":              zid,
            "zone_name":            zone["name"],
            "prev_construction_m2": round(prev_c, 1),
            "curr_construction_m2": round(curr_c, 1),
            "delta_m2":             round(delta_c, 1),
            "percent_change":       pct_change,
            "status":               "progressing" if delta_c > 10 else
                                    "stalled"     if delta_c > -10 else
                                    "regressed",
        })

    return deltas


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def detect_changes(prev_period: str, curr_period: str,
                   processed_dir: str, config: dict) -> str:
    """
    Run change detection between two periods.
    Returns path to change map PNG.
    """
    cfg_cd = config["change_detection"]
    out_dir = os.path.join(processed_dir, f"changes_{prev_period}_vs_{curr_period}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def load_ortho(period):
        path = os.path.join(processed_dir, period, f"orthomosaic_{period}.png")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Orthomosaic not found: {path}")
        return img

    def load_mask(period):
        path = os.path.join(processed_dir, period, f"segmentation_mask_{period}.png")
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Segmentation mask not found: {path}")
        return mask

    def load_seg_stats(period):
        path = os.path.join(processed_dir, period, f"class_areas_{period}.json")
        with open(path) as f:
            return json.load(f)

    log.info(f"Loading data for periods: {prev_period} → {curr_period}")
    prev_ortho = load_ortho(prev_period)
    curr_ortho = load_ortho(curr_period)
    prev_mask  = load_mask(prev_period)
    curr_mask  = load_mask(curr_period)
    prev_stats = load_seg_stats(prev_period)
    curr_stats = load_seg_stats(curr_period)

    # Resize to common size
    target_size = (curr_ortho.shape[1], curr_ortho.shape[0])
    prev_ortho_r = cv2.resize(prev_ortho, target_size)
    prev_mask_r  = cv2.resize(prev_mask,  target_size, interpolation=cv2.INTER_NEAREST)
    curr_mask_r  = cv2.resize(curr_mask,  target_size, interpolation=cv2.INTER_NEAREST)

    # Align
    if cfg_cd["alignment_method"] == "homography":
        _, H_mat = align_images(prev_ortho_r, curr_ortho)
        prev_mask_aligned = align_masks(prev_mask_r, curr_mask_r, H_mat)
    else:
        prev_mask_aligned = prev_mask_r

    # Change detection
    min_area_px = max(1, int(cfg_cd["min_change_area_sqm"] /
                             (config["georeferencing"]["output_resolution_cm_per_px"] / 100) ** 2))
    log.info(f"Computing change map (min area: {min_area_px}px)...")
    change_viz, change_stats = compute_change_map(prev_mask_aligned, curr_mask_r, min_area_px)

    # Heatmap
    heatmap = generate_change_heatmap(change_viz)

    # Comparison image
    comparison = create_comparison_image(prev_ortho_r, curr_ortho, change_viz,
                                          prev_period, curr_period)

    # Zone deltas
    zone_deltas = compute_zone_deltas(prev_stats, curr_stats, config.get("zones", []))

    # Save outputs
    change_map_path    = os.path.join(out_dir, f"change_map_{prev_period}_{curr_period}.png")
    heatmap_path       = os.path.join(out_dir, f"change_heatmap_{prev_period}_{curr_period}.png")
    comparison_path    = os.path.join(out_dir, f"comparison_{prev_period}_{curr_period}.jpg")
    change_stats_path  = os.path.join(out_dir, f"change_stats_{prev_period}_{curr_period}.json")

    cv2.imwrite(change_map_path, change_viz)
    cv2.imwrite(heatmap_path, heatmap)
    cv2.imwrite(comparison_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 90])

    full_stats = {
        "prev_period": prev_period,
        "curr_period": curr_period,
        "change_map_path": change_map_path,
        "heatmap_path": heatmap_path,
        "comparison_path": comparison_path,
        "change_stats": change_stats,
        "zone_deltas": zone_deltas,
        "prev_global_areas": prev_stats.get("global_areas", {}),
        "curr_global_areas": curr_stats.get("global_areas", {}),
    }
    with open(change_stats_path, "w") as f:
        json.dump(full_stats, f, indent=2)

    log.info(f"✓ Change detection complete: {prev_period} → {curr_period}")
    for ct, s in change_stats.items():
        if ct != "no_change":
            log.info(f"  {ct:25s}: {s['pixels']:>8,}px ({s['percent_of_site']:.2f}%)")
    log.info(f"  Zone deltas:")
    for zd in zone_deltas:
        log.info(f"    [{zd['zone_id']}] {zd['zone_name']}: "
                 f"{zd['delta_m2']:+.0f} m² construction ({zd['status']})")

    return change_map_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 4: Change detection")
    parser.add_argument("--prev",      required=True, help="Previous period ID")
    parser.add_argument("--curr",      required=True, help="Current period ID")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--config",    default="config/pipeline_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    detect_changes(args.prev, args.curr, args.processed, cfg)
