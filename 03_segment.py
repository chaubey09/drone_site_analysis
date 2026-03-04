"""
03_segment.py
─────────────
Step 3: Semantic Segmentation

Detects and classifies construction elements on orthomosaic:
  - Concrete structures (pillars, beams, slabs)
  - Steel rebar
  - Formwork / shuttering
  - Earthwork / excavation
  - Equipment (cranes, JCBs)
  - Material stockpiles
  - Workers
  - Vegetation, roads (baseline unchanged)

Two modes:
  A) Pretrained model (Mask2Former / SAM fine-tuned on construction)
  B) HSV + texture heuristics (no model needed — good starting baseline)

Output:
  - segmentation_mask_{period}.png   (color-coded class map)
  - class_areas_{period}.json        (pixel counts per class)
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

# Class ID → RGB color for visualization
CLASS_COLORS = {
    0:  (50,  50,  50),    # background         → dark gray
    1:  (180, 180, 180),   # concrete_structure → light gray
    2:  (150, 75,  0),     # steel_rebar        → brown
    3:  (255, 165, 0),     # formwork           → orange
    4:  (139, 90,  43),    # earthwork          → dirt brown
    5:  (60,  60,  60),    # road_surface       → charcoal
    6:  (255, 50,  50),    # equipment          → red
    7:  (255, 200, 0),     # material_stockpile → yellow
    8:  (0,   100, 255),   # water_body         → blue
    9:  (34,  139, 34),    # vegetation         → green
    10: (255, 105, 180),   # workers            → pink
}

CLASS_NAMES = {
    0: "background", 1: "concrete_structure", 2: "steel_rebar",
    3: "formwork", 4: "earthwork", 5: "road_surface",
    6: "equipment", 7: "material_stockpile", 8: "water_body",
    9: "vegetation", 10: "workers"
}


# ─── HEURISTIC SEGMENTER (no ML model required) ───────────────────────────────

class HeuristicSegmenter:
    """
    HSV + texture-based heuristic segmentation.
    Works out-of-the-box with no training. Less accurate than deep learning
    but provides a solid baseline and progress proxy.
    """

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Returns a mask array of shape (H, W) with integer class IDs.
        """
        H, W = image.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        H_ch, S_ch, V_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # ── Vegetation (green) ──────────────────────────────────────────────
        veg = (
            (H_ch >= 35) & (H_ch <= 85) &
            (S_ch >= 40) & (V_ch >= 40)
        )
        mask[veg] = 9

        # ── Water (blue) ────────────────────────────────────────────────────
        water = (
            (H_ch >= 100) & (H_ch <= 130) &
            (S_ch >= 50) & (V_ch >= 50)
        )
        mask[water] = 8

        # ── Equipment (bright red/orange machinery) ──────────────────────
        equip = (
            ((H_ch <= 10) | (H_ch >= 170)) &
            (S_ch >= 100) & (V_ch >= 100)
        ) | (
            (H_ch >= 10) & (H_ch <= 25) &
            (S_ch >= 150) & (V_ch >= 100)
        )
        mask[equip] = 6

        # ── Concrete structures (light gray, low saturation, high brightness)
        concrete = (
            (S_ch < 30) & (V_ch >= 150) & (V_ch <= 240)
        )
        mask[concrete] = 1

        # ── Road surface (dark gray, very low saturation) ───────────────
        road = (
            (S_ch < 20) & (V_ch >= 30) & (V_ch < 120)
        )
        mask[road] = 5

        # ── Earthwork (brown/ochre bare soil) ───────────────────────────
        earth = (
            (H_ch >= 10) & (H_ch <= 35) &
            (S_ch >= 30) & (S_ch <= 180) &
            (V_ch >= 60) & (V_ch <= 180)
        )
        mask[earth] = 4

        # ── Material stockpile (sandy yellow, high value) ───────────────
        stockpile = (
            (H_ch >= 20) & (H_ch <= 40) &
            (S_ch >= 50) & (S_ch <= 180) &
            (V_ch >= 160)
        )
        mask[stockpile] = 7

        # ── Rebar / steel (dark brown-gray with texture) ─────────────────
        # Use edge density as proxy for rebar texture
        edges = cv2.Canny(gray, 80, 180)
        kernel = np.ones((5, 5), np.uint8)
        edge_density = cv2.dilate(edges, kernel)
        rebar = (
            (edge_density > 0) & (S_ch < 40) &
            (V_ch >= 30) & (V_ch < 120) &
            (mask == 0)
        )
        mask[rebar] = 2

        # ── Formwork (wooden/plywood — warm mid-brightness) ─────────────
        formwork = (
            (H_ch >= 15) & (H_ch <= 30) &
            (S_ch >= 60) & (S_ch <= 160) &
            (V_ch >= 80) & (V_ch <= 160) &
            (mask == 0)
        )
        mask[formwork] = 3

        # Remaining unclassified → background
        mask[mask == 0] = 0

        return mask

    def segment_large_image(self, image: np.ndarray, tile_size: int = 1024,
                             overlap: int = 64) -> np.ndarray:
        """Tile-based segmentation for large orthomosaics."""
        H, W = image.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)

        for y in range(0, H, tile_size - overlap):
            for x in range(0, W, tile_size - overlap):
                y2 = min(y + tile_size, H)
                x2 = min(x + tile_size, W)
                tile = image[y:y2, x:x2]
                tile_mask = self.segment(tile)
                # Only write to non-overlap area to avoid seams
                wy1 = overlap // 2 if y > 0 else 0
                wx1 = overlap // 2 if x > 0 else 0
                mask[y + wy1:y2, x + wx1:x2] = tile_mask[wy1:, wx1:]

        return mask


# ─── DEEP LEARNING SEGMENTER ──────────────────────────────────────────────────

class DeepSegmenter:
    """
    Wrapper for a fine-tuned segmentation model (Mask2Former or SAM).
    
    To use:
    1. Fine-tune Mask2Former on construction imagery using Hugging Face transformers
    2. Save model to models/segmentation_model/
    3. Set config segmentation.model = "mask2former"
    
    Training data tips:
    - Use ~500-1000 labeled orthomosaic patches (256×256 px)
    - Label using CVAT, Labelme, or Roboflow
    - Augment with rotations, brightness shifts (drone view varies a lot)
    """

    def __init__(self, model_weights: str, num_classes: int = 11, device: str = "cuda"):
        self.device = device
        self.num_classes = num_classes
        self._load_model(model_weights)

    def _load_model(self, weights_path: str):
        try:
            import torch
            from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

            self.processor = AutoImageProcessor.from_pretrained(weights_path)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(weights_path)
            self.model.eval()
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                else:
                    self.device = "cpu"
            log.info(f"Loaded Mask2Former from {weights_path} on {self.device}")
        except ImportError:
            log.error("transformers/torch not installed. Use: pip install transformers torch")
            raise
        except Exception as e:
            log.error(f"Model load failed: {e}. Falling back to heuristic segmenter.")
            raise

    def segment(self, image: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil_img, return_tensors="pt")

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        pred = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.shape[:2]]
        )[0]

        return pred.cpu().numpy().astype(np.uint8)

    def segment_large_image(self, image: np.ndarray, tile_size: int = 512,
                             overlap: int = 64) -> np.ndarray:
        H, W = image.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        for y in range(0, H, tile_size - overlap):
            for x in range(0, W, tile_size - overlap):
                y2 = min(y + tile_size, H)
                x2 = min(x + tile_size, W)
                tile = image[y:y2, x:x2]
                tile_mask = self.segment(tile)
                wy1 = overlap // 2 if y > 0 else 0
                wx1 = overlap // 2 if x > 0 else 0
                mask[y + wy1:y2, x + wx1:x2] = tile_mask[wy1:, wx1:]
        return mask


# ─── VISUALIZATION ────────────────────────────────────────────────────────────

def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert integer class mask to BGR color image."""
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, rgb in CLASS_COLORS.items():
        color[mask == cls_id] = rgb[::-1]  # RGB→BGR
    return color


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend original image with color mask."""
    color_mask = mask_to_color(mask)
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)


def draw_legend(image: np.ndarray) -> np.ndarray:
    """Add class legend to image."""
    img = image.copy()
    x, y = 10, 10
    box_size = 20
    for cls_id, name in CLASS_NAMES.items():
        rgb = CLASS_COLORS[cls_id]
        bgr = rgb[::-1]
        cv2.rectangle(img, (x, y), (x + box_size, y + box_size), bgr, -1)
        cv2.putText(img, name, (x + box_size + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 28
    return img


# ─── COMPUTE CLASS AREAS ──────────────────────────────────────────────────────

def compute_class_areas(mask: np.ndarray, m_per_px: float = 0.05) -> dict:
    """
    Returns area in m² per class.
    m_per_px: meters per pixel (default 5cm/px = 0.05 m/px)
    """
    areas = {}
    total_px = mask.size
    if total_px == 0:
        # Empty mask (zone out of bounds) — return zeros for all classes
        for name in CLASS_NAMES.values():
            areas[name] = {"pixels": 0, "area_m2": 0.0, "percent": 0.0}
        return areas
    px_area_sqm = m_per_px ** 2
    for cls_id, name in CLASS_NAMES.items():
        px_count = int(np.sum(mask == cls_id))
        areas[name] = {
            "pixels": px_count,
            "area_m2": round(px_count * px_area_sqm, 1),
            "percent": round(100 * px_count / total_px, 2)
        }
    return areas


# ─── ZONE-LEVEL BREAKDOWN ─────────────────────────────────────────────────────

def compute_zone_areas(mask: np.ndarray, zones: list[dict],
                        m_per_px: float = 0.05) -> dict:
    """Compute per-zone class area breakdown."""
    results = {}
    H, W = mask.shape[:2]
    for zone in zones:
        zid = zone["id"]
        x1, y1, x2, y2 = zone["bbox"]
        # Clip to mask dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            # Zone bbox is entirely outside the orthomosaic — skip gracefully
            log.warning(f"Zone {zid} bbox {zone['bbox']} is outside orthomosaic "
                        f"({W}x{H}). Zone will show zero areas. "
                        f"Update the bbox in pipeline_config.yaml.")
            results[zid] = {
                "name": zone["name"],
                "areas": compute_class_areas(np.zeros((0,), dtype=np.uint8), m_per_px)
            }
            continue

        zone_mask = mask[y1:y2, x1:x2]
        results[zid] = {
            "name": zone["name"],
            "areas": compute_class_areas(zone_mask, m_per_px)
        }
    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def segment_period(period_id: str, processed_dir: str, config: dict) -> str:
    """
    Run segmentation on the orthomosaic for a given period.
    Returns path to segmentation mask PNG.
    """
    cfg_seg = config["segmentation"]
    cfg_geo = config["georeferencing"]

    ortho_path = os.path.join(processed_dir, period_id, f"orthomosaic_{period_id}.png")
    out_dir = os.path.join(processed_dir, period_id)

    log.info(f"Loading orthomosaic: {ortho_path}")
    image = cv2.imread(ortho_path)
    if image is None:
        raise FileNotFoundError(f"Orthomosaic not found: {ortho_path}")

    # Choose segmenter
    model_type = cfg_seg.get("model", "heuristic")
    if model_type == "heuristic" or not os.path.exists(cfg_seg.get("model_weights", "")):
        log.info("Using heuristic segmenter")
        segmenter = HeuristicSegmenter()
    else:
        log.info(f"Using deep learning segmenter: {cfg_seg['model']}")
        try:
            segmenter = DeepSegmenter(cfg_seg["model_weights"])
        except Exception:
            log.warning("Deep model failed, falling back to heuristic")
            segmenter = HeuristicSegmenter()

    log.info(f"Segmenting {image.shape[1]}×{image.shape[0]} orthomosaic...")
    mask = segmenter.segment_large_image(image)

    # Save raw mask (single-channel class IDs)
    mask_path = os.path.join(out_dir, f"segmentation_mask_{period_id}.png")
    cv2.imwrite(mask_path, mask)

    # Save colored visualization
    color_mask = mask_to_color(mask)
    overlay = overlay_mask(image, mask)
    overlay_legend = draw_legend(overlay)
    overlay_path = os.path.join(out_dir, f"segmentation_overlay_{period_id}.jpg")
    cv2.imwrite(overlay_path, overlay_legend)

    # Compute areas
    m_per_px = cfg_geo.get("output_resolution_cm_per_px", 5) / 100.0
    areas = compute_class_areas(mask, m_per_px)
    zone_areas = compute_zone_areas(mask, config.get("zones", []), m_per_px)

    stats = {
        "period_id": period_id,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "global_areas": areas,
        "zone_areas": zone_areas,
    }
    stats_path = os.path.join(out_dir, f"class_areas_{period_id}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"✓ Segmentation complete for {period_id}")
    log.info("  Class areas (m²):")
    for cls, d in areas.items():
        if d["area_m2"] > 0:
            log.info(f"    {cls:25s}: {d['area_m2']:>10.0f} m² ({d['percent']:.1f}%)")

    return mask_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 3: Semantic segmentation")
    parser.add_argument("--period",    required=True)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--config",    default="config/pipeline_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    segment_period(args.period, args.processed, cfg)