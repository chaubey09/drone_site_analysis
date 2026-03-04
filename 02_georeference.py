"""
02_georeference.py
──────────────────
Step 2: Georeferencing & Orthomosaic Generation

Stitches extracted frames into a single georeferenced orthomosaic image
that represents the entire construction site as a top-down map.

Two modes:
  A) GPS-based:   Uses GPS coords from SRT/metadata to place tiles
  B) Feature-based: SIFT/ORB feature matching to stitch overlapping frames

Output:
  - orthomosaic_{period}.tif  (GeoTIFF if GPS available, PNG otherwise)
  - orthomosaic_{period}.png  (visual preview)
  - stitch_metadata.json
"""

import cv2
import json
import numpy as np
import os
from pathlib import Path
from typing import Optional
import logging
import yaml
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── GPS → PIXEL MAPPING ──────────────────────────────────────────────────────

class GPSMapper:
    """
    Maps GPS coordinates to pixel positions on the orthomosaic canvas.
    Uses a simple equirectangular projection suitable for small areas (<10km).
    """
    def __init__(self, frame_metas: list[dict], canvas_resolution_cm_per_px: float = 5,
                 avg_altitude_m: float = 80):
        self.metas = [m for m in frame_metas if m.get("lat") and m.get("lon")]
        if not self.metas:
            raise ValueError("No GPS data in frame metadata")

        self.lats = [m["lat"] for m in self.metas]
        self.lons = [m["lon"] for m in self.metas]
        self.lat_min, self.lat_max = min(self.lats), max(self.lats)
        self.lon_min, self.lon_max = min(self.lons), max(self.lons)

        # Ground coverage per pixel
        self.m_per_px = canvas_resolution_cm_per_px / 100.0

        # Earth radius for small-area approximation
        R = 6371000
        lat_mid = np.radians((self.lat_min + self.lat_max) / 2)
        self.m_per_deg_lat = R * np.pi / 180
        self.m_per_deg_lon = R * np.cos(lat_mid) * np.pi / 180

        # Canvas size in pixels
        width_m = (self.lon_max - self.lon_min) * self.m_per_deg_lon
        height_m = (self.lat_max - self.lat_min) * self.m_per_deg_lat
        self.canvas_w = max(int(width_m / self.m_per_px) + 200, 100)
        self.canvas_h = max(int(height_m / self.m_per_px) + 200, 100)

        # FOV-based footprint estimate at given altitude
        self.altitude_m = avg_altitude_m
        self.fov_h_deg = 84    # DJI typical horizontal FOV
        self.fov_v_deg = 53    # DJI typical vertical FOV
        self.footprint_w_m = 2 * avg_altitude_m * np.tan(np.radians(self.fov_h_deg / 2))
        self.footprint_h_m = 2 * avg_altitude_m * np.tan(np.radians(self.fov_v_deg / 2))

        log.info(f"Canvas: {self.canvas_w}×{self.canvas_h}px covering "
                 f"{width_m:.0f}m × {height_m:.0f}m at {canvas_resolution_cm_per_px}cm/px")

    def gps_to_pixel(self, lat: float, lon: float) -> tuple[int, int]:
        x = int((lon - self.lon_min) * self.m_per_deg_lon / self.m_per_px)
        y = int((self.lat_max - lat) * self.m_per_deg_lat / self.m_per_px)
        return x, y

    def frame_footprint_px(self) -> tuple[int, int]:
        fw = int(self.footprint_w_m / self.m_per_px)
        fh = int(self.footprint_h_m / self.m_per_px)
        return fw, fh


# ─── GPS-BASED MOSAIC ─────────────────────────────────────────────────────────

def build_gps_orthomosaic(frame_metas: list[dict], mapper: GPSMapper,
                          output_path: str) -> np.ndarray:
    """Place resized frames on canvas at their GPS positions."""
    log.info("Building GPS-positioned orthomosaic...")
    canvas = np.zeros((mapper.canvas_h, mapper.canvas_w, 3), dtype=np.uint8)
    weight = np.zeros((mapper.canvas_h, mapper.canvas_w), dtype=np.float32)

    fw, fh = mapper.frame_footprint_px()

    gps_metas = [m for m in frame_metas if m.get("lat") and m.get("lon")]
    for meta in tqdm(gps_metas, desc="Placing frames"):
        img = cv2.imread(meta["filepath"])
        if img is None:
            continue
        tile = cv2.resize(img, (fw, fh))

        cx, cy = mapper.gps_to_pixel(meta["lat"], meta["lon"])
        x1, y1 = cx - fw // 2, cy - fh // 2
        x2, y2 = x1 + fw, y1 + fh

        # Clip to canvas
        sx1 = max(0, -x1); sy1 = max(0, -y1)
        ex1 = min(fw, mapper.canvas_w - x1)
        ey1 = min(fh, mapper.canvas_h - y1)
        cx1 = max(0, x1); cy1 = max(0, y1)
        cx2 = cx1 + (ex1 - sx1); cy2 = cy1 + (ey1 - sy1)

        if cx2 <= cx1 or cy2 <= cy1:
            continue

        # Weighted blend (favor later/higher frames by incrementing weight)
        w_slice = weight[cy1:cy2, cx1:cx2]
        c_slice = canvas[cy1:cy2, cx1:cx2].astype(np.float32)
        t_slice = tile[sy1:ey1, sx1:ex1].astype(np.float32)

        new_w = w_slice + 1
        canvas[cy1:cy2, cx1:cx2] = ((c_slice * w_slice[..., None] + t_slice) /
                                     new_w[..., None]).astype(np.uint8)
        weight[cy1:cy2, cx1:cx2] = new_w

    cv2.imwrite(output_path, canvas)
    log.info(f"GPS orthomosaic saved: {output_path}")
    return canvas


# ─── FEATURE-BASED MOSAIC ─────────────────────────────────────────────────────

def build_feature_orthomosaic(frame_metas: list[dict], output_path: str,
                               max_frames: int = 200) -> np.ndarray:
    """
    SIFT-based image stitching for when GPS is unavailable or imprecise.
    Uses OpenCV's Stitcher for robustness.
    Subsample frames to keep stitching tractable.
    """
    log.info("Building feature-matched orthomosaic (SIFT stitching)...")

    # Subsample evenly
    step = max(1, len(frame_metas) // max_frames)
    selected = frame_metas[::step][:max_frames]
    log.info(f"Using {len(selected)} frames for stitching")

    images = []
    for meta in tqdm(selected, desc="Loading frames"):
        img = cv2.imread(meta["filepath"])
        if img is not None:
            # Downscale for speed
            images.append(cv2.resize(img, (960, 540)))

    if len(images) < 2:
        raise RuntimeError("Not enough valid frames for stitching")

    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    stitcher.setPanoConfidenceThresh(0.5)

    status, mosaic = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        log.warning(f"Stitcher returned status {status}. Falling back to simple grid mosaic.")
        mosaic = _fallback_grid_mosaic(images, output_path)
    else:
        cv2.imwrite(output_path, mosaic)
        log.info(f"Feature orthomosaic saved: {output_path} ({mosaic.shape[1]}×{mosaic.shape[0]})")

    return mosaic


def _fallback_grid_mosaic(images: list[np.ndarray], output_path: str) -> np.ndarray:
    """Last resort: arrange frames in a grid (no stitching)."""
    log.info("Using fallback grid layout")
    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h, w = images[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img
    cv2.imwrite(output_path, canvas)
    return canvas


# ─── GEOTIFF EXPORT ───────────────────────────────────────────────────────────

def export_geotiff(mosaic: np.ndarray, mapper: GPSMapper, output_path: str):
    """
    Export orthomosaic as GeoTIFF with embedded CRS and transform.
    Requires rasterio.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        transform = from_bounds(
            west=mapper.lon_min, south=mapper.lat_min,
            east=mapper.lon_max, north=mapper.lat_max,
            width=mosaic.shape[1], height=mosaic.shape[0]
        )
        crs = CRS.from_epsg(4326)  # WGS84

        with rasterio.open(
            output_path, "w", driver="GTiff",
            height=mosaic.shape[0], width=mosaic.shape[1],
            count=3, dtype=mosaic.dtype,
            crs=crs, transform=transform
        ) as dst:
            # rasterio: channels first
            dst.write(mosaic[:, :, 2], 1)  # R
            dst.write(mosaic[:, :, 1], 2)  # G
            dst.write(mosaic[:, :, 0], 3)  # B

        log.info(f"GeoTIFF exported: {output_path}")
    except ImportError:
        log.warning("rasterio not installed. Skipping GeoTIFF export. "
                    "Install with: pip install rasterio")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def georeference(period_id: str, processed_dir: str, config: dict) -> str:
    """
    Full georeferencing pipeline for one period.
    Returns path to output orthomosaic PNG.
    """
    cfg_geo = config["georeferencing"]
    frames_dir = os.path.join(processed_dir, period_id, "frames")
    out_dir = os.path.join(processed_dir, period_id)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load frame metadata
    meta_path = os.path.join(frames_dir, f"{period_id}_frame_metadata.json")
    with open(meta_path) as f:
        frame_metas = json.load(f)

    log.info(f"Georeferencing {len(frame_metas)} frames for period {period_id}")

    mosaic_png = os.path.join(out_dir, f"orthomosaic_{period_id}.png")
    mosaic_tif = os.path.join(out_dir, f"orthomosaic_{period_id}.tif")

    method = cfg_geo["method"]
    has_gps = any(m.get("lat") for m in frame_metas)

    if method in ("gps_metadata", "hybrid") and has_gps:
        avg_alt = np.mean([m["altitude_m"] for m in frame_metas if m.get("altitude_m")]) or 80
        mapper = GPSMapper(frame_metas,
                           canvas_resolution_cm_per_px=cfg_geo["output_resolution_cm_per_px"],
                           avg_altitude_m=avg_alt)
        mosaic = build_gps_orthomosaic(frame_metas, mapper, mosaic_png)
        export_geotiff(mosaic, mapper, mosaic_tif)

    elif method in ("feature_matching", "hybrid"):
        mosaic = build_feature_orthomosaic(frame_metas, mosaic_png)

    else:
        raise ValueError(f"Unknown georeferencing method: {method}")

    # Save stitch info
    info = {
        "period_id": period_id,
        "method": method,
        "frames_used": len(frame_metas),
        "orthomosaic_png": mosaic_png,
        "orthomosaic_tif": mosaic_tif,
        "mosaic_shape": list(mosaic.shape[:2]),
    }
    with open(os.path.join(out_dir, "stitch_metadata.json"), "w") as f:
        json.dump(info, f, indent=2)

    log.info(f"✓ Georeferencing complete for {period_id}")
    return mosaic_png


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 2: Georeferencing & orthomosaic")
    parser.add_argument("--period",    required=True)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--config",    default="config/pipeline_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    georeference(args.period, args.processed, cfg)
