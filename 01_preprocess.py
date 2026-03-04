"""
01_preprocess.py
────────────────
Step 1: Video Preprocessing & Frame Extraction

- Extracts frames at configurable intervals from 30-min 30fps drone video
- Filters blurry/duplicate frames
- Applies video stabilization
- Parses GPS/SRT metadata from drone footage
- Outputs: clean frames + metadata JSON

30 min × 30 fps = 54,000 raw frames → ~900 useful frames (1 per 2 sec)
"""

import cv2
import json
import os
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import yaml
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class FrameMeta:
    frame_idx: int
    timestamp_sec: float
    filepath: str
    blur_score: float
    lat: Optional[float] = None
    lon: Optional[float] = None
    altitude_m: Optional[float] = None
    heading_deg: Optional[float] = None


def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── GPS / SRT PARSER ─────────────────────────────────────────────────────────

def parse_dji_srt(srt_path: str) -> dict[float, dict]:
    """
    Parse DJI-style SRT subtitle file that contains GPS metadata per frame.
    Returns dict: {timestamp_sec: {lat, lon, alt, heading}}
    """
    gps_data = {}
    if not os.path.exists(srt_path):
        log.warning(f"SRT file not found: {srt_path}. GPS will be None.")
        return gps_data

    with open(srt_path, "r") as f:
        content = f.read()

    # DJI SRT format: each block has timestamp and GPS line
    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            # Timecode line e.g.: 00:00:02,000 --> 00:00:03,000
            tc_match = re.search(r"(\d+):(\d+):(\d+),(\d+)", lines[1])
            if not tc_match:
                continue
            h, m, s, ms = map(int, tc_match.groups())
            ts = h * 3600 + m * 60 + s + ms / 1000.0

            # GPS line e.g.: [iso:100] [shutter:1/1000] [fnum:2.8] [ev:0] [ct:5500]
            #                [color_md:D_LOG] [focal_len:24] [latitude: 28.6139] [longitude: 77.2090] [altitude: 45.3]
            gps_line = " ".join(lines[2:])
            lat = float(re.search(r"\[latitude:\s*([-\d.]+)\]", gps_line).group(1))
            lon = float(re.search(r"\[longitude:\s*([-\d.]+)\]", gps_line).group(1))
            alt = float(re.search(r"\[altitude:\s*([-\d.]+)\]", gps_line).group(1))
            heading_match = re.search(r"\[heading:\s*([-\d.]+)\]", gps_line)
            heading = float(heading_match.group(1)) if heading_match else None
            gps_data[ts] = {"lat": lat, "lon": lon, "altitude_m": alt, "heading_deg": heading}
        except (AttributeError, ValueError):
            continue

    log.info(f"Parsed {len(gps_data)} GPS entries from SRT")
    return gps_data


def get_nearest_gps(gps_data: dict, timestamp: float) -> dict:
    """Find GPS entry closest in time to given timestamp."""
    if not gps_data:
        return {}
    nearest_ts = min(gps_data.keys(), key=lambda t: abs(t - timestamp))
    if abs(nearest_ts - timestamp) < 5.0:   # within 5 seconds
        return gps_data[nearest_ts]
    return {}


# ─── BLUR DETECTION ───────────────────────────────────────────────────────────

def laplacian_blur_score(frame: np.ndarray) -> float:
    """Higher = sharper. Below threshold → blurry."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─── FRAME SIMILARITY (SSIM proxy) ────────────────────────────────────────────

def frame_too_similar(frame: np.ndarray, prev_frame: np.ndarray, threshold: float) -> bool:
    """Fast mean-absolute-difference check instead of full SSIM for speed."""
    if prev_frame is None:
        return False
    f1 = cv2.resize(frame, (320, 180)).astype(np.float32)
    f2 = cv2.resize(prev_frame, (320, 180)).astype(np.float32)
    diff = np.mean(np.abs(f1 - f2)) / 255.0
    return diff < (1.0 - threshold)  # if very similar → diff is small


# ─── VIDEO STABILIZATION ──────────────────────────────────────────────────────

class VideoStabilizer:
    """
    Simple 2-pass stabilization using optical flow.
    For production: consider vidstab library or FFmpeg vidstabdetect.
    """
    def __init__(self, smoothing_radius: int = 30):
        self.smoothing_radius = smoothing_radius

    def stabilize_video(self, input_path: str, output_path: str):
        log.info("Stabilizing video (this may take a while)...")
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pass 1: compute transforms
        transforms = []
        ret, prev = cap.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        for _ in tqdm(range(total - 1), desc="Stabilize pass 1"):
            ret, curr = cap.read()
            if not ret:
                break
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                               qualityLevel=0.01, minDistance=30)
            if prev_pts is None:
                transforms.append((0, 0, 0))
                prev_gray = curr_gray
                continue
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            good_prev = prev_pts[status == 1]
            good_curr = curr_pts[status == 1]
            m, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
            dx = m[0, 2] if m is not None else 0
            dy = m[1, 2] if m is not None else 0
            da = np.arctan2(m[1, 0], m[0, 0]) if m is not None else 0
            transforms.append((dx, dy, da))
            prev_gray = curr_gray

        cap.release()

        # Smooth trajectory
        trajectory = np.cumsum(transforms, axis=0)
        smoothed = np.copy(trajectory)
        r = self.smoothing_radius
        for i in range(len(trajectory)):
            start = max(0, i - r)
            end = min(len(trajectory), i + r + 1)
            smoothed[i] = np.mean(trajectory[start:end], axis=0)
        smooth_transforms = smoothed - trajectory + np.array(transforms)

        # Pass 2: apply transforms
        cap = cv2.VideoCapture(input_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i, (dx, dy, da) in enumerate(tqdm(smooth_transforms, desc="Stabilize pass 2")):
            ret, frame = cap.read()
            if not ret:
                break
            m = np.array([[np.cos(da), -np.sin(da), dx],
                          [np.sin(da),  np.cos(da), dy]], dtype=np.float32)
            stabilized = cv2.warpAffine(frame, m, (w, h))
            out.write(stabilized)

        cap.release()
        out.release()
        log.info(f"Stabilized video saved: {output_path}")


# ─── MAIN EXTRACTOR ───────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    period_id: str,
    config: dict,
    srt_path: Optional[str] = None,
) -> list[FrameMeta]:
    """
    Main frame extraction pipeline.

    Args:
        video_path:  Path to drone .mp4/.mov file
        output_dir:  Directory to save extracted frames
        period_id:   E.g. "2024_03" (used in filenames)
        config:      Loaded YAML config
        srt_path:    Optional DJI SRT GPS file

    Returns:
        List of FrameMeta objects (also saved as JSON)
    """
    cfg_video = config["video"]
    cfg_pre = config["preprocessing"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Optional stabilization
    working_video = video_path
    if cfg_video.get("stabilization"):
        stable_path = str(Path(output_dir) / "stabilized.mp4")
        if not os.path.exists(stable_path):
            stabilizer = VideoStabilizer()
            stabilizer.stabilize_video(video_path, stable_path)
        working_video = stable_path

    # Parse GPS
    gps_data = {}
    if srt_path:
        gps_data = parse_dji_srt(srt_path)

    cap = cv2.VideoCapture(working_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {working_video}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or cfg_video["fps"]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(actual_fps * cfg_video["frame_extract_interval_sec"])
    resize_w, resize_h = cfg_pre["resize_output"]

    log.info(f"Video: {total_frames} frames @ {actual_fps:.1f}fps → "
             f"extracting every {interval_frames} frames (~{total_frames // interval_frames} frames total)")

    frame_metas: list[FrameMeta] = []
    prev_frame = None
    frame_idx = 0
    saved_count = 0
    blur_skipped = 0
    dup_skipped = 0

    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval_frames == 0:
                timestamp = frame_idx / actual_fps

                # Blur check
                blur = laplacian_blur_score(frame)
                if blur < cfg_pre["blur_threshold"]:
                    blur_skipped += 1
                    frame_idx += 1
                    pbar.update(1)
                    continue

                # Duplicate check
                if frame_too_similar(frame, prev_frame, cfg_pre["duplicate_ssim_threshold"]):
                    dup_skipped += 1
                    frame_idx += 1
                    pbar.update(1)
                    continue

                # Resize and save
                resized = cv2.resize(frame, (resize_w, resize_h))
                fname = f"{period_id}_frame_{saved_count:05d}_t{timestamp:.1f}s.jpg"
                fpath = str(Path(output_dir) / fname)
                cv2.imwrite(fpath, resized, [cv2.IMWRITE_JPEG_QUALITY, 92])

                # GPS lookup
                gps = get_nearest_gps(gps_data, timestamp)

                meta = FrameMeta(
                    frame_idx=frame_idx,
                    timestamp_sec=round(timestamp, 2),
                    filepath=fpath,
                    blur_score=round(blur, 2),
                    lat=gps.get("lat"),
                    lon=gps.get("lon"),
                    altitude_m=gps.get("altitude_m"),
                    heading_deg=gps.get("heading_deg"),
                )
                frame_metas.append(meta)
                prev_frame = frame
                saved_count += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()

    # Save metadata
    meta_path = str(Path(output_dir) / f"{period_id}_frame_metadata.json")
    with open(meta_path, "w") as f:
        json.dump([asdict(m) for m in frame_metas], f, indent=2)

    log.info(f"✓ Extracted {saved_count} frames | "
             f"Skipped: {blur_skipped} blurry, {dup_skipped} duplicates")
    log.info(f"  Metadata saved → {meta_path}")

    return frame_metas


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 1: Video preprocessing & frame extraction")
    parser.add_argument("--video",    required=True, help="Path to drone video file")
    parser.add_argument("--period",   required=True, help="Period ID e.g. 2024_03")
    parser.add_argument("--output",   default="data/processed", help="Output directory")
    parser.add_argument("--srt",      default=None, help="DJI SRT GPS file path (optional)")
    parser.add_argument("--config",   default="config/pipeline_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = os.path.join(args.output, args.period, "frames")
    extract_frames(args.video, out_dir, args.period, cfg, args.srt)
