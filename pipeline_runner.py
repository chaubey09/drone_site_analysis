"""
pipeline_runner.py
Master Pipeline Runner - Windows compatible

Usage:
  python pipeline_runner.py --curr 2024_01 --video drone_video.mp4
  python pipeline_runner.py --curr 2024_03 --prev 2024_01 --video drone_video.mp4
  python pipeline_runner.py --batch
"""

import argparse
import os
import sys
import io
import yaml
import logging
import importlib.util
from pathlib import Path
from datetime import datetime

# ── Fix Windows console Unicode crash (box-drawing chars like ─ etc.) ─────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Make sure the folder containing pipeline_runner.py is on sys.path ─────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ── Logging: UTF-8 file + console ─────────────────────────────────────────────
log_handlers = [
    logging.FileHandler("pipeline.log", encoding="utf-8"),
    logging.StreamHandler(sys.stdout),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=log_handlers,
)
log = logging.getLogger("pipeline_runner")


# ── Helper: load a sibling .py file as a module ───────────────────────────────

def load_module(filename: str):
    """Load one of the pipeline step files by filename (e.g. '01_preprocess.py')."""
    path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pipeline module not found: {path}\n"
            f"Make sure all pipeline .py files are in the same folder as pipeline_runner.py"
        )
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected at: {os.path.abspath(config_path)}"
        )
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Main period runner ────────────────────────────────────────────────────────

def run_period(curr_period: str, prev_period,
               video_path, srt_path,
               config: dict, args) -> bool:

    processed_dir = "data/processed"
    log.info("=" * 60)
    log.info(f"PERIOD: {curr_period}  |  prev: {prev_period or 'N/A'}")
    log.info("=" * 60)
    start_time = datetime.now()

    # STEP 1 - Preprocess
    if not args.skip_preprocess:
        if not video_path:
            log.error("--video is required. Add --skip-preprocess to skip this step.")
            return False
        if not os.path.exists(video_path):
            log.error(f"Video file not found: {os.path.abspath(video_path)}")
            return False
        log.info("STEP 1: Preprocessing & Frame Extraction")
        mod = load_module("01_preprocess.py")
        out_dir = os.path.join(processed_dir, curr_period, "frames")
        mod.extract_frames(video_path, out_dir, curr_period, config, srt_path)
    else:
        log.info("STEP 1: SKIPPED")

    # STEP 2 - Georeference
    if not args.skip_georeference:
        log.info("STEP 2: Georeferencing & Orthomosaic")
        mod = load_module("02_georeference.py")
        mod.georeference(curr_period, processed_dir, config)
    else:
        log.info("STEP 2: SKIPPED")

    # STEP 3 - Segmentation
    if not args.skip_segment:
        log.info("STEP 3: Semantic Segmentation")
        mod = load_module("03_segment.py")
        mod.segment_period(curr_period, processed_dir, config)
    else:
        log.info("STEP 3: SKIPPED")

    # STEP 4 - Change Detection
    if prev_period and not args.skip_changes:
        log.info("STEP 4: Change Detection")
        mod = load_module("04_change_detection.py")
        mod.detect_changes(prev_period, curr_period, processed_dir, config)
    else:
        reason = "no previous period" if not prev_period else "--skip-changes"
        log.info(f"STEP 4: SKIPPED ({reason})")

    # STEP 5 - Metrics
    if not args.skip_metrics:
        log.info("STEP 5: Progress Metrics")
        mod = load_module("05_metrics.py")
        mod.compute_metrics(curr_period, prev_period, processed_dir, config)
    else:
        log.info("STEP 5: SKIPPED")

    # STEP 6 - Report
    if not args.skip_report:
        log.info("STEP 6: Report Generation")
        mod = load_module("06_report_generator.py")
        report_paths = mod.generate_report(curr_period, prev_period, processed_dir, config)
        log.info(f"Reports saved: {report_paths}")
    else:
        log.info("STEP 6: SKIPPED")

    elapsed = (datetime.now() - start_time).total_seconds()
    log.info(f"Period {curr_period} complete in {elapsed:.0f}s")
    return True


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(config: dict, args):
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        log.error("data/raw directory not found. Create it and place period folders inside.")
        return

    periods = sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ])
    if not periods:
        log.error(f"No period sub-folders found in {raw_dir}")
        return

    log.info(f"Batch mode: {len(periods)} periods: {periods}")
    prev = None
    for curr in periods:
        period_dir = os.path.join(raw_dir, curr)
        video_path = None
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            candidates = list(Path(period_dir).glob(f"*{ext}"))
            if candidates:
                video_path = str(candidates[0])
                break

        srt_path = None
        srt_candidates = list(Path(period_dir).glob("*.srt"))
        if srt_candidates:
            srt_path = str(srt_candidates[0])

        success = run_period(curr, prev, video_path, srt_path, config, args)
        if success:
            prev = curr
        else:
            log.error(f"Period {curr} failed. Stopping batch.")
            break


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Metro Construction Progress Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  First period:
    python pipeline_runner.py --curr 2024_01 --video drone_video.mp4

  Second period with comparison:
    python pipeline_runner.py --curr 2024_03 --prev 2024_01 --video drone_video.mp4

  With DJI SRT GPS file:
    python pipeline_runner.py --curr 2024_03 --prev 2024_01 --video DJI_0001.mp4 --srt DJI_0001.srt

  Batch (auto-finds videos in data/raw/<period>/ folders):
    python pipeline_runner.py --batch

  Redo only the report:
    python pipeline_runner.py --curr 2024_03 --prev 2024_01
      --skip-preprocess --skip-georeference --skip-segment --skip-changes --skip-metrics
        """
    )
    parser.add_argument("--curr",   help="Current period ID e.g. 2024_03")
    parser.add_argument("--prev",   default=None, help="Previous period ID")
    parser.add_argument("--video",  default=None, help="Path to drone video (.mp4/.mov)")
    parser.add_argument("--srt",    default=None, help="DJI SRT GPS file (optional)")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--batch",  action="store_true",
                        help="Process all periods in data/raw/")
    parser.add_argument("--skip-preprocess",   action="store_true", dest="skip_preprocess")
    parser.add_argument("--skip-georeference", action="store_true", dest="skip_georeference")
    parser.add_argument("--skip-segment",      action="store_true", dest="skip_segment")
    parser.add_argument("--skip-changes",      action="store_true", dest="skip_changes")
    parser.add_argument("--skip-metrics",      action="store_true", dest="skip_metrics")
    parser.add_argument("--skip-report",       action="store_true", dest="skip_report")

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    if args.batch:
        run_batch(config, args)
    elif args.curr:
        run_period(args.curr, args.prev, args.video, args.srt, config, args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()