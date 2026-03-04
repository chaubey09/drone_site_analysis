"""
05_metrics.py
─────────────
Step 5: Progress Metrics & KPIs

Aggregates all change detection data into:
  - Overall % completion estimate per zone
  - Activity score (how much work is happening per zone)
  - Timeline chart data
  - Alert flags (stalled zones, rapid changes)

Output:
  - progress_metrics_{curr}.json
  - timeline_data.json (cumulative across all periods)
"""

import json
import os
import glob
from pathlib import Path
from typing import Optional
import logging
import yaml
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── COMPLETION ESTIMATION ────────────────────────────────────────────────────

# Weights for each class in estimating "construction completeness"
# Higher weight = more indicative of finished work
COMPLETION_WEIGHTS = {
    "concrete_structure": 1.0,   # Finished concrete = high progress
    "steel_rebar":        0.4,   # Rebar = in-progress
    "formwork":           0.3,   # Formwork = in-progress
    "earthwork":          0.1,   # Earthwork = early stage
    "road_surface":       0.0,   # Pre-existing road
    "equipment":          0.0,   # Equipment = activity, not completion
    "material_stockpile": 0.05,  # Materials = preparation
    "water_body":         0.0,
    "vegetation":         0.0,
    "background":         0.0,
    "workers":            0.0,
}

# Maximum plausible construction coverage for a metro site
# (in m² — used to cap percentage estimates; set based on your actual site size)
DEFAULT_MAX_CONSTRUCTION_M2 = 500_000


def estimate_completion_pct(areas: dict, max_construction_m2: float = DEFAULT_MAX_CONSTRUCTION_M2) -> float:
    """
    Estimate % completion of a zone/site based on class areas.
    Returns 0-100.
    """
    weighted_sum = sum(
        COMPLETION_WEIGHTS.get(cls, 0) * data.get("area_m2", 0)
        for cls, data in areas.items()
    )
    return min(100.0, round(100 * weighted_sum / max_construction_m2, 1))


def compute_activity_score(areas: dict) -> float:
    """
    Activity score = normalized indicator of active work.
    High if lots of equipment, workers, earthwork, formwork.
    """
    activity_classes = {
        "equipment": 3.0,
        "workers": 2.0,
        "formwork": 1.5,
        "steel_rebar": 1.0,
        "earthwork": 1.0,
        "material_stockpile": 0.5,
    }
    raw = sum(
        weight * areas.get(cls, {}).get("area_m2", 0)
        for cls, weight in activity_classes.items()
    )
    # Normalize to 0-100 scale (assume 10,000 m² of active area = score 100)
    return min(100.0, round(raw / 100, 1))


# ─── PERIOD METRICS ───────────────────────────────────────────────────────────

def compute_period_metrics(curr_period: str, prev_period: Optional[str],
                            processed_dir: str, config: dict) -> dict:
    """
    Compute full metrics for a period.
    """
    zones = config.get("zones", [])
    max_m2 = config.get("metrics", {}).get("max_construction_m2", DEFAULT_MAX_CONSTRUCTION_M2)

    # Load current segmentation stats
    seg_path = os.path.join(processed_dir, curr_period, f"class_areas_{curr_period}.json")
    with open(seg_path) as f:
        curr_seg = json.load(f)

    curr_global = curr_seg.get("global_areas", {})
    curr_zones  = curr_seg.get("zone_areas", {})

    # Load change stats if previous period exists
    change_stats = None
    zone_deltas = []
    if prev_period:
        change_dir = os.path.join(processed_dir, f"changes_{prev_period}_vs_{curr_period}")
        change_path = os.path.join(change_dir, f"change_stats_{prev_period}_{curr_period}.json")
        if os.path.exists(change_path):
            with open(change_path) as f:
                change_stats = json.load(f)
            zone_deltas = change_stats.get("zone_deltas", [])

    # Global metrics
    global_completion_pct = estimate_completion_pct(curr_global, max_m2)
    global_activity_score = compute_activity_score(curr_global)

    # Zone-level metrics
    zone_metrics = []
    zone_delta_map = {zd["zone_id"]: zd for zd in zone_deltas}

    for zone in zones:
        zid = zone["id"]
        zone_areas = curr_zones.get(zid, {}).get("areas", {})

        completion = estimate_completion_pct(zone_areas, max_m2 / max(len(zones), 1))
        activity   = compute_activity_score(zone_areas)
        delta      = zone_delta_map.get(zid, {})

        # Determine alert status
        alert = None
        if prev_period:
            if delta.get("status") == "stalled" and completion < 95:
                alert = "⚠️ Zone stalled — no significant progress"
            elif delta.get("status") == "regressed":
                alert = "🔴 Regression detected — construction reduced"
            elif delta.get("percent_change", 0) > 50:
                alert = "🚀 Rapid progress this period"

        zone_metrics.append({
            "zone_id":          zid,
            "zone_name":        zone["name"],
            "completion_pct":   completion,
            "activity_score":   activity,
            "delta_m2":         delta.get("delta_m2", 0),
            "percent_change":   delta.get("percent_change", 0),
            "status":           delta.get("status", "baseline"),
            "alert":            alert,
            "key_areas_m2": {
                "concrete": round(zone_areas.get("concrete_structure", {}).get("area_m2", 0), 1),
                "earthwork": round(zone_areas.get("earthwork", {}).get("area_m2", 0), 1),
                "formwork":  round(zone_areas.get("formwork", {}).get("area_m2", 0), 1),
                "equipment": round(zone_areas.get("equipment", {}).get("area_m2", 0), 1),
            }
        })

    # Change summary (from change detection output)
    change_summary = {}
    if change_stats:
        cs = change_stats.get("change_stats", {})
        change_summary = {
            "new_construction_m2":   round(cs.get("new_construction", {}).get("pixels", 0) *
                                           (config["georeferencing"]["output_resolution_cm_per_px"]/100)**2, 1),
            "demolition_m2":         round(cs.get("demolition_removal", {}).get("pixels", 0) *
                                           (config["georeferencing"]["output_resolution_cm_per_px"]/100)**2, 1),
            "earthwork_progress_m2": round(cs.get("earthwork_progress", {}).get("pixels", 0) *
                                           (config["georeferencing"]["output_resolution_cm_per_px"]/100)**2, 1),
            "equipment_movements":   cs.get("equipment_movement", {}).get("pixels", 0),
        }

    metrics = {
        "period_id":            curr_period,
        "prev_period_id":       prev_period,
        "computed_at":          datetime.now().isoformat(),
        "global": {
            "completion_pct":   global_completion_pct,
            "activity_score":   global_activity_score,
            "total_site_area_m2": round(sum(
                d.get("area_m2", 0) for d in curr_global.values()), 1),
            "construction_area_m2": round(
                curr_global.get("concrete_structure", {}).get("area_m2", 0) +
                curr_global.get("steel_rebar", {}).get("area_m2", 0) +
                curr_global.get("formwork", {}).get("area_m2", 0), 1),
        },
        "change_summary":       change_summary,
        "zones":                zone_metrics,
        "alerts":               [zm["alert"] for zm in zone_metrics if zm["alert"]],
    }

    return metrics


# ─── TIMELINE AGGREGATOR ──────────────────────────────────────────────────────

def update_timeline(curr_metrics: dict, processed_dir: str) -> dict:
    """
    Update rolling timeline JSON with new period's metrics.
    timeline_data.json tracks completion % and activity over all periods.
    """
    timeline_path = os.path.join(processed_dir, "timeline_data.json")

    if os.path.exists(timeline_path):
        with open(timeline_path) as f:
            timeline = json.load(f)
    else:
        timeline = {"periods": [], "zones": {}}

    period_id = curr_metrics["period_id"]

    # Global entry
    period_entry = {
        "period_id":        period_id,
        "completion_pct":   curr_metrics["global"]["completion_pct"],
        "activity_score":   curr_metrics["global"]["activity_score"],
        "construction_m2":  curr_metrics["global"]["construction_area_m2"],
        "new_construction_m2": curr_metrics["change_summary"].get("new_construction_m2", 0),
    }

    # Replace or append
    existing = [p for p in timeline["periods"] if p["period_id"] != period_id]
    timeline["periods"] = existing + [period_entry]
    timeline["periods"].sort(key=lambda x: x["period_id"])

    # Zone entries
    for zm in curr_metrics["zones"]:
        zid = zm["zone_id"]
        if zid not in timeline["zones"]:
            timeline["zones"][zid] = {"name": zm["zone_name"], "history": []}

        zone_entry = {
            "period_id":      period_id,
            "completion_pct": zm["completion_pct"],
            "activity_score": zm["activity_score"],
            "delta_m2":       zm["delta_m2"],
        }
        existing_z = [h for h in timeline["zones"][zid]["history"]
                      if h["period_id"] != period_id]
        timeline["zones"][zid]["history"] = existing_z + [zone_entry]
        timeline["zones"][zid]["history"].sort(key=lambda x: x["period_id"])

    with open(timeline_path, "w") as f:
        json.dump(timeline, f, indent=2)

    log.info(f"Timeline updated: {len(timeline['periods'])} periods tracked")
    return timeline


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def compute_metrics(curr_period: str, prev_period: Optional[str],
                    processed_dir: str, config: dict) -> dict:
    log.info(f"Computing metrics for {curr_period} (vs {prev_period})")

    metrics = compute_period_metrics(curr_period, prev_period, processed_dir, config)

    # Save
    out_path = os.path.join(processed_dir, curr_period, f"progress_metrics_{curr_period}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Update timeline
    timeline = update_timeline(metrics, processed_dir)

    log.info(f"✓ Metrics computed for {curr_period}")
    log.info(f"  Overall completion:  {metrics['global']['completion_pct']}%")
    log.info(f"  Activity score:      {metrics['global']['activity_score']}")
    if metrics["alerts"]:
        log.warning(f"  ALERTS: {len(metrics['alerts'])}")
        for alert in metrics["alerts"]:
            log.warning(f"    {alert}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 5: Progress metrics")
    parser.add_argument("--curr",      required=True)
    parser.add_argument("--prev",      default=None)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--config",    default="config/pipeline_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    compute_metrics(args.curr, args.prev, args.processed, cfg)
