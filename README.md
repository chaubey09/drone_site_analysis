# Metro Construction Progress Tracking Pipeline

## Overview
Bi-monthly drone video analysis pipeline for metro line construction progress tracking.
Compares current period footage against previous period to generate automated progress reports.

## Pipeline Architecture

```
Raw Drone Video (30min @ 30fps)
        │
        ▼
┌─────────────────────┐
│  1. Preprocessing   │  Frame extraction, stabilization, GPS sync
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  2. Georeferencing  │  Stitch frames → orthomosaic, align to map
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  3. Segmentation    │  Detect: structures, equipment, materials, bare ground
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  4. Change Detection│  Diff current vs previous period
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  5. Progress Metrics│  % completion per zone, activity hotspots
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  6. Report Generator│  PDF/HTML report with maps, charts, comparisons
└─────────────────────┘
```

## Folder Structure
```
metro_pipeline/
├── config/
│   └── pipeline_config.yaml
├── data/
│   ├── raw/               # Raw drone videos per period
│   │   ├── 2024_01/
│   │   ├── 2024_03/
│   │   └── ...
│   ├── processed/         # Extracted frames, orthomosaics
│   └── reports/           # Generated progress reports
├── src/
│   ├── 01_preprocess.py
│   ├── 02_georeference.py
│   ├── 03_segment.py
│   ├── 04_change_detection.py
│   ├── 05_metrics.py
│   ├── 06_report_generator.py
│   └── pipeline_runner.py
├── models/
│   └── segmentation_model/
├── requirements.txt
└── README.md
```



python pipeline_runner.py --curr 2024_01 --video drone_video.mp4