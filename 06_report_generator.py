"""
06_report_generator.py
──────────────────────
Step 6: Progress Report Generator

Generates professional bi-monthly construction progress report as:
  - PDF (via ReportLab)
  - HTML (standalone, with embedded charts)

Report sections:
  1. Executive Summary
  2. Orthomosaic Comparison (before/after)
  3. Zone-by-Zone Progress Table
  4. Change Detection Map
  5. Activity Heatmap
  6. Timeline Chart (completion % over all periods)
  7. Alerts & Observations
  8. Appendix: Sample Frames
"""

import json
import os
import base64
from pathlib import Path
from typing import Optional
import logging
import yaml
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── IMAGE UTILITIES ──────────────────────────────────────────────────────────

def img_to_b64(path: str, max_w: int = 1200) -> Optional[str]:
    """Load image, optionally resize, return base64 string."""
    try:
        import cv2
        import numpy as np
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > max_w:
            img = cv2.resize(img, (max_w, int(h * max_w / w)))
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode()
    except Exception as e:
        log.warning(f"Could not encode image {path}: {e}")
        return None


# ─── HTML REPORT ──────────────────────────────────────────────────────────────

def generate_html_report(metrics: dict, change_stats: dict,
                          timeline: dict, image_paths: dict,
                          config: dict, output_path: str):
    """
    Generate standalone HTML report with embedded images and charts.
    """
    proj = config["report"].get("project_name", "Metro Construction")
    curr_id = metrics["period_id"]
    prev_id = metrics.get("prev_period_id", "N/A")
    report_date = datetime.now().strftime("%d %B %Y")

    # Encode images to base64
    def img_tag(key, alt="", width="100%"):
        path = image_paths.get(key)
        if path and os.path.exists(path):
            b64 = img_to_b64(path)
            if b64:
                return f'<img src="data:image/jpeg;base64,{b64}" alt="{alt}" style="width:{width};border-radius:8px;margin:8px 0;">'
        return f'<div style="background:#2a2a2a;padding:40px;text-align:center;color:#666;border-radius:8px;">[Image not available: {key}]</div>'

    # Zone table rows
    zone_rows = ""
    for zm in metrics["zones"]:
        status_color = {"progressing": "#22c55e", "stalled": "#f59e0b",
                        "regressed": "#ef4444", "baseline": "#6b7280"}.get(zm["status"], "#6b7280")
        alert_html = f'<span style="font-size:0.85em">{zm["alert"]}</span>' if zm["alert"] else "—"
        zone_rows += f"""
        <tr>
          <td>{zm['zone_id']}</td>
          <td>{zm['zone_name']}</td>
          <td>
            <div style="background:#1e293b;border-radius:4px;height:18px;position:relative;overflow:hidden;">
              <div style="background:linear-gradient(90deg,#3b82f6,#22c55e);width:{zm['completion_pct']}%;height:100%;border-radius:4px;"></div>
              <span style="position:absolute;left:50%;transform:translateX(-50%);top:1px;font-size:0.75em;color:#fff;">{zm['completion_pct']}%</span>
            </div>
          </td>
          <td style="color:{status_color};font-weight:600">{zm['status'].upper()}</td>
          <td style="color:{'#22c55e' if zm['delta_m2']>=0 else '#ef4444'}">{zm['delta_m2']:+,.0f} m²</td>
          <td>{zm['key_areas_m2']['concrete']:,.0f}</td>
          <td>{zm['key_areas_m2']['earthwork']:,.0f}</td>
          <td>{alert_html}</td>
        </tr>"""

    # Timeline chart data (Chart.js)
    periods = [p["period_id"] for p in timeline["periods"]]
    completion_data = [p["completion_pct"] for p in timeline["periods"]]
    activity_data   = [p["activity_score"] for p in timeline["periods"]]
    new_const_data  = [p.get("new_construction_m2", 0) for p in timeline["periods"]]

    # Zone timeline datasets
    zone_datasets = ""
    colors = ["#3b82f6","#22c55e","#f59e0b","#ef4444","#8b5cf6","#ec4899","#06b6d4"]
    for i, (zid, zdata) in enumerate(timeline["zones"].items()):
        zname = zdata["name"]
        z_periods = {h["period_id"]: h["completion_pct"] for h in zdata["history"]}
        z_vals = [z_periods.get(p, "null") for p in periods]
        color = colors[i % len(colors)]
        zone_datasets += f"""{{
          label: '{zname}',
          data: {z_vals},
          borderColor: '{color}',
          backgroundColor: '{color}22',
          tension: 0.4,
          pointRadius: 5,
        }},"""

    # Global completion/activity % change
    g = metrics["global"]
    cs = metrics["change_summary"]

    alerts_html = ""
    for alert in metrics.get("alerts", []):
        alert_color = "#ef4444" if "🔴" in alert else "#f59e0b" if "⚠️" in alert else "#22c55e"
        alerts_html += f'<div style="background:{alert_color}22;border-left:4px solid {alert_color};padding:12px 16px;margin:8px 0;border-radius:0 8px 8px 0;color:{alert_color};">{alert}</div>'

    if not alerts_html:
        alerts_html = '<p style="color:#6b7280;">No alerts for this period. All zones progressing normally.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{proj} — Progress Report {curr_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --surface2: #0f172a;
    --border: #334155; --text: #e2e8f0; --muted: #94a3b8;
    --accent: #3b82f6; --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif;
         line-height: 1.6; }}
  .header {{ background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 60%);
             border-bottom: 2px solid var(--accent); padding: 40px 48px; }}
  .header h1 {{ font-size: 2rem; font-weight: 700; color: #fff; margin-bottom: 6px; }}
  .header .subtitle {{ color: var(--muted); font-size: 1rem; }}
  .header .meta {{ margin-top: 16px; display: flex; gap: 32px; flex-wrap: wrap; }}
  .header .meta-item {{ display: flex; flex-direction: column; }}
  .header .meta-item .label {{ font-size: 0.7rem; text-transform: uppercase;
                                letter-spacing: 0.1em; color: var(--muted); }}
  .header .meta-item .value {{ font-size: 1rem; font-weight: 600; color: var(--accent); }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 40px 32px; }}
  .section {{ margin-bottom: 48px; }}
  .section-title {{ font-size: 1.3rem; font-weight: 700; color: var(--accent);
                    border-bottom: 1px solid var(--border); padding-bottom: 10px;
                    margin-bottom: 24px; text-transform: uppercase;
                    letter-spacing: 0.05em; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr));
               gap: 16px; margin-bottom: 24px; }}
  .kpi-card {{ background: var(--surface); border: 1px solid var(--border);
               border-radius: 12px; padding: 20px; }}
  .kpi-card .kpi-label {{ font-size: 0.75rem; text-transform: uppercase;
                           letter-spacing: 0.1em; color: var(--muted); margin-bottom: 8px; }}
  .kpi-card .kpi-value {{ font-size: 2rem; font-weight: 800; line-height: 1; }}
  .kpi-card .kpi-sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 6px; }}
  .img-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .img-grid.triple {{ grid-template-columns: 1fr 1fr 1fr; }}
  .img-label {{ font-size: 0.8rem; color: var(--muted); text-align: center;
                margin-top: 6px; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
  th {{ background: var(--surface); color: var(--muted); text-transform: uppercase;
        letter-spacing: 0.05em; font-size: 0.7rem; padding: 12px 14px; text-align: left;
        border-bottom: 2px solid var(--border); }}
  td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: #ffffff08; }}
  .chart-container {{ background: var(--surface); border: 1px solid var(--border);
                       border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
  .chart-title {{ font-size: 0.9rem; font-weight: 600; color: var(--muted);
                  margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  canvas {{ max-height: 320px; }}
  @media (max-width: 768px) {{
    .img-grid, .two-col {{ grid-template-columns: 1fr; }}
    .img-grid.triple {{ grid-template-columns: 1fr; }}
    .container {{ padding: 20px 16px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>🚇 {proj}</h1>
  <div class="subtitle">Bi-Monthly Construction Progress Report</div>
  <div class="meta">
    <div class="meta-item"><span class="label">Report Period</span><span class="value">{curr_id}</span></div>
    <div class="meta-item"><span class="label">Compared With</span><span class="value">{prev_id}</span></div>
    <div class="meta-item"><span class="label">Report Date</span><span class="value">{report_date}</span></div>
    <div class="meta-item"><span class="label">Overall Completion</span><span class="value">{g['completion_pct']}%</span></div>
  </div>
</div>

<div class="container">

<!-- 1. EXECUTIVE SUMMARY -->
<div class="section">
  <div class="section-title">1. Executive Summary</div>
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Overall Completion</div>
      <div class="kpi-value" style="color:var(--accent)">{g['completion_pct']}%</div>
      <div class="kpi-sub">Weighted by construction class</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Activity Score</div>
      <div class="kpi-value" style="color:var(--green)">{g['activity_score']}</div>
      <div class="kpi-sub">Equipment + workers + earthwork proxy</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">New Construction</div>
      <div class="kpi-value" style="color:var(--green)">{cs.get('new_construction_m2',0):,.0f} m²</div>
      <div class="kpi-sub">vs previous period</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Earthwork Progress</div>
      <div class="kpi-value" style="color:var(--yellow)">{cs.get('earthwork_progress_m2',0):,.0f} m²</div>
      <div class="kpi-sub">Excavation / grading</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Construction Area</div>
      <div class="kpi-value" style="color:#a78bfa">{g['construction_area_m2']:,.0f} m²</div>
      <div class="kpi-sub">Concrete + rebar + formwork total</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Active Alerts</div>
      <div class="kpi-value" style="color:{'var(--red)' if metrics['alerts'] else 'var(--green)'}">{len(metrics['alerts'])}</div>
      <div class="kpi-sub">Stalled / regressed zones</div>
    </div>
  </div>
</div>

<!-- 2. ORTHOMOSAIC COMPARISON -->
<div class="section">
  <div class="section-title">2. Orthomosaic Comparison</div>
  <div class="img-grid">
    <div>
      {img_tag('prev_ortho', f'Previous: {prev_id}')}
      <div class="img-label">Previous Period: {prev_id}</div>
    </div>
    <div>
      {img_tag('curr_ortho', f'Current: {curr_id}')}
      <div class="img-label">Current Period: {curr_id}</div>
    </div>
  </div>
  {img_tag('comparison', 'Side-by-side comparison')}
  <div class="img-label">Left: {prev_id} | Center: {curr_id} | Right: Changes</div>
</div>

<!-- 3. SEGMENTATION OVERLAYS -->
<div class="section">
  <div class="section-title">3. Segmentation Analysis</div>
  <div class="img-grid">
    <div>
      {img_tag('prev_seg_overlay', f'Segmentation {prev_id}')}
      <div class="img-label">Segmentation Overlay — {prev_id}</div>
    </div>
    <div>
      {img_tag('curr_seg_overlay', f'Segmentation {curr_id}')}
      <div class="img-label">Segmentation Overlay — {curr_id}</div>
    </div>
  </div>
</div>

<!-- 4. CHANGE DETECTION -->
<div class="section">
  <div class="section-title">4. Change Detection</div>
  <div class="img-grid">
    <div>
      {img_tag('change_map', 'Change Map')}
      <div class="img-label">Change Map (Green=New Construction, Red=Removal, Orange=Equipment)</div>
    </div>
    <div>
      {img_tag('change_heatmap', 'Activity Heatmap')}
      <div class="img-label">Activity Density Heatmap</div>
    </div>
  </div>
</div>

<!-- 5. ZONE PROGRESS TABLE -->
<div class="section">
  <div class="section-title">5. Zone-by-Zone Progress</div>
  <table>
    <thead>
      <tr>
        <th>Zone</th><th>Name</th><th>Completion</th><th>Status</th>
        <th>Δ Construction</th><th>Concrete (m²)</th><th>Earthwork (m²)</th><th>Alert</th>
      </tr>
    </thead>
    <tbody>{zone_rows}</tbody>
  </table>
</div>

<!-- 6. TIMELINE CHART -->
<div class="section">
  <div class="section-title">6. Progress Timeline</div>
  <div class="two-col">
    <div class="chart-container">
      <div class="chart-title">Overall Completion & Activity Over Time</div>
      <canvas id="globalChart"></canvas>
    </div>
    <div class="chart-container">
      <div class="chart-title">New Construction Added Per Period (m²)</div>
      <canvas id="newConstChart"></canvas>
    </div>
  </div>
  <div class="chart-container">
    <div class="chart-title">Per-Zone Completion History</div>
    <canvas id="zoneChart"></canvas>
  </div>
</div>

<!-- 7. ALERTS -->
<div class="section">
  <div class="section-title">7. Alerts & Observations</div>
  {alerts_html}
</div>

</div>

<script>
const chartDefaults = {{
  plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
    y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
  }}
}};

// Global chart
new Chart(document.getElementById('globalChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(periods)},
    datasets: [
      {{ label: 'Completion %', data: {completion_data},
         borderColor: '#3b82f6', backgroundColor: '#3b82f622',
         tension: 0.4, fill: true, yAxisID: 'y' }},
      {{ label: 'Activity Score', data: {activity_data},
         borderColor: '#22c55e', backgroundColor: '#22c55e22',
         tension: 0.4, fill: true, yAxisID: 'y' }},
    ]
  }},
  options: {{ ...chartDefaults, responsive: true,
    scales: {{ ...chartDefaults.scales,
      y: {{ ...chartDefaults.scales.y, min: 0, max: 100, title: {{ display:true, text:'Score / %', color:'#94a3b8' }} }} }}
  }}
}});

// New construction bar
new Chart(document.getElementById('newConstChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(periods)},
    datasets: [{{ label: 'New Construction m²', data: {new_const_data},
                  backgroundColor: '#22c55e88', borderColor: '#22c55e', borderWidth: 1 }}]
  }},
  options: {{ ...chartDefaults, responsive: true }}
}});

// Zone chart
new Chart(document.getElementById('zoneChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(periods)},
    datasets: [{zone_datasets}]
  }},
  options: {{ ...chartDefaults, responsive: true,
    scales: {{ ...chartDefaults.scales,
      y: {{ ...chartDefaults.scales.y, min: 0, max: 100,
             title: {{ display: true, text: 'Completion %', color: '#94a3b8' }} }} }}
  }}
}});
</script>

</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"HTML report saved: {output_path}")


# ─── PDF REPORT ───────────────────────────────────────────────────────────────

def generate_pdf_report(metrics: dict, change_stats: dict,
                         image_paths: dict, config: dict, output_path: str):
    """
    Generate PDF report using ReportLab.
    Falls back with a message if ReportLab is not installed.
    """
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image as RLImage,
                                         PageBreak, HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        log.warning("reportlab not installed. PDF report skipped. "
                    "Install: pip install reportlab")
        return

    proj  = config["report"].get("project_name", "Metro Construction")
    curr  = metrics["period_id"]
    prev  = metrics.get("prev_period_id", "N/A")
    g     = metrics["global"]
    cs    = metrics["change_summary"]
    zones = metrics["zones"]

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                             topMargin=1.5*cm, bottomMargin=1.5*cm,
                             leftMargin=2*cm, rightMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style  = ParagraphStyle("Title2", fontSize=22, fontName="Helvetica-Bold",
                                   spaceAfter=6, textColor=colors.HexColor("#1d4ed8"))
    head_style   = ParagraphStyle("Head", fontSize=13, fontName="Helvetica-Bold",
                                   spaceAfter=8, spaceBefore=16,
                                   textColor=colors.HexColor("#1d4ed8"),
                                   borderPad=4)
    normal_style = styles["Normal"]
    normal_style.fontSize = 10

    story = []

    # Cover
    story.append(Paragraph(f"🚇 {proj}", title_style))
    story.append(Paragraph(f"Bi-Monthly Construction Progress Report", styles["Normal"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"Report Period: <b>{curr}</b> | Compared With: <b>{prev}</b> | "
                             f"Generated: <b>{datetime.now().strftime('%d %b %Y')}</b>",
                             normal_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1d4ed8"),
                              spaceAfter=12))

    # KPI table
    story.append(Paragraph("Executive Summary", head_style))
    kpi_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Overall Completion", f"{g['completion_pct']}%",
         "Activity Score",    f"{g['activity_score']}"],
        ["New Construction",  f"{cs.get('new_construction_m2',0):,.0f} m²",
         "Earthwork Progress",f"{cs.get('earthwork_progress_m2',0):,.0f} m²"],
        ["Construction Area", f"{g['construction_area_m2']:,.0f} m²",
         "Active Alerts",     str(len(metrics["alerts"]))],
    ]
    ts = TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",    (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",    (2,1), (2,-1), "Helvetica-Bold"),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#334155")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("ALIGN",       (1,0), (1,-1), "RIGHT"),
        ("ALIGN",       (3,0), (3,-1), "RIGHT"),
    ])
    t = Table(kpi_data, colWidths=[5.5*cm, 3.5*cm, 5.5*cm, 3.5*cm])
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Images
    def add_image(key, caption, max_w=16*cm):
        path = image_paths.get(key)
        if path and os.path.exists(path):
            try:
                import cv2
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    ratio = min(max_w / w, (10*cm) / h)
                    rw, rh = w * ratio, h * ratio
                    story.append(RLImage(path, width=rw, height=rh))
                    story.append(Paragraph(f"<i>{caption}</i>", normal_style))
                    story.append(Spacer(1, 0.3*cm))
            except Exception as e:
                log.warning(f"Could not add image {path}: {e}")

    story.append(Paragraph("Orthomosaic Comparison", head_style))
    add_image("comparison", f"Left: {prev} | Center: {curr} | Right: Changes")

    story.append(Paragraph("Change Detection Map", head_style))
    add_image("change_map", "Change Map — Green: New Construction | Red: Removal | Orange: Equipment")
    add_image("change_heatmap", "Activity Density Heatmap")

    # Zone table
    story.append(PageBreak())
    story.append(Paragraph("Zone-by-Zone Progress", head_style))
    zone_data = [["Zone", "Name", "Completion", "Status", "Δ m²", "Alert"]]
    for zm in zones:
        zone_data.append([
            zm["zone_id"], zm["zone_name"][:28],
            f"{zm['completion_pct']}%",
            zm["status"].upper(),
            f"{zm['delta_m2']:+,.0f}",
            (zm["alert"] or "—")[:40],
        ])
    zt = Table(zone_data, colWidths=[1.5*cm, 5*cm, 2.5*cm, 2.5*cm, 2*cm, 5*cm])
    zt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ALIGN",      (2,0), (4,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(zt)

    # Alerts
    if metrics["alerts"]:
        story.append(Paragraph("Alerts & Observations", head_style))
        for alert in metrics["alerts"]:
            story.append(Paragraph(f"• {alert}", normal_style))

    doc.build(story)
    log.info(f"PDF report saved: {output_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def generate_report(curr_period: str, prev_period: Optional[str],
                    processed_dir: str, config: dict) -> dict:
    """Generate HTML and PDF reports for a period comparison."""
    reports_dir = os.path.join("data", "reports", curr_period)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    metrics_path = os.path.join(processed_dir, curr_period,
                                 f"progress_metrics_{curr_period}.json")
    with open(metrics_path) as f:
        metrics = json.load(f)

    timeline_path = os.path.join(processed_dir, "timeline_data.json")
    timeline = {"periods": [], "zones": {}}
    if os.path.exists(timeline_path):
        with open(timeline_path) as f:
            timeline = json.load(f)

    change_stats = {}
    if prev_period:
        cs_path = os.path.join(processed_dir,
                                f"changes_{prev_period}_vs_{curr_period}",
                                f"change_stats_{prev_period}_{curr_period}.json")
        if os.path.exists(cs_path):
            with open(cs_path) as f:
                change_stats = json.load(f)

    # Collect image paths
    def p(period, filename):
        return os.path.join(processed_dir, period, filename)
    def chp(filename):
        if prev_period:
            return os.path.join(processed_dir,
                                 f"changes_{prev_period}_vs_{curr_period}", filename)
        return ""

    image_paths = {
        "prev_ortho":        p(prev_period, f"orthomosaic_{prev_period}.png") if prev_period else "",
        "curr_ortho":        p(curr_period, f"orthomosaic_{curr_period}.png"),
        "prev_seg_overlay":  p(prev_period, f"segmentation_overlay_{prev_period}.jpg") if prev_period else "",
        "curr_seg_overlay":  p(curr_period, f"segmentation_overlay_{curr_period}.jpg"),
        "change_map":        chp(f"change_map_{prev_period}_{curr_period}.png") if prev_period else "",
        "change_heatmap":    chp(f"change_heatmap_{prev_period}_{curr_period}.png") if prev_period else "",
        "comparison":        chp(f"comparison_{prev_period}_{curr_period}.jpg") if prev_period else "",
    }

    html_path = os.path.join(reports_dir, f"progress_report_{curr_period}.html")
    pdf_path  = os.path.join(reports_dir, f"progress_report_{curr_period}.pdf")

    generate_html_report(metrics, change_stats, timeline, image_paths, config, html_path)
    generate_pdf_report(metrics, change_stats, image_paths, config, pdf_path)

    log.info(f"✓ Reports generated in: {reports_dir}")
    return {"html": html_path, "pdf": pdf_path}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 6: Report generation")
    parser.add_argument("--curr",      required=True)
    parser.add_argument("--prev",      default=None)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--config",    default="config/pipeline_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    generate_report(args.curr, args.prev, args.processed, cfg)
