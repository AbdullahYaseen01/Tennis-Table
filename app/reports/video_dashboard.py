from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj

def generate_video_dashboard(
    out_path: Path,
    *,
    video_name: str,
    ball_type: str,
    metrics: dict[str, Any],
    frames: list[dict[str, Any]],
    bounce_frames: list[int],
    fps: float,
) -> Path:
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _json_safe({
        "video_name": video_name,
        "ball_type": ball_type.replace("_", " ").title(),
        "fps": round(float(fps), 3),
        "metrics": metrics,
        "bounce_frames": bounce_frames,
        "frames": frames,
    })
    data_json = json.dumps(payload, separators=(",", ":"))

    html = _HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    out_path.write_text(html, encoding="utf-8")
    return out_path

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Compression Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0b0f14;
      --card: #1a2332;
      --lime: #c8f135;
      --teal: #00d4aa;
      --coral: #ff6b4a;
      --sky: #4dabf7;
      --gold: #fbbf24;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --border: rgba(255,255,255,0.08);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "DM Sans", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(200,241,53,0.10), transparent),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(0,212,170,0.06), transparent);
    }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 1.75rem 1.25rem 3rem; }
    header { margin-bottom: 1.5rem; }
    h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }
    .sub { color: var(--muted); margin-top: 0.35rem; font-size: 0.9rem; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
      margin-bottom: 1.25rem;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1rem;
      text-align: center;
    }
    .card-val { font-size: 1.45rem; font-weight: 700; }
    .card-val.lime { color: var(--lime); }
    .card-val.coral { color: var(--coral); }
    .card-val.teal { color: var(--teal); }
    .card-val.gold { color: var(--gold); }
    .card-label { font-size: 0.72rem; color: var(--muted); margin-top: 0.25rem; }
    .chart-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }
    @media (max-width: 800px) { .chart-grid { grid-template-columns: 1fr; } }
    .chart-box {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1rem 1rem 0.5rem;
    }
    .chart-box.full { grid-column: 1 / -1; }
    .chart-box h2 {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--muted);
      margin-bottom: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .chart-wrap { position: relative; height: 260px; }
    .chart-wrap.tall { height: 300px; }
    footer {
      margin-top: 1.5rem;
      text-align: center;
      font-size: 0.75rem;
      color: var(--muted);
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1 id="title">Compression Dashboard</h1>
      <p class="sub" id="subtitle"></p>
    </header>
    <div class="cards" id="summaryCards"></div>
    <div class="chart-grid">
      <div class="chart-box full">
        <h2>Compression vs Time</h2>
        <div class="chart-wrap tall"><canvas id="chartCompression"></canvas></div>
      </div>
      <div class="chart-box">
        <h2>Diameter (mm)</h2>
        <div class="chart-wrap"><canvas id="chartDiameter"></canvas></div>
      </div>
      <div class="chart-box">
        <h2>Ball Height (Y position)</h2>
        <div class="chart-wrap"><canvas id="chartHeight"></canvas></div>
      </div>
      <div class="chart-box">
        <h2>Detection Confidence</h2>
        <div class="chart-wrap"><canvas id="chartConfidence"></canvas></div>
      </div>
      <div class="chart-box full">
        <h2>Peak Compression Windows</h2>
        <div class="chart-wrap"><canvas id="chartPeaks"></canvas></div>
      </div>
    </div>
    <footer>Generated by Tennis Ball Tester · frame-by-frame compression analysis</footer>
  </div>
  <script>
    const DATA = __DATA_JSON__;
    document.getElementById("title").textContent = DATA.video_name + " — Dashboard";
    document.getElementById("subtitle").textContent =
      DATA.ball_type + " · " + DATA.frames.length + " frames · " + DATA.fps.toFixed(1) + " fps";

    const m = DATA.metrics;
    const cards = [
      ["Baseline", m.baseline_mm != null ? m.baseline_mm.toFixed(1) + " mm" : "—", "lime"],
      ["Peak Compression", m.max_compression_pct != null ? m.max_compression_pct.toFixed(1) + "%" : "—", "coral"],
      ["Bounces", m.bounce_count != null ? String(m.bounce_count) : "—", "teal"],
      ["Frames Tracked", m.detected_frames != null ? m.detected_frames + "/" + m.frames : "—", "gold"],
      ["Track Rate", m.detection_rate_pct != null ? m.detection_rate_pct + "%" : "—", "teal"],
    ];
    document.getElementById("summaryCards").innerHTML = cards.map(([label, val, cls]) =>
      `<div class="card"><div class="card-val ${cls}">${val}</div><div class="card-label">${label}</div></div>`
    ).join("");

    const times = DATA.frames.map(f => f.time_s);
    const compressions = DATA.frames.map(f => f.compression_pct);
    const diameters = DATA.frames.map(f => f.diameter_mm);
    const heights = DATA.frames.map(f => f.cy_norm);
    const confidences = DATA.frames.map(f => f.confidence * 100);

    const bounceSet = new Set(DATA.bounce_frames);
    const bouncePoints = DATA.bounce_frames.map(fi => {
      const fr = DATA.frames.find(x => x.frame === fi);
      return fr ? { x: fr.time_s, y: fr.compression_pct } : null;
    }).filter(Boolean);

    const chartDefaults = {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { family: "DM Sans" } } },
        tooltip: {
          backgroundColor: "rgba(26,35,50,0.95)",
          titleColor: "#f1f5f9",
          bodyColor: "#94a3b8",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Time (s)", color: "#64748b" },
          ticks: { color: "#64748b" },
          grid: { color: "rgba(255,255,255,0.05)" },
        },
        y: {
          ticks: { color: "#64748b" },
          grid: { color: "rgba(255,255,255,0.05)" },
        },
      },
    };

    function lineChart(id, label, data, color, yLabel, extraDatasets) {
      const ctx = document.getElementById(id);
      new Chart(ctx, {
        type: "line",
        data: {
          datasets: [
            {
              label,
              data: times.map((t, i) => ({ x: t, y: data[i] })),
              borderColor: color,
              backgroundColor: color + "22",
              fill: true,
              tension: 0.15,
              pointRadius: 0,
              borderWidth: 2,
            },
            ...(extraDatasets || []),
          ],
        },
        options: {
          ...chartDefaults,
          scales: {
            ...chartDefaults.scales,
            y: { ...chartDefaults.scales.y, title: { display: true, text: yLabel, color: "#64748b" } },
          },
        },
      });
    }

    lineChart("chartCompression", "Compression %", compressions, "#ff6b4a", "Compression (%)", bouncePoints.length ? [{
      label: "Bounce",
      data: bouncePoints,
      type: "scatter",
      pointRadius: 8,
      pointStyle: "triangle",
      backgroundColor: "#00d4aa",
      borderColor: "#00d4aa",
      showLine: false,
    }] : []);

    lineChart("chartDiameter", "Diameter", diameters, "#4dabf7", "Diameter (mm)");
    lineChart("chartHeight", "Y position", heights, "#c8f135", "Normalized Y (0=top, 1=bottom)");
    lineChart("chartConfidence", "Confidence", confidences, "#fbbf24", "Confidence (%)");

    // Peak windows: bar chart of max compression in 0.5s bins
    const binSec = 0.5;
    const maxTime = times.length ? times[times.length - 1] : 0;
    const bins = [];
    for (let t = 0; t <= maxTime + binSec; t += binSec) {
      let peak = 0;
      for (let i = 0; i < times.length; i++) {
        if (times[i] >= t && times[i] < t + binSec) peak = Math.max(peak, compressions[i]);
      }
      bins.push({ x: t + binSec / 2, y: peak });
    }
    new Chart(document.getElementById("chartPeaks"), {
      type: "bar",
      data: {
        datasets: [{
          label: "Peak compression per 0.5s",
          data: bins,
          backgroundColor: bins.map(b => b.y >= 12 ? "#ff6b4a" : "#00d4aa88"),
          borderRadius: 4,
        }],
      },
      options: {
        ...chartDefaults,
        scales: {
          ...chartDefaults.scales,
          y: { ...chartDefaults.scales.y, title: { display: true, text: "Peak %", color: "#64748b" }, max: Math.max(35, ...(compressions.length ? compressions : [0])) },
        },
      },
    });
  </script>
</body>
</html>
"""

def default_dashboard_path(video_output: Path) -> Path:
    
    stem = video_output.stem
    if stem.endswith("-tracked"):
        stem = stem[: -len("-tracked")]
    return video_output.with_name(f"{stem}-dashboard.html")
