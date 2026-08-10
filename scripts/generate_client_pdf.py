from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def _fmt_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %B %Y")
    except ValueError:
        return iso[:10]

def build_pdf(report: dict, output_path: Path) -> None:
    video = report.get("video", "Test video")
    test_date = _fmt_date(report.get("tested_at", ""))
    ball_type = report.get("ball_type", "tennis").title()
    spec_mm = report.get("known_diameter_mm", 67.0)

    baseline = report.get("baseline_mm")
    max_comp = report.get("max_compression_pct")
    rest_mm = report.get("rest_mean_minor_mm")
    rest_err = report.get("rest_error_pct")
    det_rate = report.get("detection_rate_pct")
    completed = report.get("test_completed", False)

    fig = plt.figure(figsize=(8.27, 11.69))  
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, transform=ax.transAxes, color="#1a472a", zorder=0))
    ax.text(0.5, 0.94, "Tennis Ball Condition Test Report", transform=ax.transAxes,
            ha="center", va="center", fontsize=22, fontweight="bold", color="white")
    ax.text(0.5, 0.905, "Automated vision analysis", transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color="#c8e6c9")

    y = 0.82
    line = 0.045

    def section(title: str) -> None:
        nonlocal y
        ax.text(0.08, y, title, transform=ax.transAxes, fontsize=13, fontweight="bold", color="#1a472a")
        y -= line * 0.6
        ax.plot([0.08, 0.92], [y, y], transform=ax.transAxes, color="#1a472a", linewidth=1.2)
        y -= line

    def row(label: str, value: str, bold: bool = False) -> None:
        nonlocal y
        weight = "bold" if bold else "normal"
        ax.text(0.1, y, label, transform=ax.transAxes, fontsize=11, color="#333333")
        ax.text(0.92, y, value, transform=ax.transAxes, fontsize=11, fontweight=weight,
                ha="right", color="#111111")
        y -= line

    
    section("Test Information")
    row("Video file", video)
    row("Test date", test_date)
    row("Ball type", f"{ball_type} (reference {spec_mm:.0f} mm)")
    row("Status", "Completed" if completed else "Incomplete", bold=True)
    y -= line * 0.3

    
    section("Results")
    if baseline is not None:
        row("Baseline diameter (at rest)", f"{baseline:.1f} mm", bold=True)
    if rest_mm is not None and rest_err is not None:
        row("Diameter vs specification", f"{rest_mm:.1f} mm  ({rest_err:+.1f}%)")
    if max_comp is not None:
        row("Maximum compression under load", f"{max_comp:.1f} %", bold=True)
    if det_rate is not None:
        row("Ball tracking success rate", f"{det_rate:.0f} %")
    y -= line * 0.3

    
    section("Summary")
    summary_lines = [
        "The ball was analysed under load using computer vision.",
        "Compression % measures deformation relative to the ball's own resting size",
        "and is the primary indicator of ball condition.",
    ]
    if max_comp is not None:
        summary_lines.append(f"This test recorded up to {max_comp:.1f}% compression under load.")
    if rest_err is not None and abs(rest_err) < 5:
        summary_lines.append(f"Resting diameter matched the {spec_mm:.0f} mm specification within {abs(rest_err):.1f}%.")

    for text in summary_lines:
        ax.text(0.1, y, text, transform=ax.transAxes, fontsize=10.5, color="#444444", wrap=True)
        y -= line * 0.85

    
    ax.text(0.5, 0.04, "Tennis Ball & Pickleball Condition Testing System",
            transform=ax.transAxes, ha="center", fontsize=9, color="#888888")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output_path)) as pdf:
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main() -> int:
    stem = "test-video"
    json_path = ROOT / "data" / "reports" / f"{stem}-test-report.json"
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"Report not found: {json_path}")
        return 1

    report = json.loads(json_path.read_text(encoding="utf-8"))
    out = json_path.parent / f"{json_path.stem.replace('-test-report', '')}-CLIENT-REPORT.pdf"
    if len(sys.argv) > 2:
        out = Path(sys.argv[2])

    build_pdf(report, out)
    print(f"PDF saved: {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
