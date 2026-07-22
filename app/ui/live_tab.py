"""Live video tab with real-time measurement and test controls."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.config import DEFAULT_SOURCE, SAMPLES_DIR
from app.core.db import RunDatabase
from app.core.models import CompressionSample, RecoveryResult, TestState
from app.core.pipeline import TestPipeline
from app.vision.calibration import Calibrator
from app.vision.camera import FrameSource

logger = logging.getLogger(__name__)


class VisionWorker(QThread):
    """Read frames and run vision pipeline off the UI thread."""

    frame_ready = Signal(object, object, object)  # overlay frame, measurement, compression
    error = Signal(str)

    def __init__(self, pipeline: TestPipeline, source: FrameSource) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.source = source
        self._running = True
        self._raw_frame: np.ndarray | None = None

    def stop(self) -> None:
        self._running = False

    def set_raw_frame_override(self, frame: np.ndarray | None) -> None:
        self._raw_frame = frame

    def run(self) -> None:
        while self._running:
            if self._raw_frame is not None:
                frame = self._raw_frame
                ts = __import__("time").perf_counter()
                ok = True
            else:
                ok, frame, ts = self.source.read()
            if not ok or frame is None:
                self.msleep(10)
                continue

            measurement, compression = self.pipeline.process_frame(frame, ts)
            overlay = self.pipeline.detector.draw_overlay(frame, measurement)

            # State overlay
            state = self.pipeline.state.value
            comp_pct = compression.compression_pct if compression else self.pipeline.compression.live_compression_pct()
            cv2.putText(
                overlay,
                f"State: {state}  Compression: {comp_pct:.1f}%",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            self.frame_ready.emit(overlay, measurement, compression)
            self.msleep(1)


class LiveTab(QWidget):
    """Camera view + real-time measurement + test state machine controls."""

    run_saved = Signal(int)

    def __init__(
        self,
        pipeline: TestPipeline,
        calibrator: Calibrator,
        db: RunDatabase,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.db = db
        self.source: FrameSource | None = None
        self.worker: VisionWorker | None = None

        self._compression_times: list[float] = []
        self._compression_values: list[float] = []
        self._recovery_times: list[float] = []
        self._recovery_values: list[float] = []
        self._recovery_release_t: float | None = None
        self._plot_frame_counter = 0
        self._last_pixmap: QPixmap | None = None

        self._build_ui()
        self._connect_pipeline()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: video + controls
        left = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left)

        self.video_label = QLabel("Open a video source to begin")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: #111; color: #aaa;")
        left.addWidget(self.video_label)

        self.status_label = QLabel("State: IDLE | Compression: — | Confidence: —")
        self.status_label.setStyleSheet("font-size: 13px; padding: 4px;")
        left.addWidget(self.status_label)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self.source_edit = QLineEdit(DEFAULT_SOURCE)
        self.source_edit.setPlaceholderText("Video path, webcam index (0), or RTSP URL")
        src_row.addWidget(self.source_edit, stretch=1)
        self.browse_btn = QPushButton("Browse Video…")
        self.browse_btn.setProperty("class", "secondary")
        self.browse_btn.clicked.connect(self._browse_video)
        src_row.addWidget(self.browse_btn)
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self._open_source)
        src_row.addWidget(self.open_btn)
        left.addLayout(src_row)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Ball type:"))
        self.ball_type_combo = QComboBox()
        self.ball_type_combo.addItems(["tennis", "pickleball"])
        self.ball_type_combo.currentTextChanged.connect(self._on_ball_type_changed)
        cfg_row.addWidget(self.ball_type_combo)
        cfg_row.addWidget(QLabel("Ball ID:"))
        self.ball_id_edit = QLineEdit("ball-001")
        cfg_row.addWidget(self.ball_id_edit)
        left.addLayout(cfg_row)

        btn_row = QHBoxLayout()
        self.baseline_btn = QPushButton("Capture Baseline")
        self.baseline_btn.clicked.connect(self._capture_baseline)
        btn_row.addWidget(self.baseline_btn)
        self.start_btn = QPushButton("Start Test")
        self.start_btn.clicked.connect(self._start_test)
        btn_row.addWidget(self.start_btn)
        self.surface_btn = QPushButton("Run Surface Scan")
        self.surface_btn.clicked.connect(self._surface_scan)
        btn_row.addWidget(self.surface_btn)
        self.save_btn = QPushButton("Save Run")
        self.save_btn.clicked.connect(self._save_run)
        btn_row.addWidget(self.save_btn)
        left.addLayout(btn_row)

        self.surface_progress = QLabel("")
        left.addWidget(self.surface_progress)

        splitter.addWidget(left_widget)

        # Right: plots
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        pg.setConfigOptions(antialias=True)

        self.live_compression_plot = pg.PlotWidget(title="Live Compression")
        self.live_compression_plot.setLabel("left", "Compression", units="%")
        self.live_compression_plot.setLabel("bottom", "Time", units="s")
        self.compression_curve = self.live_compression_plot.plot(pen=pg.mkPen("y", width=2))
        plots_layout.addWidget(self.live_compression_plot)

        self.live_recovery_plot = pg.PlotWidget(title="Recovery")
        self.live_recovery_plot.setLabel("left", "Diameter", units="mm")
        self.live_recovery_plot.setLabel("bottom", "Time since release", units="s")
        self.recovery_scatter = self.live_recovery_plot.plot(
            pen=None, symbol="o", symbolSize=4, symbolBrush="c"
        )
        self.recovery_fit_curve = self.live_recovery_plot.plot(pen=pg.mkPen("m", width=2))
        plots_layout.addWidget(self.live_recovery_plot)

        self.recovery_info = QLabel("Recovery: —")
        plots_layout.addWidget(self.recovery_info)

        splitter.addWidget(plots_widget)
        splitter.setSizes([650, 450])
        layout.addWidget(splitter)

    def _connect_pipeline(self) -> None:
        self.pipeline.on("state_changed", self._on_state_changed)
        self.pipeline.on("compression", self._on_compression)
        self.pipeline.on("recovery", self._on_recovery)
        self.pipeline.on("surface", self._on_surface)
        self.pipeline.on("scale_updated", self._on_scale_updated)

    def _on_ball_type_changed(self, ball_type: str) -> None:
        self.pipeline.set_ball_type(ball_type)

    def _browse_video(self) -> None:
        start_dir = str(SAMPLES_DIR)
        current = self.source_edit.text().strip()
        if current and Path(current).parent.exists():
            start_dir = str(Path(current).parent)

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            start_dir,
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.webm);;All Files (*.*)",
        )
        if not path:
            return

        self.source_edit.setText(path)
        self._open_source()

    def _open_source(self) -> None:
        self._stop_worker()
        try:
            src = self.source_edit.text().strip()
            source: str | int = int(src) if src.isdigit() else src
            self.source = FrameSource(source)
            self.worker = VisionWorker(self.pipeline, self.source)
            self.worker.frame_ready.connect(self._on_frame_ready)
            self.worker.error.connect(lambda e: QMessageBox.warning(self, "Error", e))
            self.worker.start()
        except Exception as exc:
            QMessageBox.warning(self, "Source Error", str(exc))

    def _stop_worker(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        if self.source:
            self.source.release()
            self.source = None

    @Slot(object, object, object)
    def _on_frame_ready(self, overlay, measurement, compression) -> None:
        self._plot_frame_counter += 1
        if self._plot_frame_counter % 3 == 0 or self._last_pixmap is None:
            self._show_frame(overlay)
        if measurement:
            conf = measurement.detection_confidence
            comp = compression.compression_pct if compression else 0.0
            baseline = self.pipeline.compression.baseline_mm
            baseline_str = f"{baseline:.2f} mm" if baseline else "—"
            self.status_label.setText(
                f"State: {self.pipeline.state.value} | "
                f"Compression: {comp:.1f}% | "
                f"Baseline: {baseline_str} | "
                f"D: {measurement.minor_mm:.2f}/{measurement.major_mm:.2f} mm | "
                f"Conf: {conf:.2f}"
            )

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._last_pixmap = self.video_label.pixmap()

    def _capture_baseline(self) -> None:
        if not self.calibrator.is_ready():
            QMessageBox.warning(self, "Calibration", "Complete calibration first (Calibration tab).")
            return
        self._compression_times.clear()
        self._compression_values.clear()
        self._recovery_times.clear()
        self._recovery_values.clear()
        self.pipeline.reset_test()
        self.pipeline.start_baseline_capture()

    def _start_test(self) -> None:
        if self.pipeline.compression.baseline_mm is None:
            QMessageBox.information(self, "Baseline", "Capture baseline first.")
            return
        self._compression_times.clear()
        self._compression_values.clear()
        self._recovery_times.clear()
        self._recovery_values.clear()
        self._recovery_release_t = None
        self.recovery_fit_curve.setData([], [])
        self.pipeline.start_test()

    def _surface_scan(self) -> None:
        if self.pipeline.last_measurement is None:
            QMessageBox.warning(self, "Surface", "No ball detected.")
            return
        self.pipeline.start_surface_scan()
        # Single-view mode for quick scan; rotate mode via repeated captures
        if self.source and self.worker:
            ok, frame, _ = self.source.read()
            if ok and frame is not None:
                result = self.pipeline.analyze_surface_single_view(frame)
                if result:
                    flagged = [z.zone_index for z in result.zones if z.flagged]
                    self.surface_progress.setText(
                        f"Surface scan complete (front hemisphere). Flagged zones: {flagged or 'none'}"
                    )

    def _save_run(self) -> None:
        if self.pipeline.state == TestState.RECOVERING:
            self.pipeline.force_finish_recovery()
        self.pipeline.set_ball_id(self.ball_id_edit.text().strip() or "ball-001")
        summary = self.pipeline.build_run_summary()
        if summary is None:
            QMessageBox.warning(self, "Save", "No baseline captured — run a test first.")
            return
        run_id = self.db.save_run(summary)
        self.run_saved.emit(run_id)
        QMessageBox.information(self, "Saved", f"Run #{run_id} saved to database.")

    def _on_state_changed(self, state: TestState) -> None:
        if state == TestState.RECOVERING:
            self._recovery_release_t = None

    def _on_compression(self, sample: CompressionSample) -> None:
        if self._plot_frame_counter % 2 != 0:
            return
        t0 = self._compression_times[0] if self._compression_times else sample.timestamp
        if not self._compression_times:
            t0 = sample.timestamp
        self._compression_times.append(sample.timestamp - t0)
        self._compression_values.append(sample.compression_pct)
        self.compression_curve.setData(self._compression_times, self._compression_values)

        if self.pipeline.state == TestState.RECOVERING:
            if self._recovery_release_t is None:
                self._recovery_release_t = sample.timestamp
            t = sample.timestamp - self._recovery_release_t
            self._recovery_times.append(t)
            self._recovery_values.append(sample.diameter_mm)
            self.recovery_scatter.setData(self._recovery_times, self._recovery_values)

    def _on_recovery(self, result: RecoveryResult) -> None:
        self.recovery_info.setText(
            f"Recovery: tau={result.tau_s:.3f}s | t95={result.t95_s:.3f}s | "
            f"Residual={result.residual_pct:.2f}% | Confidence={result.fit_confidence:.2f}"
        )
        if result.fitted_curve:
            t, d = result.fitted_curve
            self.recovery_fit_curve.setData(t, d)

    def _on_surface(self, result) -> None:
        flagged = [z.zone_index for z in result.zones if z.flagged]
        self.surface_progress.setText(f"Surface analysis: flagged zones {flagged or 'none'}")

    def _on_scale_updated(self, error_pct: float) -> None:
        self.surface_progress.setText(
            f"Scale auto-calibrated from baseline (accuracy {error_pct:+.2f}% vs known ball spec)."
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_worker()
        super().closeEvent(event)
