from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import (
    CHECKERBOARD_COLS,
    CHECKERBOARD_ROWS,
    CHECKERBOARD_SQUARE_MM,
    DEFAULT_SOURCE,
    REFERENCE_DIAMETERS_MM,
    SAMPLES_DIR,
)
from app.vision.calibration import Calibrator
from app.vision.camera import FrameSource
from app.vision.detect import BallDetector

class CalibrationTab(QWidget):
    

    calibration_updated = Signal()

    def __init__(self, calibrator: Calibrator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.calibrator = calibrator
        self.detector = BallDetector(calibrator)
        self.source: FrameSource | None = None
        self._ball_scale_frames: list = []
        self._current_frame: np.ndarray | None = None
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_frame)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)

        
        left = QVBoxLayout()
        self.preview_label = QLabel("Open a video source to begin calibration")
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background: #1a1a1a; color: #ccc;")
        left.addWidget(self.preview_label)

        src_row = QHBoxLayout()
        self.source_edit = QLineEdit(DEFAULT_SOURCE)
        self.source_edit.setPlaceholderText("Video path, webcam index (0), or RTSP URL")
        src_row.addWidget(QLabel("Source:"))
        src_row.addWidget(self.source_edit, stretch=1)
        self.browse_btn = QPushButton("Browse Video…")
        self.browse_btn.setProperty("class", "secondary")
        self.browse_btn.clicked.connect(self._browse_video)
        src_row.addWidget(self.browse_btn)
        self.open_btn = QPushButton("Open Source")
        self.open_btn.clicked.connect(self._open_source)
        src_row.addWidget(self.open_btn)
        left.addLayout(src_row)
        layout.addLayout(left, stretch=2)

        
        right = QVBoxLayout()

        
        intrinsics_box = QGroupBox("2a. Lens Intrinsics (Checkerboard)")
        intrinsics_layout = QVBoxLayout(intrinsics_box)
        intrinsics_layout.addWidget(
            QLabel(
                f"Print a {CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS} checkerboard "
                f"({CHECKERBOARD_SQUARE_MM} mm squares). Capture ~15 varied angles."
            )
        )
        self.intrinsic_count_label = QLabel("Frames captured: 0")
        intrinsics_layout.addWidget(self.intrinsic_count_label)
        btn_row = QHBoxLayout()
        self.capture_intrinsic_btn = QPushButton("Capture Frame")
        self.capture_intrinsic_btn.clicked.connect(self._capture_intrinsic)
        btn_row.addWidget(self.capture_intrinsic_btn)
        self.compute_intrinsic_btn = QPushButton("Compute Intrinsics")
        self.compute_intrinsic_btn.clicked.connect(self._compute_intrinsics)
        btn_row.addWidget(self.compute_intrinsic_btn)
        self.reset_intrinsic_btn = QPushButton("Reset")
        self.reset_intrinsic_btn.clicked.connect(self._reset_intrinsics)
        btn_row.addWidget(self.reset_intrinsic_btn)
        intrinsics_layout.addLayout(btn_row)
        self.intrinsic_status = QLabel("Status: not calibrated")
        intrinsics_layout.addWidget(self.intrinsic_status)
        right.addWidget(intrinsics_box)

        
        scale_box = QGroupBox("2b. Real-World Scale (same plane as ball)")
        scale_layout = QVBoxLayout(scale_box)
        scale_layout.addWidget(
            QLabel(
                "Place checkerboard at the EXACT plane where the ball will sit. "
                "Parallax error occurs if the reference is at a different depth."
            )
        )
        self.calibrate_scale_btn = QPushButton("Calibrate Scale from Checkerboard")
        self.calibrate_scale_btn.clicked.connect(self._calibrate_scale)
        scale_layout.addWidget(self.calibrate_scale_btn)

        ball_scale_row = QHBoxLayout()
        self.ball_type_combo = QComboBox()
        self.ball_type_combo.addItems(["tennis", "pickleball"])
        ball_scale_row.addWidget(QLabel("Ball:"))
        ball_scale_row.addWidget(self.ball_type_combo)
        self.capture_ball_scale_btn = QPushButton("Add Frame")
        self.capture_ball_scale_btn.clicked.connect(self._capture_ball_scale_frame)
        ball_scale_row.addWidget(self.capture_ball_scale_btn)
        self.ball_scale_btn = QPushButton("Calibrate Scale from Ball (Recommended)")
        self.ball_scale_btn.clicked.connect(self._calibrate_scale_from_ball)
        scale_layout.addLayout(ball_scale_row)
        scale_layout.addWidget(self.ball_scale_btn)
        self.ball_scale_count = QLabel("Ball frames: 0 (need ~20, ball at rest)")
        scale_layout.addWidget(self.ball_scale_count)
        self.scale_status = QLabel("Scale: not set")
        scale_layout.addWidget(self.scale_status)
        right.addWidget(scale_box)

        
        val_box = QGroupBox("2c. Accuracy Validation")
        val_layout = QFormLayout(val_box)
        self.known_diameter_spin = QDoubleSpinBox()
        self.known_diameter_spin.setRange(1, 200)
        self.known_diameter_spin.setValue(67.0)
        self.known_diameter_spin.setSuffix(" mm")
        val_layout.addRow("Known diameter:", self.known_diameter_spin)
        self.validate_btn = QPushButton("Validate Accuracy")
        self.validate_btn.clicked.connect(self._validate)
        val_layout.addRow(self.validate_btn)
        self.validation_label = QLabel("Validation: not performed")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        val_layout.addRow(self.validation_label)
        right.addWidget(val_box)

        right.addStretch()
        layout.addLayout(right, stretch=1)
        self._refresh_status()

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
        self._close_source()
        try:
            src = self.source_edit.text().strip()
            source: str | int = int(src) if src.isdigit() else src
            self.source = FrameSource(source)
            self._timer.start(33)
        except Exception as exc:
            QMessageBox.warning(self, "Source Error", str(exc))

    def _close_source(self) -> None:
        self._timer.stop()
        if self.source:
            self.source.release()
            self.source = None

    def _poll_frame(self) -> None:
        if not self.source:
            return
        ok, frame, _ = self.source.read()
        if not ok or frame is None:
            return
        self._current_frame = frame
        display = self.calibrator.undistort(frame)
        found, corners = self.calibrator.detect_checkerboard(display)
        if found and corners is not None:
            display = self.calibrator.draw_checkerboard(display, corners)
        self._show_frame(display)

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _capture_intrinsic(self) -> None:
        if self._current_frame is None:
            return
        if self.calibrator.add_intrinsic_frame(self._current_frame):
            self.intrinsic_count_label.setText(
                f"Frames captured: {self.calibrator.intrinsic_frame_count()}"
            )
        else:
            QMessageBox.information(self, "Capture", "Checkerboard not detected in frame.")

    def _compute_intrinsics(self) -> None:
        if self._current_frame is None:
            return
        h, w = self._current_frame.shape[:2]
        if self.calibrator.compute_intrinsics((w, h)):
            self.intrinsic_status.setText("Status: calibrated ✓")
            self.calibration_updated.emit()
            self._refresh_status()
        else:
            QMessageBox.warning(
                self,
                "Calibration",
                f"Need at least 10 frames (have {self.calibrator.intrinsic_frame_count()}).",
            )

    def _reset_intrinsics(self) -> None:
        self.calibrator.reset_intrinsic_capture()
        self.intrinsic_count_label.setText("Frames captured: 0")

    def _capture_ball_scale_frame(self) -> None:
        if self._current_frame is None:
            return
        self.detector.set_ball_type(self.ball_type_combo.currentText())
        self._ball_scale_frames.append(self._current_frame.copy())
        self.ball_scale_count.setText(
            f"Ball frames: {len(self._ball_scale_frames)} (need ~20, ball at rest)"
        )

    def _calibrate_scale_from_ball(self) -> None:
        if len(self._ball_scale_frames) < 10:
            QMessageBox.information(
                self,
                "Ball Scale",
                "Capture at least 10 frames with the ball at rest (Add Frame).",
            )
            return
        ball_type = self.ball_type_combo.currentText()
        self.detector.set_ball_type(ball_type)
        known_mm = REFERENCE_DIAMETERS_MM.get(ball_type, 67.0)
        d_px = self.detector.measure_median_diameter_px(self._ball_scale_frames, axis="minor")
        if d_px is None:
            QMessageBox.warning(self, "Ball Scale", "Could not detect ball in captured frames.")
            return
        if self.calibrator.calibrate_scale_from_ball_diameter(d_px, known_mm):
            self._refresh_status()
            self.calibration_updated.emit()
            val = self.calibrator.validate_with_measurement(known_mm, known_mm)
            QMessageBox.information(
                self,
                "Ball Scale",
                f"Scale set: {self.calibrator.pixels_per_mm:.3f} px/mm\n"
                f"From {len(self._ball_scale_frames)} frames, "
                f"reference {known_mm:.1f} mm ball.\n"
                f"Expected error after calibration: ~{val.error_pct:.2f}%",
            )
        else:
            QMessageBox.warning(self, "Ball Scale", "Scale calibration failed.")

    def _calibrate_scale(self) -> None:
        if self._current_frame is None:
            return
        if self.calibrator.calibrate_scale_from_frame(self._current_frame):
            self._refresh_status()
            self.calibration_updated.emit()
            QMessageBox.information(
                self,
                "Scale",
                f"Scale calibrated: {self.calibrator.pixels_per_mm:.2f} px/mm",
            )
        else:
            QMessageBox.warning(self, "Scale", "Checkerboard not detected.")

    def _validate(self) -> None:
        if self._current_frame is None:
            return
        known = self.known_diameter_spin.value()
        self.detector.set_ball_type(self.ball_type_combo.currentText())
        result = self.calibrator.validate_known_diameter(
            self._current_frame, known, detector=self.detector
        )
        if result is None:
            QMessageBox.warning(self, "Validation", "Could not measure object. Ensure ball/coin is visible.")
            return
        self.validation_label.setText(
            f"Measured: {result.measured_mm:.2f} mm | Known: {result.known_mm:.2f} mm\n"
            f"Error: {result.error_mm:+.2f} mm ({result.error_pct:+.2f}%)"
        )
        color = "green" if abs(result.error_pct) < 2 else "orange" if abs(result.error_pct) < 5 else "red"
        self.validation_label.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {color};")
        self.calibration_updated.emit()

    def _refresh_status(self) -> None:
        if self.calibrator.camera_matrix is not None:
            self.intrinsic_status.setText("Status: calibrated ✓")
        if self.calibrator.pixels_per_mm:
            self.scale_status.setText(f"Scale: {self.calibrator.pixels_per_mm:.2f} px/mm")
        val = self.calibrator.get_validation()
        if val:
            self.validation_label.setText(
                f"Measured: {val.measured_mm:.2f} mm | Known: {val.known_mm:.2f} mm\n"
                f"Error: {val.error_mm:+.2f} mm ({val.error_pct:+.2f}%)"
            )

    def closeEvent(self, event) -> None:  
        self._close_source()
        super().closeEvent(event)
