from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from app.config import DATA_DIR
from app.ui.theme import APP_STYLESHEET
from app.core.db import RunDatabase
from app.core.pipeline import TestPipeline
from app.ui.calibration_tab import CalibrationTab
from app.ui.dashboard_tab import DashboardTab
from app.ui.live_tab import LiveTab
from app.vision.calibration import Calibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pickleball Condition Tester")
        self.resize(1280, 800)

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.calibrator = Calibrator()
        self.db = RunDatabase()
        self.pipeline = TestPipeline(self.calibrator)

        tabs = QTabWidget()
        self.live_tab = LiveTab(self.pipeline, self.calibrator, self.db)
        self.calibration_tab = CalibrationTab(self.calibrator)
        self.dashboard_tab = DashboardTab(self.db)

        tabs.addTab(self.live_tab, "Live Test")
        tabs.addTab(self.calibration_tab, "Calibration")
        tabs.addTab(self.dashboard_tab, "Dashboard")

        self.calibration_tab.calibration_updated.connect(self._on_calibration_updated)
        self.live_tab.run_saved.connect(self._on_run_saved)
        self.live_tab.pipeline.on("scale_updated", lambda _e: self._on_calibration_updated())

        self.setCentralWidget(tabs)
        self._update_title()

    def _on_calibration_updated(self) -> None:
        self._update_title()

    def _on_run_saved(self, run_id: int) -> None:
        logger.info("Run %d saved", run_id)
        self.dashboard_tab.refresh()

    def _update_title(self) -> None:
        parts = ["Pickleball Condition Tester"]
        if self.calibrator.is_ready():
            parts.append("Calibrated")
            val = self.calibrator.get_validation()
            if val:
                parts.append(f"Accuracy: {val.error_pct:+.1f}%")
        else:
            parts.append("Calibration required")
        self.setWindowTitle(" — ".join(parts))

def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
