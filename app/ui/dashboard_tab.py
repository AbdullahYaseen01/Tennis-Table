from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.db import RunDatabase
from app.ui.theme import (
    ACCENT_CORAL,
    ACCENT_LIME,
    ACCENT_TEAL,
    CHART_COMPRESSION,
    CHART_RECOVERY,
    CHART_ZONE_FLAG,
    CHART_ZONE_OK,
    SIMPLE_DASHBOARD_STYLESHEET,
    TEXT_SECONDARY,
    style_plot,
)

pg.setConfigOptions(antialias=False, useOpenGL=False)

class _DashboardLoader(QThread):
    runs_loaded = Signal(list, list)

    def __init__(self, db_path, ball_type: str | None) -> None:
        super().__init__()
        self._db_path = db_path
        self._ball_type = ball_type

    def run(self) -> None:
        db = RunDatabase(self._db_path)
        runs = db.list_runs(ball_type=self._ball_type)
        ball_ids = db.list_ball_ids(ball_type=self._ball_type)
        self.runs_loaded.emit(runs, ball_ids)

class _RunDetailLoader(QThread):
    detail_loaded = Signal(int, object)

    def __init__(self, db_path, run_id: int) -> None:
        super().__init__()
        self._db_path = db_path
        self._run_id = run_id

    def run(self) -> None:
        db = RunDatabase(self._db_path)
        run = db.get_run(self._run_id, decimate=True)
        self.detail_loaded.emit(self._run_id, run)

class DashboardTab(QWidget):
    

    def __init__(self, db: RunDatabase, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.db = db
        self._selected_run_id: int | None = None
        self._runs_cache: list[dict] = []
        self._loader: _DashboardLoader | None = None
        self._detail_loader: _RunDetailLoader | None = None
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(250)
        self._filter_timer.timeout.connect(self.refresh)

        self.setStyleSheet(SIMPLE_DASHBOARD_STYLESHEET)
        self._build_ui()
        self._init_plot_items()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        top = QHBoxLayout()
        title = QLabel("Test Results")
        title.setObjectName("dashTitle")
        top.addWidget(title)
        self.count_label = QLabel("")
        self.count_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        top.addWidget(self.count_label)
        top.addStretch()

        top.addWidget(QLabel("Type:"))
        self.type_filter = QComboBox()
        self.type_filter.addItems(["pickleball"])
        self.type_filter.setMinimumWidth(100)
        self.type_filter.currentIndexChanged.connect(self._schedule_refresh)
        top.addWidget(self.type_filter)

        top.addWidget(QLabel("Ball ID:"))
        self.id_filter = QComboBox()
        self.id_filter.setEditable(True)
        self.id_filter.setMinimumWidth(110)
        self.id_filter.currentTextChanged.connect(self._on_id_filter_changed)
        top.addWidget(self.id_filter)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setProperty("class", "secondary")
        self.refresh_btn.clicked.connect(self.refresh)
        top.addWidget(self.refresh_btn)
        layout.addLayout(top)

        stats_row = QHBoxLayout()
        self.stat_runs = self._stat_chip("Runs", "—")
        self.stat_comp = self._stat_chip("Avg Compression", "—")
        self.stat_recovery = self._stat_chip("Avg Recovery", "—")
        stats_row.addWidget(self.stat_runs)
        stats_row.addWidget(self.stat_comp)
        stats_row.addWidget(self.stat_recovery)
        stats_row.addStretch()
        layout.addLayout(stats_row)

        self.summary_label = QLabel("Select a run to view measurement charts.")
        self.summary_label.setObjectName("dashSummary")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.run_table = QTableWidget(0, 5)
        self.run_table.setObjectName("simpleTable")
        self.run_table.setHorizontalHeaderLabels(
            ["Date", "Ball ID", "Type", "Compression", "Recovery"]
        )
        self.run_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.run_table.verticalHeader().setVisible(False)
        self.run_table.setAlternatingRowColors(True)
        self.run_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.run_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.run_table.itemSelectionChanged.connect(self._on_run_selected)
        splitter.addWidget(self.run_table)

        charts = QWidget()
        charts_layout = QVBoxLayout(charts)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(8)

        self.compression_plot = pg.PlotWidget(title="Diameter vs Time")
        style_plot(self.compression_plot)
        self.compression_plot.setMinimumHeight(170)
        self.compression_plot.setLabel("left", "mm")
        self.compression_plot.setLabel("bottom", "s")
        charts_layout.addWidget(self.compression_plot)

        self.zone_plot = pg.PlotWidget(title="Surface Zones (8 sectors)")
        style_plot(self.zone_plot)
        self.zone_plot.setMinimumHeight(130)
        self.zone_plot.setLabel("left", "Score")
        self.zone_plot.setLabel("bottom", "Zone")
        charts_layout.addWidget(self.zone_plot)

        splitter.addWidget(charts)
        splitter.setSizes([460, 540])
        layout.addWidget(splitter, stretch=2)

        trend_box = QFrame()
        trend_box.setObjectName("trendBox")
        trend_layout = QVBoxLayout(trend_box)
        trend_layout.setContentsMargins(12, 8, 12, 8)
        self.trend_hint = QLabel("Filter by Ball ID to see wear trend across tests.")
        self.trend_hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        trend_layout.addWidget(self.trend_hint)

        self.fatigue_plot = pg.PlotWidget(title="Ball Wear Trend")
        style_plot(self.fatigue_plot)
        self.fatigue_plot.setMinimumHeight(150)
        self.fatigue_plot.setLabel("bottom", "Test #")
        self.fatigue_plot.addLegend(offset=(8, 8))
        trend_layout.addWidget(self.fatigue_plot)
        layout.addWidget(trend_box, stretch=1)

    @staticmethod
    def _stat_chip(title: str, value: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("statChip")
        frame.setStyleSheet(
            "QFrame#statChip { background: #1a2332; border: 1px solid rgba(255,255,255,0.08); "
            "border-radius: 8px; }"
        )
        lay = QVBoxLayout(frame)
        lay.setSpacing(2)
        lay.setContentsMargins(10, 6, 10, 6)
        t = QLabel(title.upper())
        t.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; font-weight: 600;")
        v = QLabel(value)
        v.setObjectName("statValue")
        v.setStyleSheet("color: #f1f5f9; font-size: 16px; font-weight: 700;")
        lay.addWidget(t)
        lay.addWidget(v)
        frame._value_label = v  
        return frame

    def _set_stat(self, chip: QFrame, value: str) -> None:
        chip._value_label.setText(value)  

    def _init_plot_items(self) -> None:
        self._comp_curve = self.compression_plot.plot(pen=pg.mkPen(CHART_COMPRESSION, width=2))
        self._recv_scatter = self.compression_plot.plot(
            pen=None, symbol="o", symbolSize=4, symbolBrush=pg.mkBrush(CHART_RECOVERY)
        )
        self._zone_bars = pg.BarGraphItem(x=[], height=[], width=0.65, brushes=[])
        self.zone_plot.addItem(self._zone_bars)
        self._trend_comp = self.fatigue_plot.plot(
            pen=pg.mkPen(ACCENT_LIME, width=2), symbol="o", name="Compression %"
        )
        self._trend_res = self.fatigue_plot.plot(
            pen=pg.mkPen(ACCENT_TEAL, width=2), symbol="o", name="Residual %"
        )

    def _schedule_refresh(self) -> None:
        self._filter_timer.start()

    def refresh(self) -> None:
        if self._loader and self._loader.isRunning():
            return
        ball_type = self.type_filter.currentText()
        ball_type = None if ball_type == "All types" else ball_type
        self.summary_label.setText("Loading runs…")
        self._loader = _DashboardLoader(self.db.db_path, ball_type)
        self._loader.runs_loaded.connect(self._apply_runs)
        self._loader.start()

    @Slot(list, list)
    def _apply_runs(self, runs: list, ball_ids: list) -> None:
        self._runs_cache = runs
        current_id = self.id_filter.currentText()

        self.id_filter.blockSignals(True)
        self.id_filter.clear()
        self.id_filter.addItem("")
        self.id_filter.addItems(ball_ids)
        if current_id:
            idx = self.id_filter.findText(current_id)
            if idx >= 0:
                self.id_filter.setCurrentIndex(idx)
            else:
                self.id_filter.setEditText(current_id)
        self.id_filter.blockSignals(False)

        self.run_table.setUpdatesEnabled(False)
        self.run_table.blockSignals(True)
        self.run_table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            comp_val = run.get("max_compression_pct")
            rec_val = run.get("recovery_t95_s")
            values = [
                run["timestamp"][:16].replace("T", " "),
                run["ball_id"],
                run["ball_type"].capitalize(),
                f"{comp_val:.1f}%" if comp_val is not None else "—",
                f"{rec_val:.1f}s" if rec_val is not None else "—",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setData(Qt.ItemDataRole.UserRole, run["id"])
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == 3 and comp_val is not None:
                    if comp_val > 15:
                        item.setForeground(pg.mkColor(ACCENT_CORAL))
                    elif comp_val > 8:
                        item.setForeground(pg.mkColor("#fbbf24"))
                    else:
                        item.setForeground(pg.mkColor(ACCENT_LIME))
                self.run_table.setItem(row, col, item)
        self.run_table.blockSignals(False)
        self.run_table.setUpdatesEnabled(True)

        n = len(runs)
        self.count_label.setText(f"{n} record{'s' if n != 1 else ''}")
        self._set_stat(self.stat_runs, str(n))

        comps = [r["max_compression_pct"] for r in runs if r.get("max_compression_pct") is not None]
        recs = [r["recovery_t95_s"] for r in runs if r.get("recovery_t95_s") is not None]
        self._set_stat(self.stat_comp, f"{sum(comps)/len(comps):.1f}%" if comps else "—")
        self._set_stat(self.stat_recovery, f"{sum(recs)/len(recs):.1f}s" if recs else "—")

        if n:
            self.summary_label.setText(f"{n} run(s) loaded — select a row for charts.")
        else:
            self.summary_label.setText("No runs saved yet. Complete a test on Live Test and click Save Run.")

        self._update_fatigue_plot()

        if self._selected_run_id:
            self._load_run_detail(self._selected_run_id)

    def _on_id_filter_changed(self) -> None:
        ball_id = self.id_filter.currentText().strip()
        if ball_id:
            for row in range(self.run_table.rowCount()):
                item = self.run_table.item(row, 1)
                if item and item.text() == ball_id:
                    self.run_table.selectRow(row)
                    break
        self._update_fatigue_plot()

    def _on_run_selected(self) -> None:
        items = self.run_table.selectedItems()
        if not items:
            return
        run_id = items[0].data(Qt.ItemDataRole.UserRole)
        if run_id is None:
            return
        self._load_run_detail(int(run_id))

    def _load_run_detail(self, run_id: int) -> None:
        if self._detail_loader and self._detail_loader.isRunning():
            self._detail_loader.requestInterruption()
        self._selected_run_id = run_id
        self._detail_loader = _RunDetailLoader(self.db.db_path, run_id)
        self._detail_loader.detail_loaded.connect(self._apply_run_detail)
        self._detail_loader.start()

    @Slot(int, object)
    def _apply_run_detail(self, run_id: int, run: dict | None) -> None:
        if run is None or run_id != self._selected_run_id:
            return

        baseline = f"{run['baseline_mm']:.1f} mm" if run.get("baseline_mm") else "—"
        comp = f"{run['max_compression_pct']:.1f}%" if run.get("max_compression_pct") is not None else "—"
        recovery = f"{run['recovery_t95_s']:.1f}s" if run.get("recovery_t95_s") is not None else "—"
        residual = f"{run['residual_pct']:.1f}%" if run.get("residual_pct") is not None else "—"
        acc = (
            f"{run['accuracy_error_pct']:+.1f}%"
            if run.get("accuracy_error_pct") == run.get("accuracy_error_pct")
            else "—"
        )

        self.summary_label.setText(
            f"Run #{run_id}  ·  {run['ball_id']} ({run['ball_type']})  ·  "
            f"Baseline {baseline}  ·  Peak {comp}  ·  Recovery {recovery}  ·  "
            f"Residual {residual}  ·  Accuracy {acc}"
        )

        ts = run.get("timeseries", [])
        if ts:
            t = np.array([p["t_seconds"] for p in ts], dtype=np.float64)
            d = np.array([p["diameter_mm"] for p in ts], dtype=np.float64)
            self._comp_curve.setData(t, d)

            recovery_pts = [p for p in ts if p.get("phase") in ("RECOVERING", "RELEASED")]
            if recovery_pts:
                rt = np.array([p["t_seconds"] for p in recovery_pts])
                rd = np.array([p["diameter_mm"] for p in recovery_pts])
                self._recv_scatter.setData(rt, rd)
            else:
                self._recv_scatter.setData([], [])
        else:
            self._comp_curve.setData([], [])
            self._recv_scatter.setData([], [])

        zones = run.get("zone_scores", [])
        if zones:
            x = [z["zone_index"] for z in zones]
            y = [z["score"] for z in zones]
            colors = [CHART_ZONE_FLAG if z["flagged"] else CHART_ZONE_OK for z in zones]
            brushes = [pg.mkBrush(c) for c in colors]
            self._zone_bars.setOpts(x=x, height=y, width=0.65, brushes=brushes)
        else:
            self._zone_bars.setOpts(x=[], height=[], brushes=[])

    def _update_fatigue_plot(self) -> None:
        ball_id = self.id_filter.currentText().strip()
        if not ball_id:
            self._trend_comp.setData([], [])
            self._trend_res.setData([], [])
            self.trend_hint.setText("Filter by Ball ID to see wear trend across tests.")
            return

        trend = self.db.get_fatigue_trend(ball_id)
        if len(trend) < 2:
            self._trend_comp.setData([], [])
            self._trend_res.setData([], [])
            self.trend_hint.setText(f"'{ball_id}': {len(trend)} test(s) — need 2+ for trend.")
            return

        self.trend_hint.setText(f"Wear trend — {ball_id} ({len(trend)} tests)")
        x = np.arange(1, len(trend) + 1, dtype=np.float64)
        comp = np.array([t["max_compression_pct"] or 0 for t in trend])
        residual = np.array([t["residual_pct"] or 0 for t in trend])
        self._trend_comp.setData(x, comp)
        self._trend_res.setData(x, residual)
