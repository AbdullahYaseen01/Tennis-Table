from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import DB_PATH
from app.core.models import CompressionSample, TestRunSummary, ZoneResult

DASHBOARD_MAX_CHART_POINTS = 500

class RunDatabase:
    

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    ball_type TEXT,
                    ball_id TEXT,
                    baseline_mm REAL,
                    max_compression_pct REAL,
                    recovery_tau_s REAL,
                    recovery_t95_s REAL,
                    residual_pct REAL,
                    accuracy_error_pct REAL,
                    notes TEXT
                );
                CREATE TABLE IF NOT EXISTS zone_scores (
                    run_id INTEGER,
                    zone_index INTEGER,
                    score REAL,
                    flagged INTEGER,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                );
                CREATE TABLE IF NOT EXISTS timeseries (
                    run_id INTEGER,
                    t_seconds REAL,
                    diameter_mm REAL,
                    phase TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_ball_id ON runs(ball_id);
                CREATE INDEX IF NOT EXISTS idx_runs_ball_type ON runs(ball_type);
                CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_timeseries_run ON timeseries(run_id, t_seconds);
                CREATE INDEX IF NOT EXISTS idx_zones_run ON zone_scores(run_id, zone_index);
                """
            )

    def save_run(self, summary: TestRunSummary) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (
                    timestamp, ball_type, ball_id, baseline_mm, max_compression_pct,
                    recovery_tau_s, recovery_t95_s, residual_pct, accuracy_error_pct, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    summary.ball_type,
                    summary.ball_id,
                    summary.baseline_mm,
                    summary.max_compression_pct,
                    summary.recovery_tau_s,
                    summary.recovery_t95_s,
                    summary.residual_pct,
                    summary.accuracy_error_pct,
                    summary.notes,
                ),
            )
            run_id = int(cur.lastrowid)

            if summary.zone_scores:
                conn.executemany(
                    "INSERT INTO zone_scores (run_id, zone_index, score, flagged) VALUES (?, ?, ?, ?)",
                    [
                        (run_id, z.zone_index, z.score, int(z.flagged))
                        for z in summary.zone_scores
                    ],
                )

            if summary.timeseries:
                t0 = summary.timeseries[0].timestamp
                conn.executemany(
                    "INSERT INTO timeseries (run_id, t_seconds, diameter_mm, phase) VALUES (?, ?, ?, ?)",
                    [
                        (run_id, s.timestamp - t0, s.diameter_mm, s.phase)
                        for s in summary.timeseries
                    ],
                )

            conn.commit()
            return run_id

    def list_runs(
        self,
        ball_type: str | None = None,
        ball_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT id, timestamp, ball_type, ball_id, baseline_mm, max_compression_pct, "
            "recovery_t95_s, recovery_tau_s, residual_pct, accuracy_error_pct "
            "FROM runs WHERE 1=1"
        )
        params: list[Any] = []
        if ball_type:
            query += " AND ball_type = ?"
            params.append(ball_type)
        if ball_id:
            query += " AND ball_id = ?"
            params.append(ball_id)
        query += " ORDER BY timestamp DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def list_ball_ids(self, ball_type: str | None = None) -> list[str]:
        query = "SELECT DISTINCT ball_id FROM runs WHERE ball_id IS NOT NULL AND ball_id != ''"
        params: list[Any] = []
        if ball_type:
            query += " AND ball_type = ?"
            params.append(ball_type)
        query += " ORDER BY ball_id"
        with self._connect() as conn:
            return [row[0] for row in conn.execute(query, params).fetchall()]

    @staticmethod
    def decimate_series(
        points: list[dict[str, Any]],
        max_points: int = DASHBOARD_MAX_CHART_POINTS,
    ) -> list[dict[str, Any]]:
        
        n = len(points)
        if n <= max_points:
            return points
        step = max(1, n // max_points)
        sampled = points[::step]
        if sampled[-1] is not points[-1]:
            sampled.append(points[-1])
        return sampled

    def get_run(
        self,
        run_id: int,
        *,
        include_timeseries: bool = True,
        decimate: bool = True,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["zone_scores"] = [
                dict(z)
                for z in conn.execute(
                    "SELECT zone_index, score, flagged FROM zone_scores "
                    "WHERE run_id = ? ORDER BY zone_index",
                    (run_id,),
                ).fetchall()
            ]
            if include_timeseries:
                ts = [
                    dict(t)
                    for t in conn.execute(
                        "SELECT t_seconds, diameter_mm, phase FROM timeseries "
                        "WHERE run_id = ? ORDER BY t_seconds",
                        (run_id,),
                    ).fetchall()
                ]
                result["timeseries"] = self.decimate_series(ts) if decimate else ts
            else:
                result["timeseries"] = []
            return result

    def get_fatigue_trend(self, ball_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, max_compression_pct, recovery_t95_s, residual_pct, baseline_mm
                FROM runs WHERE ball_id = ? ORDER BY timestamp ASC
                """,
                (ball_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_run(self, run_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM zone_scores WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM timeseries WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()

    def run_count(self, ball_type: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM runs WHERE 1=1"
        params: list[Any] = []
        if ball_type:
            query += " AND ball_type = ?"
            params.append(ball_type)
        with self._connect() as conn:
            return int(conn.execute(query, params).fetchone()[0])
