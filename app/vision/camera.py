"""Pluggable frame source: webcam, network stream, or video file."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}


def _parse_source(source: str | int) -> tuple[str, Any]:
    """Return (source_type, capture_arg)."""
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return "webcam", int(source)
    if isinstance(source, str):
        path = Path(source)
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.exists():
            return "file", str(path)
        if source.startswith(("rtsp://", "http://", "https://")):
            return "network", source
        if path.exists():
            return "file", str(path)
    raise ValueError(f"Unsupported or missing video source: {source!r}")


class FrameSource:
    """Abstract video input — webcam, RTSP/HTTP, or looping file."""

    def __init__(self, source: str | int, *, loop_file: bool = True) -> None:
        self._source_raw = source
        self._source_type, self._capture_arg = _parse_source(source)
        self._loop_file = loop_file
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_ok = False
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._frame_index = 0
        self._wall_start: float | None = None
        self._manual_exposure = None
        self._manual_wb = None
        self._open_capture()

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height

    @property
    def fps(self) -> float:
        return self._fps

    def _open_capture(self) -> None:
        self._stop_event.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        self._stop_event.clear()

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._source_type == "webcam":
            self._cap = cv2.VideoCapture(self._capture_arg, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self._capture_arg)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self._capture_arg}")

        self._request_max_resolution()
        self._disable_auto_adjustments()

        if self._manual_exposure is not None:
            self.set_exposure(self._manual_exposure)
        if self._manual_wb is not None:
            self.set_white_balance(self._manual_wb)

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        logger.info(
            "FrameSource opened (%s): %dx%d @ %.1f fps — granted resolution",
            self._source_type,
            self._width,
            self._height,
            self._fps,
        )

        if self._source_type == "network":
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._reader_thread = threading.Thread(
                target=self._network_reader_loop, daemon=True, name="FrameSourceNetwork"
            )
            self._reader_thread.start()

    def _request_max_resolution(self) -> None:
        assert self._cap is not None
        for w, h in [(3840, 2160), (1920, 1080), (1280, 720), (640, 480)]:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            got_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            got_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if got_w >= w * 0.9 and got_h >= h * 0.9:
                break

    def _disable_auto_adjustments(self) -> None:
        assert self._cap is not None
        if self._source_type == "file":
            return
        # Auto exposure off (DSHOW: 0.25 manual, 0.75 auto)
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self._cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        # Some backends expose temperature instead
        try:
            self._cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except cv2.error:
            pass

    def set_exposure(self, value: float) -> None:
        """Set manual exposure (backend-specific; negative = shorter on many webcams)."""
        self._manual_exposure = value
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, value)

    def set_white_balance(self, temperature: float) -> None:
        """Set manual white balance temperature where supported."""
        self._manual_wb = temperature
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self._cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temperature)

    def _network_reader_loop(self) -> None:
        reconnect_delay = 1.0
        while not self._stop_event.is_set():
            cap = self._cap
            if cap is None or not cap.isOpened():
                time.sleep(reconnect_delay)
                try:
                    self._open_capture()
                except RuntimeError as exc:
                    logger.warning("Network reconnect failed: %s", exc)
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ok = True
                reconnect_delay = 1.0
            else:
                logger.warning("Network stream dropped — reconnecting...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, 10.0)
                try:
                    self._open_capture()
                except RuntimeError as exc:
                    logger.warning("Reconnect attempt failed: %s", exc)

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Return (ok, frame, timestamp_seconds).

        File sources use media timeline (frame_index / fps).
        Live sources use elapsed monotonic wall time.
        """
        if self._source_type == "network":
            ts = time.perf_counter()
            with self._lock:
                if self._latest_ok and self._latest_frame is not None:
                    return True, self._latest_frame.copy(), ts
                return False, None, ts

        assert self._cap is not None
        ok, frame = self._cap.read()

        if not ok or frame is None:
            if self._source_type == "file" and self._loop_file:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_index = 0
                ok, frame = self._cap.read()
            if not ok or frame is None:
                return False, None, self._timestamp_for_frame()

        ts = self._timestamp_for_frame()
        self._frame_index += 1
        return True, frame, ts

    def _timestamp_for_frame(self) -> float:
        if self._source_type == "file":
            return self._frame_index / max(self._fps, 1.0)
        if self._wall_start is None:
            self._wall_start = time.perf_counter()
        return time.perf_counter() - self._wall_start

    def release(self) -> None:
        self._stop_event.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
