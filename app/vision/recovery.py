"""Recovery curve fitting after compression release."""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import curve_fit

from app.core.models import RecoveryResult

logger = logging.getLogger(__name__)


def _exp_recovery(t: np.ndarray, d_final: float, amplitude: float, tau: float) -> np.ndarray:
    return d_final - amplitude * np.exp(-t / np.maximum(tau, 1e-6))


class RecoveryAnalyzer:
    """Fit exponential recovery model to post-release diameter samples."""

    def __init__(self) -> None:
        self._points: list[tuple[float, float]] = []
        self._release_time: float | None = None
        self._baseline_mm: float | None = None

    def reset(self) -> None:
        self._points.clear()
        self._release_time = None

    def set_baseline(self, baseline_mm: float) -> None:
        self._baseline_mm = baseline_mm

    def start_recovery(self, release_timestamp: float) -> None:
        self._release_time = release_timestamp
        self._points.clear()

    def add_sample(self, timestamp: float, diameter_mm: float) -> None:
        if self._release_time is None:
            return
        t = timestamp - self._release_time
        if t >= 0:
            self._points.append((t, diameter_mm))

    @property
    def point_count(self) -> int:
        return len(self._points)

    @property
    def raw_points(self) -> list[tuple[float, float]]:
        return list(self._points)

    def analyze(self) -> RecoveryResult | None:
        if len(self._points) < 5 or self._baseline_mm is None:
            return None

        t_arr = np.array([p[0] for p in self._points])
        d_arr = np.array([p[1] for p in self._points])

        # Light smoothing — preserves timing, removes single-frame spikes
        if len(d_arr) >= 5:
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            d_smooth = np.convolve(d_arr, kernel, mode="same")
            d_arr = 0.85 * d_arr + 0.15 * d_smooth

        d_final_guess = float(np.median(d_arr[-5:]))
        amplitude_guess = float(self._baseline_mm - d_arr[0])
        tau_guess = max(0.1, float(t_arr[-1] / 3))

        fit_confidence = 1.0
        fitted_curve = None

        try:
            popt, pcov = curve_fit(
                _exp_recovery,
                t_arr,
                d_arr,
                p0=[d_final_guess, amplitude_guess, tau_guess],
                bounds=(
                    [0, 0, 0.01],
                    [self._baseline_mm * 1.5, self._baseline_mm, 60.0],
                ),
                maxfev=8000,
            )
            d_final, amplitude, tau = float(popt[0]), float(popt[1]), float(popt[2])
            # Fit confidence from parameter uncertainty
            if pcov is not None and np.all(np.isfinite(pcov)):
                perr = np.sqrt(np.maximum(np.diag(pcov), 0))
                rel_err = float(np.mean(perr / (np.abs(popt) + 1e-6)))
                fit_confidence = float(np.clip(1.0 - rel_err, 0.2, 1.0))
            t_fine = np.linspace(0, max(t_arr[-1], tau * 5), 100)
            d_fine = _exp_recovery(t_fine, d_final, amplitude, tau)
            fitted_curve = (t_fine, d_fine)

            # t95: time to reach 95% of recovery
            d_start = float(d_arr[0])
            target = d_start + 0.95 * (d_final - d_start)
            t95 = self._threshold_crossing(t_arr, d_arr, target)
            if t95 is None:
                # Analytical from fit
                if abs(d_final - d_start) > 1e-6 and amplitude > 1e-6:
                    frac = (d_final - target) / amplitude
                    if 0 < frac < 1:
                        t95 = float(-tau * np.log(frac))
                    else:
                        t95 = float(t_arr[-1])
                else:
                    t95 = 0.0
        except (RuntimeError, ValueError) as exc:
            logger.warning("Recovery curve fit failed: %s — using threshold fallback", exc)
            fit_confidence = 0.3
            d_final = float(np.median(d_arr[-5:]))
            tau = float(t_arr[-1] / 3) if t_arr[-1] > 0 else 0.0
            d_start = float(d_arr[0])
            target = d_start + 0.95 * (d_final - d_start)
            t95 = self._threshold_crossing(t_arr, d_arr, target) or float(t_arr[-1])

        residual_pct = (
            (self._baseline_mm - d_final) / self._baseline_mm * 100.0
            if self._baseline_mm > 0
            else 0.0
        )

        return RecoveryResult(
            tau_s=tau,
            t95_s=t95 or 0.0,
            residual_pct=residual_pct,
            d_final_mm=d_final,
            fit_confidence=fit_confidence,
            fitted_curve=fitted_curve,
            raw_points=list(self._points),
        )

    @staticmethod
    def _threshold_crossing(
        t_arr: np.ndarray, d_arr: np.ndarray, target: float
    ) -> float | None:
        for i in range(1, len(d_arr)):
            if d_arr[i - 1] < target <= d_arr[i] or d_arr[i - 1] > target >= d_arr[i]:
                # Linear interpolate
                if abs(d_arr[i] - d_arr[i - 1]) < 1e-9:
                    return float(t_arr[i])
                frac = (target - d_arr[i - 1]) / (d_arr[i] - d_arr[i - 1])
                return float(t_arr[i - 1] + frac * (t_arr[i] - t_arr[i - 1]))
        return None
