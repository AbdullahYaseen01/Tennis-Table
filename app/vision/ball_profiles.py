from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TENNIS_DIAMETER_MM = 67.0
PICKLEBALL_DIAMETER_MM = 74.0
GOLF_DIAMETER_MM = 42.7


@dataclass(frozen=True)
class BallProfile:
    name: str
    diameter_mm: float
    hsv_lower: np.ndarray
    hsv_upper: np.ndarray
    use_lab: bool = True
    lab_a_min: int = 110
    lab_a_max: int = 255
    lab_b_min: int = 0
    lab_b_max: int = 255
    use_convex_hull: bool = False
    min_circularity: float = 0.35
    roundness_min: float = 0.82
    aspect_min: float = 0.88
    aspect_max: float = 1.12
    loose_sat_drop: int = 30
    loose_val_drop: int = 25
    fallback_type: str | None = None
    floor_contact_y_frac: float = 0.73


def _hsv(h0: int, s0: int, v0: int, h1: int, s1: int, v1: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([h0, s0, v0], dtype=np.uint8),
        np.array([h1, s1, v1], dtype=np.uint8),
    )


PROFILES: dict[str, BallProfile] = {}


def _register(key: str, profile: BallProfile) -> None:
    PROFILES[key] = profile


_lo, _hi = _hsv(18, 45, 70, 48, 255, 255)
_register(
    "tennis",
    BallProfile(
        name="tennis",
        diameter_mm=TENNIS_DIAMETER_MM,
        hsv_lower=_lo,
        hsv_upper=_hi,
        use_lab=True,
        lab_a_min=110,
        fallback_type="golf",
    ),
)

_lo, _hi = _hsv(22, 70, 110, 55, 255, 255)
_register(
    "pickleball",
    BallProfile(
        name="pickleball",
        diameter_mm=PICKLEBALL_DIAMETER_MM,
        hsv_lower=_lo,
        hsv_upper=_hi,
        use_lab=True,
        lab_a_min=95,
        lab_b_min=120,
        lab_b_max=255,
        use_convex_hull=True,
        min_circularity=0.26,
        roundness_min=0.68,
        aspect_min=0.82,
        aspect_max=1.18,
        loose_sat_drop=35,
        loose_val_drop=30,
        fallback_type="tennis",
        floor_contact_y_frac=0.60,
    ),
)

_lo, _hi = _hsv(18, 90, 140, 42, 255, 255)
_register(
    "pickleball_indoor",
    BallProfile(
        name="pickleball_indoor",
        diameter_mm=PICKLEBALL_DIAMETER_MM,
        hsv_lower=_lo,
        hsv_upper=_hi,
        use_lab=True,
        lab_a_min=100,
        lab_b_min=130,
        use_convex_hull=True,
        min_circularity=0.26,
        roundness_min=0.68,
        aspect_min=0.82,
        aspect_max=1.18,
        fallback_type="pickleball",
        floor_contact_y_frac=0.60,
    ),
)

_lo, _hi = _hsv(0, 0, 140, 180, 90, 255)
_register(
    "golf",
    BallProfile(
        name="golf",
        diameter_mm=GOLF_DIAMETER_MM,
        hsv_lower=_lo,
        hsv_upper=_hi,
        use_lab=False,
        roundness_min=0.68,
        aspect_min=0.65,
        aspect_max=1.35,
    ),
)

_register(
    "white",
    BallProfile(
        name="white",
        diameter_mm=TENNIS_DIAMETER_MM,
        hsv_lower=_lo,
        hsv_upper=_hi,
        use_lab=False,
        roundness_min=0.68,
        aspect_min=0.65,
        aspect_max=1.35,
    ),
)


def get_profile(ball_type: str) -> BallProfile:
    key = ball_type.lower().strip()
    if key in PROFILES:
        return PROFILES[key]
    if key.startswith("pickle"):
        return PROFILES["pickleball"]
    return PROFILES["tennis"]


def reference_diameter_mm(ball_type: str) -> float:
    return get_profile(ball_type).diameter_mm


def roundness_min(ball_type: str) -> float:
    return get_profile(ball_type).roundness_min


def aspect_bounds(ball_type: str) -> tuple[float, float]:
    p = get_profile(ball_type)
    return p.aspect_min, p.aspect_max


def floor_contact_y_frac(ball_type: str) -> float:
    return get_profile(ball_type).floor_contact_y_frac


def to_legacy_color_dict(profile: BallProfile) -> dict:
    return {
        "hsv_lower": tuple(int(x) for x in profile.hsv_lower),
        "hsv_upper": tuple(int(x) for x in profile.hsv_upper),
    }
