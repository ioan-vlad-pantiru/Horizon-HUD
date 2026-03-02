from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    label: str


@dataclasses.dataclass
class Track:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    velocity_px_s: tuple[float, float]
    age: int
    time_since_update: float
    hits: int
    label: str = ""
    class_id: int = -1


@dataclasses.dataclass
class IMUReading:
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]
    mag: tuple[float, float, float]
    timestamp: float


@dataclasses.dataclass
class Orientation:
    roll: float
    pitch: float
    yaw: float
    timestamp: float


@dataclasses.dataclass
class RiskAssessment:
    track_id: int
    risk_score: float
    risk_level: str
    reasons: list[str]
    ttc_s: Optional[float] = None
