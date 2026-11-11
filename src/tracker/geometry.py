"""Bounding boxes use zero-based, half-open pixel coordinates internally."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if not all(isfinite(v) for v in (self.x, self.y, self.w, self.h)):
            raise ValueError("box coordinates must be finite")
        if self.w <= 0 or self.h <= 0:
            raise ValueError("box width and height must be positive")

    @classmethod
    def from_mot(cls, left: float, top: float, width: float, height: float) -> "Box":
        if left < 1 or top < 1:
            raise ValueError("MOT coordinates are 1-based")
        return cls(left - 1, top - 1, width, height)

    def to_mot(self) -> tuple[float, float, float, float]:
        return self.x + 1, self.y + 1, self.w, self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def within(self, width: int, height: int) -> bool:
        return self.x >= 0 and self.y >= 0 and self.x + self.w <= width and self.y + self.h <= height


def iou(a: Box, b: Box) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.w, b.x + b.w)
    bottom = min(a.y + a.h, b.y + b.h)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = a.w * a.h + b.w * b.h - intersection
    return intersection / union
