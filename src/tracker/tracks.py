"""IoU baseline tracker and track lifecycle."""

from dataclasses import dataclass
from math import sqrt

from .assignment import match_boxes, match_costs
from .geometry import Box, iou
from .kalman import BoxKalmanFilter
from .mot import Detection, Result


@dataclass(frozen=True)
class TrackerConfig:
    min_iou: float = 0.3
    min_hits: int = 2
    max_missed: int = 15
    duplicate_iou: float = 0.9

    def __post_init__(self) -> None:
        if not 0 <= self.min_iou <= 1 or not 0 <= self.duplicate_iou <= 1:
            raise ValueError("IoU thresholds must be in [0, 1]")
        if self.min_hits < 1 or self.max_missed < 0:
            raise ValueError("invalid track age settings")


@dataclass(frozen=True)
class MotionConfig(TrackerConfig):
    gate_threshold: float = 9.488  # chi-square, four measured dimensions, 95%
    gate_reference_height: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.gate_threshold <= 0:
            raise ValueError("gate threshold must be positive")
        if self.gate_reference_height is not None and self.gate_reference_height <= 0:
            raise ValueError("gate reference height must be positive")


@dataclass(frozen=True)
class AppearanceConfig(MotionConfig):
    appearance_weight: float = 0.6
    min_similarity: float = 0.3
    max_cost: float = 0.7
    feature_momentum: float = 0.8

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.appearance_weight <= 1 or not -1 <= self.min_similarity <= 1:
            raise ValueError("invalid appearance association weights")
        if not 0 <= self.max_cost <= 1 or not 0 <= self.feature_momentum < 1:
            raise ValueError("invalid appearance cost or momentum")


@dataclass
class Track:
    identity: int
    box: Box
    confidence: float
    hits: int = 1
    missed: int = 0
    age: int = 1
    confirmed: bool = False
    kalman: BoxKalmanFilter | None = None
    feature: tuple[float, ...] | None = None


def suppress_duplicate_indices(detections: list[Detection], threshold: float) -> list[int]:
    """Keep the higher-confidence box when detections overlap heavily."""
    kept: list[int] = []
    for index in sorted(range(len(detections)), key=lambda index: -detections[index].confidence):
        if all(iou(detections[index].box, detections[previous].box) < threshold for previous in kept):
            kept.append(index)
    return kept


def suppress_duplicates(detections: list[Detection], threshold: float) -> list[Detection]:
    return [detections[index] for index in suppress_duplicate_indices(detections, threshold)]


class IoUTracker:
    def __init__(self, config: TrackerConfig = TrackerConfig()) -> None:
        self.config = config
        self.tracks: list[Track] = []
        self.next_identity = 1
        self.previous_frame = 0

    def _predict(self) -> None:
        pass

    def _match(self, detections: list[Detection], features: list[tuple[float, ...] | None]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        return match_boxes([track.box for track in self.tracks],
                           [detection.box for detection in detections], self.config.min_iou)

    def _update(self, track: Track, detection: Detection, feature: tuple[float, ...] | None) -> None:
        track.box = detection.box

    def _new_track(self, identity: int, detection: Detection, feature: tuple[float, ...] | None) -> Track:
        return Track(identity, detection.box, detection.confidence, confirmed=self.config.min_hits == 1)

    def step(self, frame: int, detections: list[Detection],
             features: list[tuple[float, ...] | None] | None = None) -> list[Result]:
        if frame != self.previous_frame + 1:
            raise ValueError("frames must be processed consecutively, starting at 1")
        if any(detection.frame != frame for detection in detections):
            raise ValueError("detection frame does not match current frame")
        if features is not None and len(features) != len(detections):
            raise ValueError("one feature is required per detection")
        self.previous_frame = frame
        features = features if features is not None else [None] * len(detections)
        kept = suppress_duplicate_indices(detections, self.config.duplicate_iou)
        detections = [detections[index] for index in kept]
        features = [features[index] for index in kept]
        self._predict()
        matches, unmatched_tracks, unmatched_detections = self._match(detections, features)
        for track_index, detection_index in matches:
            track, detection = self.tracks[track_index], detections[detection_index]
            self._update(track, detection, features[detection_index])
            track.confidence = detection.confidence
            track.hits += 1
            track.age += 1
            track.missed = 0
            track.confirmed = track.hits >= self.config.min_hits
        for track_index in unmatched_tracks:
            track = self.tracks[track_index]
            track.age += 1
            track.missed += 1
        self.tracks = [track for track in self.tracks if track.missed <= self.config.max_missed]
        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            track = self._new_track(self.next_identity, detection, features[detection_index])
            self.next_identity += 1
            self.tracks.append(track)
        return [Result(frame, track.identity, track.box, track.confidence)
                for track in self.tracks if track.confirmed and track.missed == 0]


class MotionTracker(IoUTracker):
    def __init__(self, config: MotionConfig = MotionConfig()) -> None:
        super().__init__(config)
        self.config = config

    def _predict(self) -> None:
        for track in self.tracks:
            assert track.kalman is not None
            track.box = track.kalman.predict()

    def _outside_gate(self, track: Track, detection: Detection) -> bool:
        assert track.kalman is not None
        threshold = self.config.gate_threshold
        if self.config.gate_reference_height is not None:
            ratio = track.box.h / self.config.gate_reference_height
            threshold *= max(1.0, ratio * ratio)
        return track.kalman.gating_distance(detection.box) > threshold

    def _match(self, detections: list[Detection], features: list[tuple[float, ...] | None]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        costs = []
        for track in self.tracks:
            assert track.kalman is not None
            row = []
            for detection in detections:
                overlap = iou(track.box, detection.box)
                gated = self._outside_gate(track, detection)
                row.append(None if gated or overlap < self.config.min_iou else 1 - overlap)
            costs.append(row)
        return match_costs(costs, len(detections), 1 - self.config.min_iou)

    def _update(self, track: Track, detection: Detection, feature: tuple[float, ...] | None) -> None:
        assert track.kalman is not None
        track.kalman.update(detection.box)
        track.box = detection.box

    def _new_track(self, identity: int, detection: Detection, feature: tuple[float, ...] | None) -> Track:
        track = super()._new_track(identity, detection, feature)
        track.kalman = BoxKalmanFilter(detection.box)
        return track


def _normalized(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("appearance feature has zero norm")
    return tuple(value / norm for value in values)


class AppearanceTracker(MotionTracker):
    def __init__(self, config: AppearanceConfig = AppearanceConfig()) -> None:
        super().__init__(config)
        self.config = config

    def _match(self, detections: list[Detection], features: list[tuple[float, ...] | None]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        costs: list[list[float | None]] = []
        for track in self.tracks:
            assert track.kalman is not None
            row = []
            for detection, feature in zip(detections, features):
                if self._outside_gate(track, detection):
                    row.append(None)
                    continue
                overlap = iou(track.box, detection.box)
                if track.feature is not None and feature is not None:
                    if len(track.feature) != len(feature):
                        raise ValueError("appearance feature dimensions differ")
                    similarity = sum(a * b for a, b in zip(track.feature, feature))
                    if overlap < self.config.min_iou and similarity < self.config.min_similarity:
                        row.append(None)
                        continue
                    cost = ((1 - self.config.appearance_weight) * (1 - overlap) +
                            self.config.appearance_weight * (1 - similarity) / 2)
                else:
                    if overlap < self.config.min_iou:
                        row.append(None)
                        continue
                    cost = 1 - overlap
                row.append(cost if cost <= self.config.max_cost else None)
            costs.append(row)
        return match_costs(costs, len(detections), self.config.max_cost)

    def _update(self, track: Track, detection: Detection, feature: tuple[float, ...] | None) -> None:
        super()._update(track, detection, feature)
        if feature is not None:
            feature = _normalized(feature)
            if track.feature is None:
                track.feature = feature
            else:
                momentum = self.config.feature_momentum
                track.feature = _normalized(tuple(momentum * old + (1 - momentum) * new
                                                  for old, new in zip(track.feature, feature)))

    def _new_track(self, identity: int, detection: Detection, feature: tuple[float, ...] | None) -> Track:
        track = super()._new_track(identity, detection, feature)
        track.feature = _normalized(feature) if feature is not None else None
        return track
