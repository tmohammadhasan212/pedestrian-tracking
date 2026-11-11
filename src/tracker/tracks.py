"""IoU baseline tracker and track lifecycle."""

from dataclasses import dataclass

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

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.gate_threshold <= 0:
            raise ValueError("gate threshold must be positive")


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


def suppress_duplicates(detections: list[Detection], threshold: float) -> list[Detection]:
    """Keep the higher-confidence box when detections overlap heavily."""
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: -item.confidence):
        if all(iou(detection.box, previous.box) < threshold for previous in kept):
            kept.append(detection)
    return kept


class IoUTracker:
    def __init__(self, config: TrackerConfig = TrackerConfig()) -> None:
        self.config = config
        self.tracks: list[Track] = []
        self.next_identity = 1
        self.previous_frame = 0

    def _predict(self) -> None:
        pass

    def _match(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        return match_boxes([track.box for track in self.tracks],
                           [detection.box for detection in detections], self.config.min_iou)

    def _update(self, track: Track, detection: Detection) -> None:
        track.box = detection.box

    def _new_track(self, identity: int, detection: Detection) -> Track:
        return Track(identity, detection.box, detection.confidence, confirmed=self.config.min_hits == 1)

    def step(self, frame: int, detections: list[Detection]) -> list[Result]:
        if frame != self.previous_frame + 1:
            raise ValueError("frames must be processed consecutively, starting at 1")
        if any(detection.frame != frame for detection in detections):
            raise ValueError("detection frame does not match current frame")
        self.previous_frame = frame
        detections = suppress_duplicates(detections, self.config.duplicate_iou)
        self._predict()
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        for track_index, detection_index in matches:
            track, detection = self.tracks[track_index], detections[detection_index]
            self._update(track, detection)
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
            track = self._new_track(self.next_identity, detection)
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

    def _match(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        costs = []
        for track in self.tracks:
            assert track.kalman is not None
            row = []
            for detection in detections:
                overlap = iou(track.box, detection.box)
                gated = track.kalman.gating_distance(detection.box) > self.config.gate_threshold
                row.append(None if gated or overlap < self.config.min_iou else 1 - overlap)
            costs.append(row)
        return match_costs(costs, len(detections), 1 - self.config.min_iou)

    def _update(self, track: Track, detection: Detection) -> None:
        assert track.kalman is not None
        track.kalman.update(detection.box)
        track.box = detection.box

    def _new_track(self, identity: int, detection: Detection) -> Track:
        track = super()._new_track(identity, detection)
        track.kalman = BoxKalmanFilter(detection.box)
        return track
