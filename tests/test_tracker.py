import pytest

from tracker.assignment import hungarian, match_boxes
from tracker.geometry import Box
from tracker.mot import Detection
from tracker.tracks import IoUTracker, TrackerConfig


def det(frame, x, confidence=0.9):
    return Detection(frame, Box(x, 0, 10, 10), confidence)


def test_hungarian_chooses_global_optimum():
    assert hungarian([[1, 2], [1.1, 100]]) == [1, 0]
    assert hungarian([]) == []


def test_assignment_with_unmatched_tracks():
    matches, missing_tracks, new_detections = match_boxes(
        [Box(0, 0, 10, 10), Box(100, 0, 10, 10)], [Box(1, 0, 10, 10), Box(200, 0, 10, 10)], 0.3
    )
    assert matches == [(0, 0)]
    assert missing_tracks == [1]
    assert new_detections == [1]


def test_track_confirmation_reactivation_aging_and_unique_ids():
    tracker = IoUTracker(TrackerConfig(min_hits=2, max_missed=1))
    assert tracker.step(1, [det(1, 0)]) == []
    assert [r.identity for r in tracker.step(2, [det(2, 1)])] == [1]
    assert tracker.step(3, []) == []
    assert [r.identity for r in tracker.step(4, [det(4, 2)])] == [1]
    assert tracker.step(5, []) == []
    assert tracker.step(6, []) == []
    assert tracker.step(7, [det(7, 2)]) == []
    assert tracker.tracks[0].identity == 2
    assert tracker.step(8, [det(8, 3)])[0].identity == 2


def test_duplicate_suppression_and_invalid_frame():
    tracker = IoUTracker(TrackerConfig(min_hits=1))
    assert len(tracker.step(1, [det(1, 0, 0.9), det(1, 0.2, 0.7)])) == 1
    with pytest.raises(ValueError, match="consecutively"):
        tracker.step(3, [])
