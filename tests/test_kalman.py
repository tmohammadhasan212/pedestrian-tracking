import pytest

from tracker.geometry import Box
from tracker.kalman import BoxKalmanFilter
from tracker.mot import Detection
from tracker.tracks import MotionConfig, MotionTracker


def test_kalman_predict_update_follows_synthetic_motion():
    filter_ = BoxKalmanFilter(Box(0, 0, 10, 20))
    filter_.predict()
    filter_.update(Box(4, 0, 10, 20))
    predicted = filter_.predict()
    assert predicted.x > 3
    assert predicted.w == pytest.approx(10)
    assert filter_.gating_distance(Box(1000, 0, 10, 20)) > 9.488
    assert filter_.gating_distance(predicted) == pytest.approx(0)
    assert all(filter_.covariance[i][i] > 0 for i in range(8))


def test_motion_tracker_preserves_id_after_short_gap():
    tracker = MotionTracker(MotionConfig(min_hits=1, max_missed=2, min_iou=0.2))
    def detection(frame, x):
        return Detection(frame, Box(x, 0, 10, 20), 0.9)
    assert tracker.step(1, [detection(1, 0)])[0].identity == 1
    assert tracker.step(2, [detection(2, 4)])[0].identity == 1
    assert tracker.step(3, []) == []
    assert tracker.step(4, [detection(4, 12)])[0].identity == 1


def test_motion_gating_rejects_implausible_jump():
    tracker = MotionTracker(MotionConfig(min_hits=1, min_iou=0, gate_threshold=0.1))
    tracker.step(1, [Detection(1, Box(0, 0, 10, 20), 0.9)])
    results = tracker.step(2, [Detection(2, Box(100, 0, 10, 20), 0.9)])
    assert [result.identity for result in results] == [2]


def test_height_scaled_gate_accepts_plausible_large_box_motion():
    first = Detection(1, Box(0, 0, 80, 200), 0.9)
    shifted = Detection(2, Box(40, 0, 80, 200), 0.9)
    fixed = MotionTracker(MotionConfig(min_hits=1))
    scaled = MotionTracker(MotionConfig(min_hits=1, gate_reference_height=50))
    for tracker in (fixed, scaled):
        tracker.step(1, [first])
    assert [result.identity for result in fixed.step(2, [shifted])] == [2]
    assert [result.identity for result in scaled.step(2, [shifted])] == [1]
    far = Detection(3, Box(1000, 0, 80, 200), 0.9)
    assert [result.identity for result in scaled.step(3, [far])] == [2]


def test_gate_reference_height_must_be_positive():
    with pytest.raises(ValueError, match="reference height"):
        MotionConfig(gate_reference_height=0)
