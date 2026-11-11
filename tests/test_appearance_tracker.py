from tracker.geometry import Box
from tracker.mot import Detection
from tracker.tracks import AppearanceConfig, AppearanceTracker


def detection(frame, x, confidence=0.9):
    return Detection(frame, Box(x, 0, 10, 20), confidence)


def test_appearance_prevents_position_based_identity_swap():
    config = AppearanceConfig(min_hits=1, min_iou=0.05, appearance_weight=0.8, max_cost=0.9)
    tracker = AppearanceTracker(config)
    tracker.step(1, [detection(1, 0), detection(1, 8)], [(1.0, 0.0), (0.0, 1.0)])
    results = tracker.step(2, [detection(2, 1), detection(2, 9)], [(0.0, 1.0), (1.0, 0.0)])
    assert {result.identity: result.box.x for result in results} == {1: 9, 2: 1}


def test_missing_crop_falls_back_to_iou_and_duplicate_feature_stays_aligned():
    tracker = AppearanceTracker(AppearanceConfig(min_hits=1))
    tracker.step(1, [detection(1, 0, 0.9), detection(1, 0.1, 0.7)],
                 [(1.0, 0.0), (0.0, 1.0)])
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].feature == (1.0, 0.0)
    results = tracker.step(2, [detection(2, 1)], [None])
    assert [result.identity for result in results] == [1]
    assert tracker.tracks[0].feature == (1.0, 0.0)


def test_motion_gate_rejects_similar_appearance_at_impossible_distance():
    tracker = AppearanceTracker(AppearanceConfig(min_hits=1, min_iou=0, gate_threshold=0.1))
    tracker.step(1, [detection(1, 0)], [(1.0, 0.0)])
    results = tracker.step(2, [detection(2, 100)], [(1.0, 0.0)])
    assert [result.identity for result in results] == [2]


def test_appearance_track_keeps_identity_across_short_occlusion():
    tracker = AppearanceTracker(AppearanceConfig(min_hits=1, max_missed=2, min_iou=0.05))
    assert tracker.step(1, [detection(1, 0)], [(1.0, 0.0)])[0].identity == 1
    assert tracker.step(2, [detection(2, 4)], [(1.0, 0.0)])[0].identity == 1
    assert tracker.step(3, [], []) == []
    assert tracker.step(4, [detection(4, 12)], [(1.0, 0.0)])[0].identity == 1
