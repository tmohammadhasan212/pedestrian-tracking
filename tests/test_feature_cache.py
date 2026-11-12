from pathlib import Path

from PIL import Image

from tracker.feature_cache import load_or_build
from tracker.geometry import Box
from tracker.mot import Detection, SequenceInfo


class FakeExtractor:
    def __init__(self):
        self.calls = 0

    def encode_boxes(self, image, boxes, use_head=False, normalize_output=False):
        assert not use_head and not normalize_output
        self.calls += 1
        return [tuple([box.x / 10, 1.0] + [0.0] * 574) for box in boxes]

    def project_raw(self, vectors):
        return vectors


def test_cache_reuse_and_detection_hash_invalidation(tmp_path):
    image_dir = tmp_path / "img1"
    image_dir.mkdir()
    for frame in (1, 2):
        Image.new("RGB", (40, 40), "white").save(image_dir / f"{frame:06d}.png")
    info = SequenceInfo("MOT17-02-FRCNN", 2, 40, 40, 30, image_dir, ".png")
    detections = {1: [Detection(1, Box(0, 0, 10, 20), 0.9)],
                  2: [Detection(2, Box(10, 0, 10, 20), 0.9)]}
    det_file = tmp_path / "det.txt"
    det_file.write_text("initial detector file")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    cache_file = tmp_path / "cache.npz"
    extractor = FakeExtractor()
    built = load_or_build(cache_file, info, detections, det_file, 0.3, weights, extractor)
    assert not built.reused and extractor.calls == 2
    assert built.frame(2, extractor)[0][0] == 1.0
    loaded = load_or_build(cache_file, info, detections, det_file, 0.3, weights, extractor)
    assert loaded.reused and extractor.calls == 2
    det_file.write_text("changed detector file")
    rebuilt = load_or_build(cache_file, info, detections, det_file, 0.3, weights, extractor)
    assert not rebuilt.reused and extractor.calls == 4
