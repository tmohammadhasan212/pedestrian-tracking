"""Verified cache of frozen CNN crop features for development ablations."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .appearance import FEATURE_DIM, FeatureExtractor
from .mot import Detection, SequenceInfo

CACHE_VERSION = 2
PREPROCESSING = "RGB crop clipped to image; resize 256x128; ImageNet normalization; raw MobileNetV3 Small pool"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class FeatureCache:
    features: np.ndarray
    valid: np.ndarray
    offsets: np.ndarray
    reused: bool

    def frame(self, frame: int, extractor: FeatureExtractor) -> list[tuple[float, ...] | None]:
        start, end = int(self.offsets[frame - 1]), int(self.offsets[frame])
        vectors = [tuple(float(value) for value in self.features[index]) if self.valid[index] else None
                   for index in range(start, end)]
        return extractor.project_raw(vectors)


def load_or_build(path: Path, info: SequenceInfo, detections: dict[int, list[Detection]],
                  detection_file: Path, min_confidence: float, weights_file: Path,
                  extractor: FeatureExtractor) -> FeatureCache:
    expected = {
        "version": CACHE_VERSION, "sequence": info.name, "frames": info.length,
        "width": info.width, "height": info.height, "preprocessing": PREPROCESSING,
        "detections_sha256": _sha256(detection_file), "min_confidence": min_confidence,
        "weights_sha256": _sha256(weights_file),
    }
    total = sum(map(len, detections.values()))
    if path.is_file():
        with np.load(path, allow_pickle=False) as loaded:
            if json.loads(str(loaded["metadata"].item())) == expected:
                features = loaded["features"]
                valid = loaded["valid"]
                offsets = loaded["offsets"]
                if (features.shape == (total, FEATURE_DIM) and valid.shape == (total,) and
                        offsets.shape == (info.length + 1,) and offsets[0] == 0 and offsets[-1] == total):
                    return FeatureCache(features, valid, offsets, reused=True)
    from PIL import Image

    features = np.zeros((total, FEATURE_DIM), dtype=np.float32)
    valid = np.zeros(total, dtype=np.bool_)
    offsets = np.zeros(info.length + 1, dtype=np.int32)
    cursor = 0
    for frame in range(1, info.length + 1):
        boxes = [detection.box for detection in detections.get(frame, [])]
        if boxes:
            with Image.open(info.frame_path(frame)) as image:
                vectors = extractor.encode_boxes(image, boxes, use_head=False, normalize_output=False)
            for vector in vectors:
                if vector is not None:
                    features[cursor] = vector
                    valid[cursor] = True
                cursor += 1
        offsets[frame] = cursor
        if frame % 100 == 0:
            print(f"cached raw features: {info.name} {frame}/{info.length}", flush=True)
    if cursor != total:
        raise ValueError("feature count differs from filtered detections")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez(handle, features=features, valid=valid, offsets=offsets, metadata=json.dumps(expected))
    temporary.replace(path)
    return FeatureCache(features, valid, offsets, reused=False)
