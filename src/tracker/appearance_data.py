"""Scene-safe identity samples for appearance-head training."""

import random
from dataclasses import dataclass
from pathlib import Path

from .geometry import Box
from .mot import read_annotations, read_sequence_info
from .split import load_split, sequence_names


@dataclass(frozen=True)
class AppearanceSample:
    sequence: str
    frame: int
    identity: int
    box: Box
    image_path: Path

    @property
    def key(self) -> tuple[str, int]:
        return self.sequence, self.identity


def development_samples(data_root: Path, split_path: Path, seed: int, max_per_identity: int = 8,
                        max_samples: int = 2048, min_visibility: float = 0.5) -> list[AppearanceSample]:
    if max_per_identity < 2 or max_samples < 2 or not 0 <= min_visibility <= 1:
        raise ValueError("invalid appearance sampling limits")
    split = load_split(split_path)
    rng = random.Random(seed)
    grouped: dict[tuple[str, int], list[AppearanceSample]] = {}
    for name in sequence_names(split, "development"):
        directory = data_root / "train" / name
        info = read_sequence_info(directory)
        for frame, annotations in read_annotations(directory / "gt" / "gt.txt", info).items():
            for annotation in annotations:
                if (annotation.confidence < 1 or annotation.category != 1 or
                        annotation.visibility < min_visibility or annotation.box.w < 8 or annotation.box.h < 16 or
                        not annotation.box.intersects_image(info.width, info.height)):
                    continue
                sample = AppearanceSample(name, frame, annotation.identity, annotation.box, info.frame_path(frame))
                grouped.setdefault(sample.key, []).append(sample)
    # Select identities across every development scene, then frames within identity.
    selected: list[AppearanceSample] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        if len(candidates) < 2:
            continue
        chosen = rng.sample(candidates, min(max_per_identity, len(candidates)))
        selected.extend(sorted(chosen, key=lambda item: item.frame))
    if len(selected) > max_samples:
        selected = rng.sample(selected, max_samples)
    selected.sort(key=lambda item: (item.sequence, item.frame, item.identity))
    if len({item.key for item in selected}) < 2:
        raise ValueError("not enough development identities for pair training")
    return selected


class PairSampler:
    """Balanced positive and same-scene negative pairs; identities are sequence-local."""

    def __init__(self, samples: list[AppearanceSample], seed: int) -> None:
        self.rng = random.Random(seed)
        self.by_identity: dict[tuple[str, int], list[int]] = {}
        for index, sample in enumerate(samples):
            self.by_identity.setdefault(sample.key, []).append(index)
        self.positive_keys = [key for key, indices in self.by_identity.items() if len(indices) >= 2]
        self.negative_keys = [key for key in self.by_identity
                              if any(other[0] == key[0] and other != key for other in self.by_identity)]
        if not self.positive_keys or not self.negative_keys:
            raise ValueError("need positive and same-scene negative identity pairs")

    def pair(self, positive: bool) -> tuple[int, int, int]:
        if positive:
            key = self.rng.choice(self.positive_keys)
            first, second = self.rng.sample(self.by_identity[key], 2)
            return first, second, 1
        key = self.rng.choice(self.negative_keys)
        other = self.rng.choice([candidate for candidate in self.by_identity
                                 if candidate[0] == key[0] and candidate != key])
        return self.rng.choice(self.by_identity[key]), self.rng.choice(self.by_identity[other]), 0
