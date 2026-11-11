"""Scene-level split; detector suffixes never determine split membership."""

import json
import random
from pathlib import Path

TRAIN_SCENES = ("02", "04", "05", "09", "10", "11", "13")


def create_split(seed: int, held_out_count: int = 2) -> dict:
    if not 0 < held_out_count < len(TRAIN_SCENES):
        raise ValueError("invalid held-out count")
    scenes = list(TRAIN_SCENES)
    random.Random(seed).shuffle(scenes)
    return {
        "dataset": "MOT17",
        "detector": "FRCNN",
        "seed": seed,
        "method": "shuffle sorted scene IDs with Python random.Random(seed); last two held out",
        "development_scenes": sorted(scenes[:-held_out_count]),
        "held_out_scenes": sorted(scenes[-held_out_count:]),
    }


def load_split(path: Path) -> dict:
    split = json.loads(path.read_text())
    expected = create_split(split["seed"])
    if split != expected:
        raise ValueError(f"{path} does not match the committed split algorithm")
    return split


def sequence_names(split: dict, partition: str) -> list[str]:
    if partition not in ("development", "held_out"):
        raise ValueError("partition must be development or held_out")
    return [f"MOT17-{scene}-{split['detector']}" for scene in split[f"{partition}_scenes"]]


def validate_training_scenes(metadata: dict, split: dict) -> None:
    training = metadata.get("training_scenes")
    allowed = set(sequence_names(split, "development"))
    if not isinstance(training, list) or not training or not set(training) <= allowed:
        raise ValueError("appearance checkpoint includes a non-development scene or no training scenes")
