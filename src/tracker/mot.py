"""Strict readers and writer for MOTChallenge 10-column text files."""

import configparser
import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Iterator

from .geometry import Box


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    length: int
    width: int
    height: int
    fps: float
    image_dir: Path
    image_ext: str

    def frame_path(self, frame: int) -> Path:
        if frame < 1 or frame > self.length:
            raise ValueError(f"frame {frame} outside sequence")
        return self.image_dir / f"{frame:06d}{self.image_ext}"


@dataclass(frozen=True)
class Detection:
    frame: int
    box: Box
    confidence: float


@dataclass(frozen=True)
class Annotation:
    frame: int
    identity: int
    box: Box
    confidence: float
    category: int
    visibility: float


@dataclass(frozen=True)
class Result:
    frame: int
    identity: int
    box: Box
    confidence: float


def read_sequence_info(sequence_dir: Path) -> SequenceInfo:
    config = configparser.ConfigParser()
    path = sequence_dir / "seqinfo.ini"
    if not config.read(path):
        raise FileNotFoundError(path)
    section = config["Sequence"]
    info = SequenceInfo(
        name=section["name"],
        length=section.getint("seqLength"),
        width=section.getint("imWidth"),
        height=section.getint("imHeight"),
        fps=section.getfloat("frameRate"),
        image_dir=sequence_dir / section.get("imDir", "img1"),
        image_ext=section.get("imExt", ".jpg"),
    )
    if min(info.length, info.width, info.height) <= 0 or not isfinite(info.fps) or info.fps <= 0:
        raise ValueError(f"invalid sequence dimensions/rate in {path}")
    if info.name != sequence_dir.name:
        raise ValueError(f"sequence name mismatch in {path}")
    return info


def _rows(path: Path, columns: int) -> Iterator[tuple[int, list[str]]]:
    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) != columns:
                raise ValueError(f"{path}:{line_number}: expected {columns} columns, got {len(row)}")
            yield line_number, row


def _parse_row(path: Path, line: int, row: list[str], info: SequenceInfo,
               require_image_intersection: bool = True) -> tuple[int, int, Box, float, list[float]]:
    try:
        values = [float(value) for value in row]
        frame_float, identity_float, left, top, width, height, confidence, *rest = values
        if not all(isfinite(value) for value in values):
            raise ValueError("non-finite value")
        if not frame_float.is_integer() or not identity_float.is_integer():
            raise ValueError("frame and identity must be integers")
        frame, identity = int(frame_float), int(identity_float)
        if not 1 <= frame <= info.length:
            raise ValueError("frame outside sequence")
        box = Box.from_mot(left, top, width, height)
        if require_image_intersection and not box.intersects_image(info.width, info.height):
            raise ValueError("box does not intersect image")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence outside [0, 1]")
        return frame, identity, box, confidence, rest
    except ValueError as exc:
        raise ValueError(f"{path}:{line}: {exc}") from exc


def read_detections(path: Path, info: SequenceInfo, min_confidence: float = 0.0) -> dict[int, list[Detection]]:
    if not 0 <= min_confidence <= 1:
        raise ValueError("minimum confidence outside [0, 1]")
    grouped: dict[int, list[Detection]] = {}
    for line, row in _rows(path, 7):
        frame, _, box, confidence, _ = _parse_row(path, line, row, info)
        if confidence >= min_confidence:
            grouped.setdefault(frame, []).append(Detection(frame, box, confidence))
    return {frame: grouped[frame] for frame in sorted(grouped)}


def read_annotations(path: Path, info: SequenceInfo) -> dict[int, list[Annotation]]:
    grouped: dict[int, list[Annotation]] = {}
    for line, row in _rows(path, 9):
        frame, identity, box, confidence, rest = _parse_row(path, line, row, info, require_image_intersection=False)
        if identity < 1 or not rest[0].is_integer() or not 0 <= rest[1] <= 1:
            raise ValueError(f"{path}:{line}: invalid ground-truth identity/category/visibility")
        grouped.setdefault(frame, []).append(Annotation(frame, identity, box, confidence, int(rest[0]), rest[1]))
    return {frame: grouped[frame] for frame in sorted(grouped)}


def write_results(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = 0
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for result in results:
            if result.frame < previous or result.identity < 1 or not 0 <= result.confidence <= 1:
                raise ValueError("invalid or unordered tracking result")
            previous = result.frame
            left, top, width, height = result.box.to_mot()
            writer.writerow((result.frame, result.identity, f"{left:.3f}", f"{top:.3f}",
                             f"{width:.3f}", f"{height:.3f}", f"{result.confidence:.4f}", -1, -1, -1))
