"""Render local, ground-truth-referenced qualitative MOT17 clips.

The generated clips contain benchmark imagery and stay under ignored outputs/.
They are evidence for the report, not files to redistribute with the repository.
"""

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

from tracker.geometry import Box, iou
from tracker.mot import read_annotations, read_sequence_info


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    ("preserved_01", "10", 4, 350, 380, "preserved"),
    ("preserved_02", "10", 2, 510, 540, "preserved"),
    ("preserved_03", "11", 26, 170, 200, "preserved"),
    ("switch_01", "10", 4, 555, 572, "switch"),
    ("switch_02", "11", 5, 97, 112, "switch"),
    ("switch_03", "11", 31, 397, 412, "switch"),
)


def read_results(path: Path) -> dict[int, list[tuple[int, Box]]]:
    grouped: dict[int, list[tuple[int, Box]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            grouped[int(row[0])].append((int(row[1]), Box.from_mot(*map(float, row[2:6]))))
    return grouped


def frame_matches(annotations, results) -> dict[int, int]:
    visible = [item for item in annotations if item.category == 1 and item.confidence == 1 and item.visibility >= 0.5]
    if not visible or not results:
        return {}
    overlaps = np.asarray([[iou(item.box, result[1]) for result in results] for item in visible])
    rows, columns = linear_sum_assignment(-overlaps)
    return {visible[row].identity: results[column][0] for row, column in zip(rows, columns)
            if overlaps[row, column] >= 0.5}


def draw_box(draw: ImageDraw.ImageDraw, box: Box, crop_left: int, crop_top: int,
             color: str, width: int = 4) -> None:
    scale = 640 / 400
    x0, y0 = (box.x - crop_left) * scale, (box.y - crop_top) * scale
    x1, y1 = (box.x + box.w - crop_left) * scale, (box.y + box.h - crop_top) * scale
    draw.rectangle((x0, y0, x1, y1), outline=color, width=width)


def render_example(output: Path, data_root: Path, result_root: Path,
                   key: str, scene: str, gt_id: int, start: int, end: int, kind: str) -> dict:
    name = f"MOT17-{scene}-FRCNN"
    sequence_dir = data_root / "train" / name
    info = read_sequence_info(sequence_dir)
    annotations = read_annotations(sequence_dir / "gt" / "gt.txt", info)
    results = read_results(result_root / f"{name}.txt")
    target = {frame: next(item for item in annotations[frame] if item.identity == gt_id)
              for frame in range(start, end + 1)}
    center_x = round(sum(item.box.x + item.box.w / 2 for item in target.values()) / len(target))
    center_y = round(sum(item.box.y + item.box.h / 2 for item in target.values()) / len(target))
    crop_left = max(0, min(info.width - 400, center_x - 200))
    crop_top = max(0, min(info.height - 400, center_y - 200))
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", "640x640", "-framerate", str(info.fps), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assignments = []
    try:
        assert process.stdin is not None
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 19)
        except OSError:
            font = ImageFont.load_default()
        for frame in range(start, end + 1):
            tracks = results.get(frame, [])
            track_id = frame_matches(annotations.get(frame, []), tracks).get(gt_id)
            assignments.append({"frame": frame, "track_id": track_id})
            with Image.open(info.frame_path(frame)) as source:
                image = source.convert("RGB").crop((crop_left, crop_top, crop_left + 400, crop_top + 400))
            image = image.resize((640, 640))
            draw = ImageDraw.Draw(image)
            draw_box(draw, target[frame].box, crop_left, crop_top, "#ff4545")
            if track_id is not None:
                matched = next(box for identity, box in tracks if identity == track_id)
                draw_box(draw, matched, crop_left, crop_top, "#42f58a", 3)
            draw.rectangle((0, 0, 640, 56), fill="#111111")
            draw.text((12, 5), f"{name}  frame {frame}  GT {gt_id}", fill="white", font=font)
            draw.text((12, 29), f"red: ground truth   green: track {track_id if track_id is not None else 'none'}",
                      fill="white", font=font)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode() if process.stderr else ""
        if process.wait():
            raise RuntimeError(f"ffmpeg failed: {error}")
    except BaseException:
        process.kill()
        process.wait()
        output.unlink(missing_ok=True)
        raise
    seen = sorted({item["track_id"] for item in assignments if item["track_id"] is not None})
    if kind == "preserved" and len(seen) != 1:
        raise ValueError(f"{key}: expected one track ID, saw {seen}")
    if kind == "switch" and len(seen) < 2:
        raise ValueError(f"{key}: expected a changed track ID, saw {seen}")
    return {"key": key, "kind": kind, "sequence": name, "gt_identity": gt_id,
            "start_frame": start, "end_frame": end, "track_ids": seen,
            "matching_rule": "one-to-one maximum IoU >= 0.5 on visible pedestrian ground truth",
            "assignments": assignments, "clip": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT17")
    parser.add_argument("--variant", default="appearance_finetuned")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "qualitative")
    args = parser.parse_args()
    result_root = ROOT / "outputs" / args.variant / "data"
    records = [render_example(args.output_dir / f"{key}.mp4", args.data_root, result_root,
                              key, scene, gt_id, start, end, kind)
               for key, scene, gt_id, start, end, kind in EXAMPLES]
    (args.output_dir / "manifest.json").write_text(json.dumps({"variant": args.variant,
                                                                "examples": records}, indent=2) + "\n")
    for record in records:
        print(record["key"], record["sequence"], record["start_frame"], record["end_frame"], record["track_ids"])


if __name__ == "__main__":
    main()
