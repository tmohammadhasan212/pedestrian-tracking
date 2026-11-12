"""Make a local MOT17 video and matching detections for the browser demo."""

import argparse
import csv
import subprocess
from pathlib import Path

from tracker.mot import SequenceInfo, read_detections, read_sequence_info
from tracker.render import probe_video


ROOT = Path(__file__).resolve().parents[1]


def prepare(sequence_dir: Path, frames: int, output_dir: Path) -> tuple[Path, Path]:
    info = read_sequence_info(sequence_dir)
    if not 1 <= frames <= info.length:
        raise ValueError(f"frames must be between 1 and {info.length}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{info.name}-first{frames}"
    video = output_dir / f"{stem}.mp4"
    detections = output_dir / f"{stem}-det.txt"
    source = info.image_dir / f"%06d{info.image_ext}"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(info.fps),
                    "-start_number", "1", "-i", str(source), "-frames:v", str(frames),
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    with (sequence_dir / "det" / "det.txt").open(newline="") as source_file, detections.open("w", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        for row in csv.reader(source_file):
            if row and int(float(row[0])) <= frames:
                writer.writerow(row)
    width, height, fps, length = probe_video(video)
    if length != frames or (width, height) != (info.width, info.height):
        raise RuntimeError("encoded video disagrees with requested frames or source dimensions")
    video_info = SequenceInfo(stem, length, width, height, fps, output_dir, "")
    read_detections(detections, video_info, 0.3)
    return video, detections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "demo_assets")
    args = parser.parse_args()
    video, detections = prepare(args.sequence_dir, args.frames, args.output_dir)
    print(f"Video: {video}")
    print(f"Detections: {detections}")
    print(f"Launch: .venv/bin/pedtrack demo --video {video} --detections {detections} --trails")


if __name__ == "__main__":
    main()
