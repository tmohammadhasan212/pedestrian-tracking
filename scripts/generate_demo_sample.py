"""Generate an original video and matching MOT detections for the browser demo."""

import argparse
import csv
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tracker.mot import SequenceInfo, read_detections
from tracker.render import probe_video


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS, FRAMES = 640, 360, 15, 150


def _frame(number: int) -> tuple[Image.Image, list[tuple[int, int, int, int, float]]]:
    image = Image.new("RGB", (WIDTH, HEIGHT), (219, 230, 227))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 174), fill=(198, 219, 226))
    draw.rectangle((0, 174, WIDTH, 198), fill=(184, 191, 181))
    draw.rectangle((0, 198, WIDTH, HEIGHT), fill=(94, 103, 110))
    for x in range(-30, WIDTH, 85):
        draw.polygon(((x, 226), (x + 42, 226), (x + 52, 248), (x - 9, 248)), fill=(220, 225, 218))
    draw.line((0, 191, WIDTH, 191), fill=(238, 239, 223), width=3)
    draw.line((0, 347, WIDTH, 347), fill=(203, 189, 121), width=3)
    for x in range(0, WIDTH, 80):
        draw.rectangle((x, 119, x + 57, 168), fill=(153, 180, 182))
        draw.rectangle((x + 6, 128, x + 48, 158), fill=(177, 204, 209))
    draw.rectangle((12, 12, 181, 42), fill=(17, 35, 45))
    draw.text((21, 20), "SYNTHETIC DEMO", fill=(243, 249, 244), font=ImageFont.load_default())

    people = ((35 + 3.05 * number, 282, 84, (244, 142, 80), 0.94),
              (565 - 2.82 * number, 315, 96, (91, 177, 215), 0.91),
              (255 + 0.83 * number, 220, 74, (170, 194, 110), 0.96))
    boxes = []
    for index, (center_x, foot_y, height, color, confidence) in sorted(
            enumerate(people), key=lambda item: item[1][1]):
        foot = foot_y + 2 * math.sin(number * 0.35 + index)
        head_r = height * 0.105
        head_y = foot - height + head_r
        torso_top, torso_bottom = foot - height * 0.75, foot - height * 0.36
        stride = 5 * math.sin(number * 0.42 + index)
        draw.ellipse((center_x - 18, foot - 7, center_x + 19, foot + 2), fill=(68, 77, 80))
        draw.line((center_x - 7, torso_bottom, center_x - 9 + stride, foot - 4), fill=(38, 56, 68), width=7)
        draw.line((center_x + 7, torso_bottom, center_x + 9 - stride, foot - 4), fill=(38, 56, 68), width=7)
        draw.line((center_x - 13, torso_top + 6, center_x - 18 - stride / 2, torso_bottom + 10), fill=color, width=6)
        draw.line((center_x + 13, torso_top + 6, center_x + 18 + stride / 2, torso_bottom + 10), fill=color, width=6)
        draw.polygon(((center_x - 12, torso_top), (center_x + 12, torso_top),
                      (center_x + 15, torso_bottom), (center_x - 15, torso_bottom)), fill=color)
        draw.ellipse((center_x - head_r, head_y - head_r, center_x + head_r, head_y + head_r),
                     fill=(214, 171, 131))
        boxes.append((round(center_x - 23), round(foot - height - 2), 46, round(height + 5), confidence))
    return image, boxes


def generate(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    video = output_dir / "demo_synthetic.mp4"
    detections = output_dir / "demo_synthetic_det.txt"
    command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-map_metadata", "-1", str(video)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert encoder.stdin is not None
        with detections.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for number in range(1, FRAMES + 1):
                image, boxes = _frame(number)
                encoder.stdin.write(image.tobytes())
                for x, y, width, height, confidence in boxes:
                    writer.writerow((number, -1, x, y, width, height, confidence))
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("FFmpeg failed to encode the demo video")
    width, height, fps, length = probe_video(video)
    if (width, height, fps, length) != (WIDTH, HEIGHT, FPS, FRAMES):
        raise RuntimeError("encoded video does not match the generated detections")
    info = SequenceInfo(video.stem, length, width, height, fps, output_dir, "")
    grouped = read_detections(detections, info, 0.3)
    if len(grouped) != FRAMES or any(len(items) != 3 for items in grouped.values()):
        raise RuntimeError("generated detections are incomplete")
    return video, detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    video, detections = generate(args.output_dir)
    print(f"Video: {video}")
    print(f"Detections: {detections}")


if __name__ == "__main__":
    main()
