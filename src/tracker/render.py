"""Stream annotated frames through ffmpeg without loading a video into memory."""

import json
import subprocess
from collections import defaultdict
from pathlib import Path

from .mot import Result, SequenceInfo


_FONT = {
    "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
    ".": ("000", "000", "000", "000", "010"), "#": ("101", "111", "101", "111", "101"),
    " ": ("000", "000", "000", "000", "000"),
}


def probe_video(path: Path) -> tuple[int, int, float, int]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
               "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)]
    stream = json.loads(subprocess.check_output(command, text=True))["streams"][0]
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    return int(stream["width"]), int(stream["height"]), numerator / denominator, int(stream["nb_read_frames"])


def _pixel(buffer: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 3
        buffer[offset:offset + 3] = bytes(color)


def _annotate(buffer: bytearray, width: int, height: int, results: list[Result]) -> None:
    for result in results:
        x0, y0, w, h = result.box.x, result.box.y, result.box.w, result.box.h
        left, top = round(x0), round(y0)
        right, bottom = round(x0 + w) - 1, round(y0 + h) - 1
        color = ((result.identity * 73) % 180 + 75, (result.identity * 137) % 180 + 75,
                 (result.identity * 199) % 180 + 75)
        for x in range(max(0, left), min(width, right + 1)):
            for offset in (0, 1):
                _pixel(buffer, width, height, x, top + offset, color)
                _pixel(buffer, width, height, x, bottom - offset, color)
        for y in range(max(0, top), min(height, bottom + 1)):
            for offset in (0, 1):
                _pixel(buffer, width, height, left + offset, y, color)
                _pixel(buffer, width, height, right - offset, y, color)
        label = f"#{result.identity} {result.confidence:.2f}"
        label_y = max(0, top - 9)
        for index, char in enumerate(label):
            for dy, row in enumerate(_FONT[char]):
                for dx, bit in enumerate(row):
                    if bit == "1":
                        _pixel(buffer, width, height, left + 4 * index + dx, label_y + dy, color)


def render_results(info: SequenceInfo, results: list[Result], output: Path, video: Path | None = None) -> None:
    source = (["-i", str(video)] if video else
              ["-framerate", str(info.fps), "-start_number", "1", "-i",
               str(info.image_dir / f"%06d{info.image_ext}")])
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", *source, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                                "-video_size", f"{info.width}x{info.height}", "-framerate", str(info.fps),
                                "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)],
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    by_frame: dict[int, list[Result]] = defaultdict(list)
    for result in results:
        by_frame[result.frame].append(result)
    frame_bytes = info.width * info.height * 3
    try:
        assert decoder.stdout is not None and encoder.stdin is not None
        for frame in range(1, info.length + 1):
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                raise RuntimeError(f"video ended before frame {frame}")
            pixels = bytearray(raw)
            _annotate(pixels, info.width, info.height, by_frame[frame])
            encoder.stdin.write(pixels)
        encoder.stdin.close()
        decoder.stdout.close()
        decoder_error = decoder.stderr.read().decode() if decoder.stderr else ""
        encoder_error = encoder.stderr.read().decode() if encoder.stderr else ""
        if decoder.wait() or encoder.wait():
            raise RuntimeError(f"ffmpeg failed: {decoder_error} {encoder_error}")
    except BaseException:
        decoder.kill()
        encoder.kill()
        decoder.wait()
        encoder.wait()
        output.unlink(missing_ok=True)
        raise
