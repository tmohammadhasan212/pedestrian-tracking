"""Render a short animated preview from the bundled synthetic tracking demo."""

import subprocess
import tempfile
from pathlib import Path

from tracker.demo import DemoSession


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def main() -> None:
    session = DemoSession(
        ASSETS / "demo_synthetic.mp4",
        ASSETS / "demo_synthetic_det.txt",
        ROOT / "configs" / "motion_scaled_gate25.json",
        trails=True,
    )
    output = ASSETS / "demo_output.gif"
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        count = 0
        frames = session.frames()
        try:
            for number, jpeg in enumerate(frames, 1):
                if number > 120:
                    break
                if 21 <= number <= 119 and number % 2 == 1:
                    (directory / f"{count:03d}.jpg").write_bytes(jpeg)
                    count += 1
        finally:
            frames.close()
        source = str(directory / "%03d.jpg")
        palette = directory / "palette.png"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-framerate", "7.5", "-i", source,
            "-vf", "palettegen=stats_mode=diff", str(palette),
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-framerate", "7.5", "-i", source,
            "-i", str(palette), "-filter_complex",
            "[0:v][1:v]paletteuse=dither=bayer:bayer_scale=4", "-loop", "0", str(output),
        ], check=True)
    print(f"Preview: {output} ({count} tracked frames)")


if __name__ == "__main__":
    main()
