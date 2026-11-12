"""Fetch only official MOT20 training metadata, public detections, and GT.

IoU and motion tracking do not read image pixels, so the benchmark JPEGs are
unnecessary for this independent fixed-detection association evaluation.
"""

import argparse
import zipfile
from pathlib import Path

from fetch_mot17 import RemoteZip


URL = "https://motchallenge.net/data/MOT20.zip"
SCENES = ("01", "02", "03", "05")
FILES = ("seqinfo.ini", "det/det.txt", "gt/gt.txt")


def selected_member(name: str) -> bool:
    parts = Path(name).parts
    if len(parts) < 4 or parts[:2] != ("MOT20", "train"):
        return False
    return parts[2] in {f"MOT20-{scene}" for scene in SCENES} and "/".join(parts[3:]) in FILES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    with zipfile.ZipFile(RemoteZip(URL)) as archive:
        members = [item for item in archive.infolist() if selected_member(item.filename)]
        members.sort(key=lambda item: item.header_offset)
        if len(members) != len(SCENES) * len(FILES):
            raise RuntimeError(f"expected 12 MOT20 evaluation files, found {len(members)}")
        print(f"selected {len(members)} files, {sum(item.compress_size for item in members):,} compressed bytes", flush=True)
        if args.list:
            for item in members:
                print(item.filename, item.file_size)
            return
        for item in members:
            target = args.output_root / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == item.file_size:
                continue
            temporary = target.with_suffix(target.suffix + ".part")
            with archive.open(item) as source, temporary.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            temporary.replace(target)
            print(target, flush=True)


if __name__ == "__main__":
    main()
