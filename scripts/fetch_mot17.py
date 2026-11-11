"""Inspect or fetch selected MOT17 FRCNN training files from the official ZIP.

HTTP byte ranges avoid downloading the DPM, SDP, and hidden-test portions.
"""

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://motchallenge.net/data/MOT17.zip"
SCENES = {"02", "04", "05", "09", "10", "11", "13"}


class RemoteZip(io.RawIOBase):
    def __init__(self, url: str, chunk_size: int = 8 * 1024 * 1024) -> None:
        self.url = url
        self.chunk_size = chunk_size
        with urlopen(Request(url, method="HEAD"), timeout=30) as response:
            self.size = int(response.headers["Content-Length"])
            if response.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError("server does not support HTTP byte ranges")
        self.position = 0
        self.cache_start = -1
        self.cache = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        if size == 0 or self.position >= self.size:
            return b""
        result = bytearray()
        while size and self.position < self.size:
            if not self.cache_start <= self.position < self.cache_start + len(self.cache):
                end = min(self.size, self.position + max(size, self.chunk_size)) - 1
                request = Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
                with urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise RuntimeError("server ignored byte range")
                    self.cache = response.read()
                self.cache_start = self.position
            available = min(size, self.cache_start + len(self.cache) - self.position)
            result.extend(self.cache[self.position - self.cache_start:self.position - self.cache_start + available])
            self.position += available
            size -= available
        return bytes(result)


def selected_member(name: str) -> bool:
    parts = Path(name).parts
    return len(parts) >= 3 and parts[0] == "MOT17" and parts[1] == "train" and parts[2] in {
        f"MOT17-{scene}-FRCNN" for scene in SCENES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--list", action="store_true", help="inspect selected archive members without extracting")
    parser.add_argument("--scene", choices=sorted(SCENES), help="download one base scene only")
    args = parser.parse_args()
    remote = RemoteZip(URL)
    with zipfile.ZipFile(remote) as archive:
        members = [item for item in archive.infolist() if selected_member(item.filename)]
        if args.scene:
            members = [item for item in members if item.filename.startswith(f"MOT17/train/MOT17-{args.scene}-FRCNN/")]
        members.sort(key=lambda item: item.header_offset)
        print(f"selected {len(members)} entries, {sum(item.compress_size for item in members):,} compressed bytes", flush=True)
        if args.list:
            for item in members[:12]:
                print(item.filename, item.file_size)
            return
        for index, item in enumerate(members, 1):
            if item.is_dir():
                continue
            relative = Path(item.filename)
            target = args.output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == item.file_size:
                continue
            temporary = target.with_suffix(target.suffix + ".part")
            with archive.open(item) as source, temporary.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            temporary.replace(target)
            if index % 250 == 0:
                print(f"extracted {index}/{len(members)} entries", flush=True)


if __name__ == "__main__":
    main()
