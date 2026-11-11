"""Run fixed-input MOT17 variants and record reproducible per-sequence provenance."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from tracker.mot import read_sequence_info
from tracker.split import load_split, sequence_names

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("iou", "motion", "appearance_pretrained", "appearance_finetuned")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT17")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--partition", choices=["development", "held_out"], required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    split = load_split(ROOT / "configs" / "split.json")
    names = sequence_names(split, args.partition)
    code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    code_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    for variant in args.variants:
        config = ROOT / "configs" / f"{variant}.json"
        config_values = json.loads(config.read_text())
        checkpoint = ROOT / config_values["head_checkpoint"] if config_values.get("head_checkpoint") else None
        checkpoint_hash = sha256(checkpoint) if checkpoint else None
        manifest_path = args.output_root / "runs" / args.partition / f"{variant}.json"
        old = json.loads(manifest_path.read_text()) if args.resume and manifest_path.is_file() else {}
        rows = {row["sequence"]: row for row in old.get("sequences", [])}
        for name in names:
            directory = args.data_root / "train" / name
            info = read_sequence_info(directory)
            detections = directory / "det" / "det.txt"
            result = args.output_root / variant / "data" / f"{name}.txt"
            fixed = {
                "sequence": name, "frames": info.length, "detector": split["detector"],
                "detections_sha256": sha256(detections), "confidence_threshold": args.min_confidence,
                "config_sha256": sha256(config), "checkpoint_sha256": checkpoint_hash,
                "code_revision": code_revision, "code_dirty": code_dirty,
            }
            previous = rows.get(name)
            if (args.resume and previous and result.is_file() and
                    all(previous.get(key) == value for key, value in fixed.items()) and
                    previous.get("result_sha256") == sha256(result)):
                print(f"resuming: {variant} {name} already verified", flush=True)
                continue
            print(f"running: {variant} {name}", flush=True)
            command = [sys.executable, "-m", "tracker.cli", "track", "--sequence-dir", str(directory),
                       "--config", str(config), "--variant", variant,
                       "--output-root", str(args.output_root), "--min-confidence", str(args.min_confidence),
                       "--device", args.device]
            started = time.perf_counter()
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            if completed.returncode:
                print(completed.stdout, flush=True)
                print(completed.stderr, file=sys.stderr, flush=True)
                raise SystemExit(completed.returncode)
            wall_seconds = time.perf_counter() - started
            print(completed.stdout, flush=True)
            match = re.search(r"([0-9.]+)s tracker time", completed.stdout)
            rows[name] = {
                **fixed, "result_sha256": sha256(result), "tracker_seconds": float(match.group(1)) if match else None,
                "process_wall_seconds": wall_seconds,
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({"partition": args.partition, "variant": variant,
                                                 "sequences": [rows[key] for key in names if key in rows]}, indent=2) + "\n")
    # The same sequence must have the same detector file and confidence cut in every variant.
    for name in names:
        signatures = set()
        for variant in args.variants:
            manifest = json.loads((args.output_root / "runs" / args.partition / f"{variant}.json").read_text())
            row = next(item for item in manifest["sequences"] if item["sequence"] == name)
            signatures.add((row["detections_sha256"], row["confidence_threshold"]))
        if len(signatures) != 1:
            raise ValueError(f"detector inputs differ across variants for {name}")
    print(f"fixed-input audit passed for {args.partition}", flush=True)


if __name__ == "__main__":
    main()
