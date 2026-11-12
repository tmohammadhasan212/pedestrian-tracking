"""Evaluate frozen IoU and size-aware motion settings on untouched MOT20 scenes."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from tracker.mot import read_sequence_info


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
VARIANTS = ("iou", "motion_scaled_gate25")
CONFIDENCE_THRESHOLD = 0.3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "MOT20")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "mot20_external")
    parser.add_argument("--trackeval-root", type=Path, default=ROOT / "data" / "TrackEval")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    for variant in VARIANTS:
        config = ROOT / "configs" / f"{variant}.json"
        manifest_path = args.output_root / "runs" / f"{variant}.json"
        previous = json.loads(manifest_path.read_text()) if args.resume and manifest_path.is_file() else {}
        rows = {row["sequence"]: row for row in previous.get("sequences", [])}
        for name in NAMES:
            directory = args.data_root / "train" / name
            info = read_sequence_info(directory)
            detections = directory / "det" / "det.txt"
            result = args.output_root / variant / "data" / f"{name}.txt"
            fixed = {
                "sequence": name,
                "frames": info.length,
                "detections_sha256": sha256(detections),
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "config_sha256": sha256(config),
                "code_revision": revision,
                "code_dirty": dirty,
            }
            old = rows.get(name)
            if (args.resume and old and result.is_file() and
                    all(old.get(key) == value for key, value in fixed.items()) and
                    old.get("result_sha256") == sha256(result)):
                print(f"resuming: {variant} {name} verified", flush=True)
                continue
            print(f"running: {variant} {name}", flush=True)
            command = [sys.executable, "-m", "tracker.cli", "track", "--sequence-dir", str(directory),
                       "--config", str(config), "--variant", variant, "--output-root", str(args.output_root),
                       "--min-confidence", str(CONFIDENCE_THRESHOLD), "--device", "cpu"]
            started = time.perf_counter()
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            if completed.returncode:
                print(completed.stdout, flush=True)
                print(completed.stderr, file=sys.stderr, flush=True)
                raise SystemExit(completed.returncode)
            wall_seconds = time.perf_counter() - started
            print(completed.stdout, flush=True)
            rows[name] = {**fixed, "result_sha256": sha256(result),
                          "process_wall_seconds": wall_seconds}
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({"dataset": "MOT20", "variant": variant,
                                                 "sequences": [rows[key] for key in NAMES if key in rows]}, indent=2) + "\n")
    for name in NAMES:
        signatures = set()
        for variant in VARIANTS:
            manifest = json.loads((args.output_root / "runs" / f"{variant}.json").read_text())
            row = next(item for item in manifest["sequences"] if item["sequence"] == name)
            signatures.add((row["detections_sha256"], row["confidence_threshold"]))
        if len(signatures) != 1:
            raise ValueError(f"fixed-input audit failed for {name}")
    print("fixed-input audit passed for MOT20", flush=True)

    seqmap = args.data_root / "train" / "seqmaps" / "MOT20-train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(NAMES) + "\n")
    trackeval_script = args.trackeval_root / "scripts" / "run_mot_challenge.py"
    if not trackeval_script.is_file():
        raise FileNotFoundError(trackeval_script)
    command = [sys.executable, str(ROOT / "scripts" / "run_trackeval_compat.py"), str(trackeval_script),
               "--BENCHMARK", "MOT20", "--SPLIT_TO_EVAL", "train",
               "--GT_FOLDER", str(args.data_root / "train"), "--TRACKERS_FOLDER", str(args.output_root),
               "--OUTPUT_SUB_FOLDER", "metrics/external_mot20", "--SKIP_SPLIT_FOL", "True",
               "--TRACKER_SUB_FOLDER", "data", "--TRACKERS_TO_EVAL", *VARIANTS,
               "--METRICS", "HOTA", "CLEAR", "Identity", "--USE_PARALLEL", "False",
               "--PLOT_CURVES", "False"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log_path = args.output_root / "trackeval.log"
    log_path.write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        print(completed.stdout[-5000:], flush=True)
        print(completed.stderr[-5000:], file=sys.stderr, flush=True)
        raise SystemExit(completed.returncode)
    print(f"official TrackEval complete: {log_path}", flush=True)


if __name__ == "__main__":
    main()
