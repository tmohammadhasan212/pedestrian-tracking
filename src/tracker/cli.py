"""Command line entry points for tracking, data audit, and official evaluation."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .mot import SequenceInfo, read_annotations, read_detections, read_sequence_info, write_results
from .render import probe_video, render_results
from .split import load_split, sequence_names
from .tracks import IoUTracker, MotionConfig, MotionTracker, TrackerConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(path: Path) -> TrackerConfig:
    values = json.loads(path.read_text())
    return MotionConfig(**values) if "gate_threshold" in values else TrackerConfig(**values)


def _track(args: argparse.Namespace) -> None:
    overall_started = time.perf_counter()
    if args.sequence_dir:
        info = read_sequence_info(args.sequence_dir)
        detection_path = args.detections or args.sequence_dir / "det" / "det.txt"
    else:
        width, height, fps, length = probe_video(args.video)
        info = SequenceInfo(args.video.stem, length, width, height, fps, args.video.parent, "")
        if not args.detections:
            raise ValueError("--detections is required with --video")
        detection_path = args.detections
    detections = read_detections(detection_path, info, args.min_confidence)
    config = _config(args.config)
    tracker = MotionTracker(config) if isinstance(config, MotionConfig) else IoUTracker(config)
    started = time.perf_counter()
    results = []
    for frame in range(1, info.length + 1):
        results.extend(tracker.step(frame, detections.get(frame, [])))
    elapsed = time.perf_counter() - started
    output = args.output_root / args.variant / "data" / f"{info.name}.txt"
    write_results(output, results)
    print(f"{info.name}: {info.length} frames, {elapsed:.3f}s tracker time, {info.length / elapsed:.2f} tracker FPS")
    print(f"MOT results: {output}")
    if args.render:
        video_output = args.output_root / args.variant / f"{info.name}.mp4"
        render_started = time.perf_counter()
        render_results(info, results, video_output, args.video)
        print(f"Annotated video: {video_output}")
        print(f"render time: {time.perf_counter() - render_started:.3f}s")
    total = time.perf_counter() - overall_started
    print(f"end-to-end time: {total:.3f}s; end-to-end FPS: {info.length / total:.2f}")


def _audit(args: argparse.Namespace) -> None:
    split = load_split(args.split)
    names = sequence_names(split, args.partition)
    for name in names:
        directory = args.data_root / "train" / name
        info = read_sequence_info(directory)
        if not info.frame_path(1).is_file() or not info.frame_path(info.length).is_file():
            raise FileNotFoundError(f"missing first or last frame in {directory}")
        detections = read_detections(directory / "det" / "det.txt", info, args.min_confidence)
        annotations = read_annotations(directory / "gt" / "gt.txt", info)
        print(f"{name}: {info.length} frames, {sum(map(len, detections.values()))} detections, "
              f"{sum(map(len, annotations.values()))} annotations")


def _evaluate(args: argparse.Namespace) -> None:
    split = load_split(args.split)
    names = sequence_names(split, args.partition)
    for variant in args.variants:
        for name in names:
            result = args.output_root / variant / "data" / f"{name}.txt"
            if not result.is_file():
                raise FileNotFoundError(result)
    seqmap = args.output_root / "seqmaps" / f"{args.partition}.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(names) + "\n")
    script = args.trackeval_root / "scripts" / "run_mot_challenge.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    command = [sys.executable, str(script), "--BENCHMARK", "MOT17", "--SPLIT_TO_EVAL", "train",
               "--GT_FOLDER", str(args.data_root / "train"), "--TRACKERS_FOLDER", str(args.output_root),
               "--OUTPUT_FOLDER", str(args.output_root / "metrics" / args.partition),
               "--SEQMAP_FILE", str(seqmap), "--SKIP_SPLIT_FOL", "True", "--TRACKER_SUB_FOLDER", "data",
               "--TRACKERS_TO_EVAL", *args.variants, "--METRICS", "HOTA", "CLEAR", "Identity",
               "--USE_PARALLEL", "False", "--PLOT_CURVES", "False"]
    print("Running official TrackEval on:", ", ".join(names), flush=True)
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pedtrack")
    subparsers = parser.add_subparsers(dest="command", required=True)
    track = subparsers.add_parser("track")
    source = track.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence-dir", type=Path)
    source.add_argument("--video", type=Path)
    track.add_argument("--detections", type=Path)
    track.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "iou.json")
    track.add_argument("--variant", default="iou")
    track.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    track.add_argument("--min-confidence", type=float, default=0.3)
    track.add_argument("--render", action="store_true")
    track.set_defaults(func=_track)
    audit = subparsers.add_parser("audit-data")
    audit.add_argument("--data-root", type=Path, required=True)
    audit.add_argument("--split", type=Path, default=PROJECT_ROOT / "configs" / "split.json")
    audit.add_argument("--partition", choices=["development", "held_out"], default="development")
    audit.add_argument("--min-confidence", type=float, default=0.3)
    audit.set_defaults(func=_audit)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--trackeval-root", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    evaluate.add_argument("--split", type=Path, default=PROJECT_ROOT / "configs" / "split.json")
    evaluate.add_argument("--partition", choices=["development", "held_out"], required=True)
    evaluate.add_argument("--variants", nargs="+", required=True)
    evaluate.set_defaults(func=_evaluate)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
