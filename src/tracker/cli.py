"""Command line entry points for tracking, data audit, and official evaluation."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from .appearance import FeatureExtractor
from .mot import SequenceInfo, read_annotations, read_detections, read_sequence_info, write_results
from .render import probe_video, render_results
from .split import load_split, sequence_names, validate_training_scenes
from .tracks import AppearanceConfig, AppearanceTracker, IoUTracker, MotionConfig, MotionTracker, TrackerConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(path: Path) -> tuple[TrackerConfig, Path | None]:
    values = json.loads(path.read_text())
    checkpoint_value = values.pop("head_checkpoint", None)
    checkpoint = (PROJECT_ROOT / checkpoint_value) if checkpoint_value else None
    if "appearance_weight" in values:
        return AppearanceConfig(**values), checkpoint
    return (MotionConfig(**values) if "gate_threshold" in values else TrackerConfig(**values)), None


def _video_images(path: Path, info: SequenceInfo):
    from PIL import Image

    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                                "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = info.width * info.height * 3
    try:
        assert decoder.stdout is not None
        for frame in range(1, info.length + 1):
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                raise RuntimeError(f"video ended before frame {frame}")
            yield Image.frombytes("RGB", (info.width, info.height), raw)
    finally:
        decoder.kill()
        decoder.wait()


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
    config, checkpoint = _config(args.config)
    variant = args.variant or args.config.stem
    frame_limit = min(args.max_frames or info.length, info.length)
    if frame_limit < 1:
        raise ValueError("--max-frames must be positive")
    output_info = replace(info, name=f"{info.name}-first{frame_limit}", length=frame_limit) if frame_limit < info.length else info
    extractor = None
    feature_cache = None
    if isinstance(config, AppearanceConfig):
        extractor = FeatureExtractor(checkpoint, device=args.device)
        if extractor.metadata is not None:
            validate_training_scenes(extractor.metadata, load_split(args.split))
        if args.feature_cache_root is not None:
            if not args.sequence_dir or frame_limit != info.length:
                raise ValueError("feature cache requires a full frame-folder sequence")
            from .feature_cache import load_or_build

            weights = PROJECT_ROOT / "data" / "torch" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
            feature_cache = load_or_build(args.feature_cache_root / f"{info.name}.npz", info,
                                          detections, detection_path, args.min_confidence, weights, extractor)
            print(f"feature cache: {'reused' if feature_cache.reused else 'built'}", flush=True)
        tracker = AppearanceTracker(config)
    elif isinstance(config, MotionConfig):
        if args.feature_cache_root is not None:
            raise ValueError("feature cache applies only to appearance variants")
        tracker = MotionTracker(config)
    else:
        if args.feature_cache_root is not None:
            raise ValueError("feature cache applies only to appearance variants")
        tracker = IoUTracker(config)
    started = time.perf_counter()
    results = []
    video_images = _video_images(args.video, output_info) if extractor and args.video else None
    try:
        for frame in range(1, frame_limit + 1):
            frame_detections = detections.get(frame, [])
            features = None
            if feature_cache is not None:
                assert extractor is not None
                features = feature_cache.frame(frame, extractor)
            elif extractor and frame_detections:
                if video_images is not None:
                    image = next(video_images)
                    features = extractor.encode_boxes(image, [detection.box for detection in frame_detections])
                else:
                    features = extractor.extract_frame(info.frame_path(frame),
                                                       [detection.box for detection in frame_detections])
            elif video_images is not None:
                next(video_images)
            results.extend(tracker.step(frame, frame_detections, features))
    finally:
        if video_images is not None:
            video_images.close()
    elapsed = time.perf_counter() - started
    output = args.output_root / variant / "data" / f"{output_info.name}.txt"
    write_results(output, results)
    print(f"{output_info.name}: {frame_limit} frames, {elapsed:.3f}s tracker time, {frame_limit / elapsed:.2f} tracker FPS")
    print(f"MOT results: {output}")
    if args.render:
        video_output = args.output_root / variant / f"{output_info.name}.mp4"
        render_started = time.perf_counter()
        render_results(output_info, results, video_output, args.video)
        print(f"Annotated video: {video_output}")
        print(f"render time: {time.perf_counter() - render_started:.3f}s")
    total = time.perf_counter() - overall_started
    print(f"end-to-end time: {total:.3f}s; end-to-end FPS: {frame_limit / total:.2f}")


def _audit(args: argparse.Namespace) -> None:
    split = load_split(args.split)
    names = sequence_names(split, args.partition)
    for name in names:
        directory = args.data_root / "train" / name
        info = read_sequence_info(directory)
        missing = next((frame for frame in range(1, info.length + 1) if not info.frame_path(frame).is_file()), None)
        if missing is not None:
            raise FileNotFoundError(info.frame_path(missing))
        for frame in (1, info.length):
            size = subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                 "-of", "csv=p=0:s=x", str(info.frame_path(frame))], text=True
            ).strip()
            if size != f"{info.width}x{info.height}":
                raise ValueError(f"{info.frame_path(frame)}: dimensions {size} disagree with seqinfo.ini")
        detections = read_detections(directory / "det" / "det.txt", info, args.min_confidence)
        annotations = read_annotations(directory / "gt" / "gt.txt", info)
        print(f"{name}: {info.length} frames, {sum(map(len, detections.values()))} detections, "
              f"{sum(map(len, annotations.values()))} annotations")


def _demo(args: argparse.Namespace) -> None:
    from .demo import DemoSession, run_demo

    session = DemoSession(args.video, args.detections, args.config, args.min_confidence,
                          args.max_frames, args.device, args.trails, args.split)
    run_demo(session, args.port, not args.no_browser)


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
    # TrackEval's CLI treats optional path arguments as lists. Use its default
    # seqmap location so the official evaluation implementation stays unmodified.
    default_seqmap = args.data_root / "train" / "seqmaps" / "MOT17-train.txt"
    default_seqmap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seqmap, default_seqmap)
    script = args.trackeval_root / "scripts" / "run_mot_challenge.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_trackeval_compat.py"), str(script),
               "--BENCHMARK", "MOT17", "--SPLIT_TO_EVAL", "train",
               "--GT_FOLDER", str(args.data_root / "train"), "--TRACKERS_FOLDER", str(args.output_root),
               "--OUTPUT_SUB_FOLDER", f"metrics/{args.partition}",
               "--SKIP_SPLIT_FOL", "True", "--TRACKER_SUB_FOLDER", "data",
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
    track.add_argument("--variant")
    track.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    track.add_argument("--min-confidence", type=float, default=0.3)
    track.add_argument("--max-frames", type=int)
    track.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    track.add_argument("--feature-cache-root", type=Path)
    track.add_argument("--split", type=Path, default=PROJECT_ROOT / "configs" / "split.json")
    track.add_argument("--render", action="store_true")
    track.set_defaults(func=_track)
    demo = subparsers.add_parser("demo", help="stream a supplied video with live tracking in a local browser")
    demo.add_argument("--video", type=Path, required=True)
    demo.add_argument("--detections", type=Path, required=True)
    demo.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "motion_scaled_gate25.json")
    demo.add_argument("--min-confidence", type=float, default=0.3)
    demo.add_argument("--max-frames", type=int)
    demo.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    demo.add_argument("--split", type=Path, default=PROJECT_ROOT / "configs" / "split.json")
    demo.add_argument("--trails", action="store_true")
    demo.add_argument("--port", type=int, default=8765)
    demo.add_argument("--no-browser", action="store_true")
    demo.set_defaults(func=_demo)
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
