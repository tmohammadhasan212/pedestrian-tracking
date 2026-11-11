# Pedestrian tracking on MOT17

This repository implements association and track management for temporary pedestrian track numbers within a single video. **A track number does not identify a real person.** The working code currently includes a strict MOT17 reader, scene-safe split, IoU/Hungarian baseline, constant-velocity Kalman prediction with gating, MOT result writer, annotated video renderer, and a wrapper for the official TrackEval metrics. Appearance features, trained embeddings, four-way ablations, and benchmark measurements remain future milestones. No benchmark data or pretrained model is included.

## Quick start

Requires Python 3.11+, FFmpeg and FFprobe for rendering, video input, and the frame audit. The IoU and motion tracker core has no third-party runtime dependencies. Appearance variants use pinned PyTorch, TorchVision, and Pillow packages. To install locally:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test,appearance]'
.venv/bin/python -m pytest -q
```

Place MOT17 under `data/MOT17/train/` as described in [DATA.md](DATA.md). Run one short sequence:

```bash
pedtrack track --sequence-dir data/MOT17/train/MOT17-02-FRCNN --config configs/iou.json --variant iou --render
pedtrack track --sequence-dir data/MOT17/train/MOT17-02-FRCNN --config configs/motion.json --variant motion --render
```

Train the embedding head with development scenes only, then try the two appearance variants on the first 30 frames:

```bash
python scripts/train_embedding.py --data-root data/MOT17 --max-samples 1024 --epochs 5 --pairs-per-epoch 2048
pedtrack track --sequence-dir data/MOT17/train/MOT17-02-FRCNN --config configs/appearance_pretrained.json --max-frames 30 --render
pedtrack track --sequence-dir data/MOT17/train/MOT17-02-FRCNN --config configs/appearance_finetuned.json --max-frames 30 --render
```

The pretrained model weights are downloaded into ignored `data/torch/`. The trained head and metadata are saved under ignored `outputs/appearance/`. A partial tracking run gets a `-first30` filename, which TrackEval will not mistake for a complete sequence. The fine-tuned configuration verifies that the checkpoint's training-scene list is within the committed development split.

The results go to `outputs/<variant>/data/MOT17-02-FRCNN.txt` and the optional video to `outputs/<variant>/MOT17-02-FRCNN.mp4`. The printed tracker FPS excludes input reading, writing, and rendering. End-to-end time and FPS include all of those steps, including rendering when `--render` is used. Both are wall-clock measurements and include no model inference yet. You can also use `--video clip.mp4 --detections det.txt`; this reads video dimensions and frame count with FFprobe. Detections are always supplied externally.

Run the scene audit before evaluation:

```bash
pedtrack audit-data --data-root data/MOT17 --partition development
pedtrack audit-data --data-root data/MOT17 --partition held_out
```

After generating result files for **every** scene in a partition, use the official TrackEval implementation (installed separately):

```bash
pedtrack evaluate --data-root data/MOT17 --trackeval-root /path/to/TrackEval --partition held_out --variants iou motion
```

The command creates a partition-specific seqmap and calls TrackEval with HOTA, CLEAR, and Identity metrics. It will fail if any expected result file is absent. It does not train, tune, or compute substitute metrics. The split is in `configs/split.json` and is never recalculated from available files.

## Current status

| Milestone | Status | Evidence |
|---|---|---|
| MOT17 adapter, split, IoU baseline | Implemented | `tests/test_data.py`, `tests/test_tracker.py` |
| Kalman prediction and gating | Implemented | `tests/test_kalman.py` |
| Short synthetic sequence to results and MP4 | Implemented | `tests/test_cli.py` |
| CNN features and embedding training | Implemented | `tests/test_appearance.py`; trained checkpoint stored locally |
| Motion plus appearance fusion | Implemented | `tests/test_appearance_tracker.py` |
| Four held-out ablations and TrackEval report | Pending | All seven FRCNN training scenes downloaded and audited; appearance variants and benchmark runs pending |
| Qualitative success/failure clips and final reports | Pending | Requires experiments |

Current output is a working pipeline, not a completed research result. The report table will be populated only after fixed-input experiments on the held-out scenes. No tracking improvement is claimed yet.
