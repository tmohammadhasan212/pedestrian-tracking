# Pedestrian tracking on MOT17

This repository implements detection-to-track association and track management for **temporary pedestrian track numbers within one video**. A track number does not identify a real person. Given externally supplied detections, the tool writes MOTChallenge results and can render boxes, IDs, and confidence to MP4.

The complete methods, scene-safe split, threshold lock, per-scene metrics, qualitative cases, limitations, and citations are in [the research report](reports/final_report.md) ([PDF](reports/final_report.pdf)). Benchmark data, pretrained weights, learned checkpoint, result files, and six local example clips are excluded from Git.

## Try the live demo

The included [synthetic video](assets/demo_synthetic.mp4) and [matching detections](assets/demo_synthetic_det.txt) are original sample assets. They demonstrate the interface and ID continuity; they are not benchmark evidence. Install Python 3.12 and FFmpeg, then run:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[demo]'
.venv/bin/pedtrack demo --video assets/demo_synthetic.mp4 --detections assets/demo_synthetic_det.txt --config configs/motion_scaled_gate25.json --trails
```

Open the printed `http://127.0.0.1:8765/` address and click **Start / restart**. Stop the server with Ctrl-C. To regenerate the sample, run `.venv/bin/python scripts/generate_demo_sample.py`.

## Measured held-out result

Official TrackEval on MOT17-10-FRCNN and MOT17-11-FRCNN, with the same public detections filtered at confidence 0.3 in every variant:

| Variant | HOTA | IDF1 | MOTA | ID switches |
|---|---:|---:|---:|---:|
| IoU/Hungarian baseline | **43.62** | **47.15** | **52.44** | **402** |
| IoU + Kalman motion | 25.41 | 23.96 | 39.14 | 2,527 |
| Motion + pretrained appearance | 26.22 | 25.02 | 39.16 | 2,534 |
| Motion + development-trained head | 27.24 | 26.39 | 38.83 | 2,598 |

The learned head improves on pretrained appearance, but all motion and appearance variants regress against the IoU baseline. These are local held-out training-scene measurements, not results on the benchmark's hidden test split. Threshold choices were locked in [configs/threshold_lock.json](configs/threshold_lock.json) before the held-out runs.

## Development follow-up and live demo

A later [motion-gate follow-up](reports/motion_gate_followup.md) scales gating tolerance with the predicted pedestrian box height. On **development scenes only**, its selected setting scored 46.48 HOTA, 53.38 IDF1, and 543 switches, compared with the IoU baseline's 45.32, 51.41, and 799. The original held-out scenes had already been examined, so these new numbers are exploratory and do not replace the held-out table above.

A subsequent [fresh MOT20 external evaluation](reports/external_mot20.md) compared that frozen motion setting with IoU on four previously unused crowded scenes. The motion setting scored 10.26 versus 10.22 aggregate HOTA and 390 versus 432 switches. The gain is small, and the absolute scores are low because the unchanged 0.3 confidence cutoff retains very few public detections in two scenes. This result supports a modest switch reduction under that fixed setup, not a high-accuracy claim.

The local browser demo accepts a **supplied video file plus a matching MOT-style detection file**. It processes frames as they stream, displaying temporary IDs, confidence, FPS, and optional trails. To prepare and show a 120-frame MOT17 example from locally obtained data:

```bash
.venv/bin/python scripts/prepare_demo_video.py --sequence-dir data/MOT17/train/MOT17-05-FRCNN --frames 120
.venv/bin/pedtrack demo --video outputs/demo_assets/MOT17-05-FRCNN-first120.mp4 --detections outputs/demo_assets/MOT17-05-FRCNN-first120-det.txt --config configs/motion_scaled_gate25.json --trails
```

The MOT17 video and its detections remain under ignored `outputs/`; no benchmark footage is redistributed. For your own video, provide detections numbered to match its frames. The project does not run a detector on arbitrary video.

## Setup

The full research reproduction requires Python 3.12, FFmpeg/FFprobe, the official MOT17 training files, and official TrackEval. The synthetic live demo only needs Python and FFmpeg. Python dependencies are pinned in `pyproject.toml`. The appearance backbone weights are downloaded by TorchVision on first use; the trained head is generated locally.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test,appearance,evaluation]'
.venv/bin/python -m pytest -q
```

Place the official FRCNN scenes at `data/MOT17/train/MOT17-XX-FRCNN/` with `seqinfo.ini`, `img1/`, `det/det.txt`, and `gt/gt.txt`. [DATA.md](DATA.md) gives the exact seven scenes, deterministic split seed, download approach, and data-use notes. Put official TrackEval in `data/TrackEval/` at commit `12c8791b303e0a0b50f753af204249e622d0281a`.

To check a short sequence and produce a video:

```bash
.venv/bin/pedtrack track --sequence-dir data/MOT17/train/MOT17-02-FRCNN --config configs/iou.json --variant demo --max-frames 30 --render
```

Results appear under `outputs/demo/`. The partial result filename includes `-first30`, so it cannot be mistaken for a full evaluated scene. The command reports tracker time and end-to-end wall time; rendering is included only when requested.

## Reproduce the comparison

The fixed detector is MOT17 FRCNN. The scene split is committed in `configs/split.json`: development 02, 04, 05, 09, 13; held-out 10, 11. The selected four held-out variants use the exact hashes in `configs/threshold_lock.json`.

```bash
.venv/bin/pedtrack audit-data --data-root data/MOT17 --partition development
.venv/bin/pedtrack audit-data --data-root data/MOT17 --partition held_out
.venv/bin/python scripts/train_embedding.py --data-root data/MOT17 --max-samples 1024 --epochs 5 --pairs-per-epoch 2048 --batch-size 128 --device cpu
.venv/bin/python scripts/run_experiments.py --partition held_out --variants iou motion appearance_pretrained_weight02 appearance_finetuned --device cpu
.venv/bin/pedtrack evaluate --data-root data/MOT17 --trackeval-root data/TrackEval --partition held_out --variants iou motion appearance_pretrained_weight02 appearance_finetuned
.venv/bin/python scripts/render_examples.py
```

The experiment runner records detector, config, checkpoint, result hashes, and per-sequence runtime at `outputs/runs/<partition>/<variant>.json`. Official per-scene and aggregate metrics are at `outputs/<variant>/metrics/<partition>/pedestrian_detailed.csv`. The local six clips and matching manifest are in ignored `outputs/qualitative/` and must be regenerated from legitimately obtained benchmark footage. Development ablation details and full reproduction commands are in the final report.

## Limitations

The held-out set has only two base scenes, the trained head used one seed and 1,024 crops, and timings are single runs on an Apple M2 Pro CPU. The simple motion gate fragments tracks on these scenes. Detector misses, occlusion, camera motion, and association mistakes require case-specific inspection; the report makes only evidence-backed claims for the displayed clips. No benchmark data, clips, or external model weights are redistributed here.

## License

The original project code, documentation, and synthetic demo assets are provided under the [MIT license](LICENSE). MOTChallenge datasets, TrackEval, and model weights retain their own licenses and terms and are not included.
