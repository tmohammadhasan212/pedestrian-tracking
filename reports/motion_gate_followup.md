# Development follow-up: box-size-aware motion gating

This follow-up was performed after the original held-out evaluation in [final_report.md](final_report.md). The previously committed threshold lock and its held-out results remain unchanged. Scenes 10 and 11 had already been inspected, so they are **not** an unbiased test for this new gate. All figures below are from the five development scenes only.

## Diagnosis and change

The original motion tracker used a fixed four-dimensional Mahalanobis gate of 9.488, while the Kalman filter represented box center and size in pixels. A replay diagnostic found that it rejected 31% of track/detection pairs with IoU at least 0.3 in scene 02, 49% in scene 05, and 53% in scene 09. Those counts include stale tracks, so they identify an overrestrictive gate as a plausible mechanism, not a causal breakdown of each ID switch. Development trials with gate thresholds 100 and 1,000,000 recovered much of the baseline performance.

The follow-up adds optional box-height scaling to the gate: `effective threshold = gate_threshold * max(1, (predicted box height / reference height)^2)`. The configured IoU threshold remains 0.3, and the original fixed gate remains the default when no reference height is set. The shared gate method applies to motion and appearance tracking. A synthetic test verifies that the scaled gate accepts a plausible shift of a large box and still rejects a distant jump. Replaying scene 02 with the original config produced a byte-identical result file, confirming that existing runs remain unchanged.

## Development results

Official TrackEval, MOT17 FRCNN public detections, confidence threshold 0.3. Percentages are shown for HOTA, IDF1, and MOTA. The two size-aware settings were run uncached, with exact detector/config/result hashes recorded in `outputs/runs/development/`. Configuration `motion_scaled_gate25.json` has SHA-256 `46212a688228b8f7c84b24bc9880f23a39bf4be030911c5db8b054dc4a7b1c5c`.

| Development variant | HOTA | IDF1 | MOTA | ID switches |
|---|---:|---:|---:|---:|
| IoU baseline | 45.32 | 51.41 | 47.57 | 799 |
| Original motion gate, 9.488 | 38.97 | 42.65 | 43.68 | 3,051 |
| Fixed gate, 100 | 45.57 | 52.31 | 47.73 | 670 |
| Effectively unbounded gate, 1,000,000 | 46.51 | 53.44 | 47.98 | 540 |
| Box-size-aware gate, reference height 50 | 46.32 | 53.20 | 47.88 | 582 |
| **Box-size-aware gate, reference height 25** | **46.48** | **53.38** | **47.97** | **543** |

The 25-pixel reference setting was selected by development HOTA among the two size-aware candidates. It improves on the IoU baseline by **1.16 HOTA points**, **1.97 IDF1 points**, and **256 fewer ID switches** across development scenes. The effectively unbounded trial scores slightly higher, but it removes nearly all practical gating. The selected size-aware gate still rejects some candidates: a replay diagnostic found 34 rejected pairs among 52,549 pairs with IoU at least 0.3. This is a very loose gate, so the measured gain is chiefly from avoiding false rejections rather than aggressive motion filtering.

| Development scene | IoU HOTA | Original motion HOTA | Size-aware HOTA | IoU switches | Size-aware switches |
|---|---:|---:|---:|---:|---:|
| 02 | 34.48 | 22.64 | 33.62 | 72 | 90 |
| 04 | 50.61 | 46.55 | 50.94 | 67 | 66 |
| 05 | 44.34 | 25.84 | 46.23 | 63 | 46 |
| 09 | 45.26 | 26.47 | 47.75 | 28 | 27 |
| 13 | 38.04 | 36.87 | 44.44 | 569 | 314 |

The new setting improves HOTA in four of five development scenes; scene 02 still regresses by 0.86 points against IoU and has 18 more switches. The aggregate motion processing rate was 141.26 FPS over 3,762 development frames on the same Apple M2 Pro CPU, excluding video rendering. This is a single run, so it is not a robust speed estimate. A fresh, independent evaluation split is required before claiming generalization or revising the original held-out conclusion.

## Supplied-video live demo

`pedtrack demo` now decodes a supplied MP4, tracks each frame as it streams, and serves annotated JPEG frames to a browser bound to `127.0.0.1`. The page shows temporary track numbers, confidence, frame count, processing FPS, and optional trails. It uses a matching MOT-style seven-column detection file; this project does not include a detector for arbitrary videos. The stream is processed on demand when **Start / restart** is clicked. It is not a pre-rendered tracking video.

To prepare a short local example from legitimately obtained MOT17 FRCNN files and launch it:

```bash
.venv/bin/python scripts/prepare_demo_video.py --sequence-dir data/MOT17/train/MOT17-05-FRCNN --frames 120
.venv/bin/pedtrack demo --video outputs/demo_assets/MOT17-05-FRCNN-first120.mp4 --detections outputs/demo_assets/MOT17-05-FRCNN-first120-det.txt --config configs/motion_scaled_gate25.json --trails
```

The command opens `http://127.0.0.1:8765/` and prints the same address. For a different supplied video, pass its path and a detection file whose frame numbers match the video. `--no-browser` leaves the URL to open manually; `--port` changes the local port; `--max-frames` limits the presentation. The local sample video and detections are under ignored `outputs/demo_assets/` and must not be committed or redistributed with the repository.

The browser stream was visually checked on the 120-frame scene-05 sample. The integration test starts a local server, verifies the HTML and health endpoint, and reads an MJPEG frame. The full suite passed **30 tests** with loopback access. The live stream reached the source clip's 14 FPS range on this motion-only sample; browser and OS scheduling may change a presentation's rate.
