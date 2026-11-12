# Fresh external evaluation: MOT20 public detections

This evaluation tests the MOT17-development-selected `motion_scaled_gate25` setting against the IoU baseline on four previously unused, labeled MOT20 training scenes. The tracker configurations were frozen before the first MOT20 tracking result was generated. No MOT20 identity annotation was used for training or threshold selection. This is an external-domain test, not a MOT17 hidden-test submission.

## Protocol and provenance

The source is the [official MOT20 archive](https://motchallenge.net/data/MOT20/). `scripts/fetch_mot20_eval.py` extracted only `seqinfo.ini`, public `det/det.txt`, and `gt/gt.txt` for scenes 01, 02, 03, and 05. No image pixels were downloaded or used; both compared trackers are detection-only association systems. The original MOT20 detection files contain ten columns. The reader now accepts their first seven MOT fields and validates the remaining numeric fields without changing any box or confidence value.

The detector confidence cutoff stayed at **0.3**, exactly as in the MOT17 evaluation, and both variants used the same original public-detection file in each scene. The fixed-input hash audit passed. The locked IoU config SHA-256 was `e2c5cfaa695a75f3adfe62638754d46d15e6d07625319f2a592f0d16396d133c`; the selected size-aware motion config SHA-256 was `46212a688228b8f7c84b24bc9880f23a39bf4be030911c5db8b054dc4a7b1c5c`. Both runs recorded clean code revision `9aeeb3cdc23d5a77353ebf9fdabaa82b0c79a3c1`. Per-sequence detector, result, and config hashes and wall times are in `outputs/mot20_external/runs/`. The official TrackEval implementation at commit `12c8791b303e0a0b50f753af204249e622d0281a` computed the metrics without formula changes.

The cutoff retained far fewer detections in MOT20's last two scenes. This was established by a detection-file audit; the threshold was not changed after reading evaluation results.

| Scene | Frames | Raw public detections | Retained at 0.3 |
|---|---:|---:|---:|
| MOT20-01 | 429 | 12,610 | 4,367 |
| MOT20-02 | 2,782 | 89,837 | 37,515 |
| MOT20-03 | 2,405 | 177,347 | 2,044 |
| MOT20-05 | 3,315 | 381,349 | 3,770 |

## Official TrackEval results

HOTA, DetA, IDF1, and MOTA are percentages; ID switches are counts. The combined row is TrackEval's aggregate, not an arithmetic average of scene percentages.

| Variant | Scene | HOTA | DetA | IDF1 | MOTA | ID switches |
|---|---|---:|---:|---:|---:|---:|
| IoU | 01 | 28.54 | 18.49 | 27.28 | 19.08 | 38 |
| Size-aware motion | 01 | 28.59 | 18.54 | 27.37 | 19.13 | 30 |
| IoU | 02 | 25.30 | 20.38 | 26.74 | 21.07 | 288 |
| Size-aware motion | 02 | 25.42 | 20.44 | 27.16 | 21.09 | 256 |
| IoU | 03 | 1.88 | 0.44 | 0.71 | 0.46 | 43 |
| Size-aware motion | 03 | 1.88 | 0.44 | 0.71 | 0.46 | 43 |
| IoU | 05 | 1.65 | 0.39 | 0.66 | 0.40 | 63 |
| Size-aware motion | 05 | 1.66 | 0.39 | 0.67 | 0.40 | 61 |
| **IoU combined** | **all four** | **10.22** | **3.46** | **5.37** | **3.56** | **432** |
| **Size-aware motion combined** | **all four** | **10.26** | **3.47** | **5.44** | **3.57** | **390** |

The size-aware gate improves aggregate HOTA by **0.04 points**, IDF1 by **0.07 points**, and reduces ID switches by **42**. It modestly improves scenes 01 and 02; scene 03 produces byte-identical tracker results for the two variants, and scene 05 differs only slightly. This fresh test supports a small reduction in switching under the frozen setup. It does **not** establish a practically large accuracy gain or a high-performing tracker.

The absolute metrics are poor, especially in scenes 03 and 05. Their retained detection counts are only about 1% of the public-detection rows under the frozen 0.3 cutoff, and TrackEval DetA is below 0.5% in both. Detection recall therefore dominates the combined result. Changing the confidence threshold after seeing these scores would turn this external set into tuning data; no such change was made. MOT20 is also substantially more crowded than the MOT17 development scenes, so a small cross-domain gain should not be presented as an independent MOT17 held-out gain.

On the Apple M2 Pro CPU, single-run end-to-end tracking time over 8,931 frames was 3.40 seconds (2,624.80 FPS) for IoU and 23.31 seconds (383.18 FPS) for motion. These process wall times include detection parsing, association, and result writing, but no image decoding, rendering, or TrackEval. They are one-off measurements without uncertainty intervals.

## Reproduction

```bash
.venv/bin/python scripts/fetch_mot20_eval.py
.venv/bin/python scripts/run_external_mot20.py --resume
```

The runner writes official detailed metrics to `outputs/mot20_external/<variant>/metrics/external_mot20/pedestrian_detailed.csv` and the full evaluator log to `outputs/mot20_external/trackeval.log`. Dataset files and results remain under ignored `data/` and `outputs/`. The official MOT20 benchmark and labels are external data; consult its access terms before reuse or redistribution. The original [MOT17 final report](final_report.md) and [development gate follow-up](motion_gate_followup.md) remain separate from this external result.
