# Research progress

This is a progress log, not the final report. The requested `final_report.md` and `final_report.pdf` will be produced after MOT17 evaluation so that their tables and findings can be supported by actual measurements.

## Architecture implemented

1. MOT17 sequence metadata, detections, and annotations are parsed with validation and a fixed detector threshold.
2. IoU cost is `1 - intersection / union`. A Hungarian assignment with dummy columns permits unmatched tracks. Detections below the IoU threshold are excluded.
3. Tracks begin tentative, become confirmed after `min_hits`, and expire after `max_missed`. IDs increase monotonically within one tracker instance. Duplicate boxes are suppressed in confidence order.
4. The motion variant uses state `[cx, cy, w, h, vx, vy, vw, vh]`. A constant-velocity transition predicts the next box and propagates covariance. The update uses measured center and size with a Joseph covariance update. A four-dimensional Mahalanobis gate rejects distant detections before the same IoU assignment.
5. The tracker writes MOTChallenge text. FFmpeg-based rendering streams frames and labels them with temporary track number and confidence.
6. `evaluate` invokes official TrackEval with an exact seqmap for the chosen partition. It does not run until all result files exist.

## Validation so far

Synthetic tests cover box conversion, IoU, input validation, scene split, Hungarian assignment with unmatched rows, lifecycle and ID uniqueness, Kalman predict/update/gating, output format, and a three-frame sequence rendered to MP4. Run `python3.12 -m pytest -q` for current status.

## Outstanding work

- Pretrained CNN features, scene-safe identity pair sampling, and embedding-head training.
- Motion/appearance fusion, missing crop handling, and four fixed-input ablations.
- Development-only threshold selection and held-out TrackEval metrics, runtime/hardware table, and qualitative clips.
- Final Markdown and faithful PDF research reports.

All seven official FRCNN training scenes were downloaded locally with `scripts/fetch_mot17.py`. Both committed partitions passed `pedtrack audit-data`, including every frame path and first/last image dimensions. At the fixed 0.3 detection-confidence cutoff, the audit counted:

| Partition | Scene | Frames | Retained detections | Annotations |
|---|---|---:|---:|---:|
| Development | 02 | 600 | 7,755 | 30,003 |
| Development | 04 | 1,050 | 27,957 | 108,005 |
| Development | 05 | 837 | 3,627 | 8,013 |
| Development | 09 | 525 | 2,980 | 10,411 |
| Development | 13 | 750 | 7,642 | 20,202 |
| Held-out | 10 | 654 | 9,095 | 17,450 |
| Held-out | 11 | 900 | 5,775 | 10,617 |

The first real smoke run used MOT17-02-FRCNN and produced 600-frame, 1920×1080 annotated MP4s for both variants. The IoU baseline wrote 7,621 MOT result rows. Measurements from one wall-clock run each on an Apple M2 Pro MacBook Pro with 16 GB memory:

| Variant | Tracker time | Tracker FPS | End-to-end time with video | End-to-end FPS |
|---|---:|---:|---:|---:|
| IoU | 0.168 s | 3,563.71 | 7.264 s | 82.60 |
| Motion | 5.217 s | 115.01 | 12.271 s | 48.89 |

These are smoke-run timings, not repeated benchmark measurements. No TrackEval metric or tracking-quality comparison has been computed yet.
