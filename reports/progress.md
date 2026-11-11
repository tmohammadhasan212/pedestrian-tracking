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

The first official FRCNN training scene has been downloaded locally and used for a smoke run. The command `pedtrack track --sequence-dir data/MOT17/train/MOT17-02-FRCNN --config configs/iou.json --variant iou --render` produced 7,621 result rows and a 600-frame, 1920×1080 MP4. One wall-clock run on an Apple M2 Pro MacBook Pro with 16 GB memory took 0.168 seconds for tracker association and 7.264 seconds end to end, including decoding and rendering. This is a smoke-run timing, not a benchmark or comparison. The other FRCNN training scenes are being retrieved for the committed split audit.
