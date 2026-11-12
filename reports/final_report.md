# Pedestrian tracking on MOT17: fixed-detection ablations

**Local research report.** This project assigns temporary track numbers within one video. A number is not a person identity. The implementation is a tracker built from MOT17 public detections, not a person-recognition system or an imported end-to-end tracker.

## Abstract and outcome

We implemented an IoU/Hungarian baseline, then added constant-velocity Kalman prediction and gating, frozen CNN appearance features, and a development-trained embedding head. All comparisons use the same official MOT17 FRCNN detections at confidence 0.3. The scene-safe split has five development and two held-out scenes. We selected thresholds from development TrackEval results and committed `configs/threshold_lock.json` at `a3af579` before any held-out result was generated.

On the two held-out scenes, **IoU alone was strongest**: 43.62 HOTA, 47.15 IDF1, and 402 ID switches. Motion obtained 25.41 HOTA and 2,527 switches. Pretrained and trained appearance obtained 26.22 and 27.24 HOTA respectively; the learned head improved on pretrained appearance by 1.02 HOTA points and 1.37 IDF1 points, but incurred 64 more switches. The learned method did **not** improve on IoU. These are local held-out training-scene results, not hidden MOTChallenge test-set results.

## Objective, scope, and criteria

The objective was stable, sequence-local pedestrian tracks through short gaps, with reproducible comparison of association methods. The concrete criteria were valid MOTChallenge results and annotated MP4 output; four fixed-input variants; official HOTA, IDF1, MOTA, and switch counts; development-only training and tuning; a held-out evaluation; and inspectable successes and failures. The pipeline and experiment protocol meet these execution criteria. The empirical accuracy goal of improving on IoU was not met.

The system consumes a frame folder or video and an external detection file. Detection is outside this implementation. The `track` command writes ten-column MOTChallenge rows; `--render` streams an annotated MP4 with boxes, confidence, and temporary IDs. The `evaluate` command invokes official TrackEval separately from tracking. Local example clips in `outputs/qualitative/` use benchmark imagery and are excluded from Git.

## Dataset and leakage controls

MOT17 has repeated detector versions of the same base scenes. We use only the official **FRCNN public detections**, and the detector file and 0.3 confidence cutoff are hashed/audited for every run. `configs/split.json` records seed **1704** and the method: shuffle sorted base scene IDs with `random.Random(seed)` and hold out the final two. Development base scenes are **02, 04, 05, 09, 13** (3,762 frames); held-out scenes are **10, 11** (1,554 frames). Other detector versions of a base scene remain in the same partition. No hidden test annotation was used.

The reader validates sequence dimensions, finite box values, frame bounds, confidence, and image intersection for detections, then returns detections in numeric frame order. MOT file boxes use one-based top-left coordinates; the reader converts them to zero-based internal boxes and the writer reverses that conversion. Identity labels are local to each sequence. The appearance training script reads only development ground truth; held-out annotations were used for data auditing and, after the threshold lock, evaluation and qualitative analysis, never training or tuning.

## System and methods

1. **IoU baseline.** For boxes A and B, IoU is intersection area divided by union area. A candidate assignment costs `1 - IoU`; pairs below configured IoU 0.3 are forbidden. Hungarian assignment with dummy unmatched choices resolves one-to-one matches. Unmatched detections start new tracks and unmatched tracks age.
2. **Motion.** Each track has state `[cx, cy, w, h, vx, vy, vw, vh]`. A constant-velocity transition predicts center and size; a measurement update uses the detected box and a Joseph-form covariance update. The squared Mahalanobis distance in four measurement dimensions is gated at 9.488 (a conventional 95% chi-square threshold). The selected motion config also requires IoU at least 0.3.
3. **Appearance.** A frozen TorchVision MobileNetV3 Small with `IMAGENET1K_V1` weights extracts a globally pooled 576-dimensional crop vector. Crops are resized to 256 x 128 and ImageNet-normalized. Vectors are L2-normalized. The compact learned head is a bias-free linear 576-to-128 projection, followed by L2 normalization. Similarity is cosine dot product. With a valid crop, association cost is `(1 - weight) * (1 - IoU) + weight * (1 - similarity) / 2`, subject to the Kalman gate, minimum similarity/IoU condition, and maximum cost 0.7. Missing crops fall back to IoU cost. A normalized exponential moving average with momentum 0.8 updates track features.
4. **Lifecycle and output.** Duplicate detections at IoU 0.9 are suppressed in descending confidence order. Tracks become confirmed after two hits, remain eligible for a short reactivation for up to 15 missed frames, then expire. IDs increase monotonically and are never reused within a run. Output lists confirmed tracks with current-frame detections only.

The association, filter, track lifecycle, IO, and renderer are original code in `src/tracker/`; PyTorch/TorchVision, SciPy, FFmpeg, MOT17, and TrackEval are external components.

### Appearance training

A scene-safe sample of 1,024 visible pedestrian crops from development ground truth provided 2,048 balanced positive/negative pairs per epoch for five CPU epochs. Positive pairs are distinct frames of the same identity; negatives use different identities in the same scene. Identity keys include scene name. AdamW used learning rate 0.001 and weight decay 0.0001. For cosine similarity `s`, the loss was `(1 - s)^2` for positive pairs and `max(0, s - 0.2)^2` for negative pairs. Mean sampled-pair loss fell from 0.070320 to 0.031113; this training loss is not a tracking metric. The backbone remained frozen. The ignored local checkpoint is `outputs/appearance/head.pt`, SHA-256 `ce2c89ef09955ffbc93585b41cf24ec49404556133bc5e542b2f61f53a0a6162`. Exact sampling counts and hyperparameters are in `reports/appearance_training.md` and the checkpoint sidecar.

## Evaluation protocol and threshold lock

Official TrackEval at commit `12c8791b303e0a0b50f753af204249e622d0281a` computed HOTA, detection accuracy (DetA), association accuracy (AssA), IDF1, MOTA, and CLEAR ID switches. A compatibility launcher restores NumPy's removed `np.float`/`np.int` names; the TrackEval metric implementation is unmodified. Each result and detector file, config, and checkpoint is SHA-256 recorded in `outputs/runs/<partition>/<variant>.json`. The fixed-input audit passed on both partitions. A cache of raw frozen CNN features accelerated development threshold trials; those cached wall times are **not** reported as end-to-end inference speed. The four default development timings and every held-out timing used uncached inference.

Development aggregate results below are percentages except ID switches. The first four rows are the pre-tuning comparison. Remaining rows are threshold trials. The selected held-out variants are marked with `*`; `appearance_pretrained_weight02` replaces the default pretrained weight 0.6 with 0.2. Config hashes and the full-precision development evidence are in the committed threshold lock.

| Development variant | HOTA | IDF1 | MOTA | ID switches |
|---|---:|---:|---:|---:|
| iou * | 45.32 | 51.41 | 47.57 | 799 |
| motion * | 38.97 | 42.65 | 43.68 | 3,051 |
| appearance_pretrained | 38.98 | 42.66 | 43.77 | 3,190 |
| appearance_finetuned * | 39.63 | 43.47 | 43.73 | 3,180 |
| motion_gate3 | 28.95 | 29.83 | 37.95 | 6,208 |
| motion_missed3 | 38.26 | 41.78 | 43.65 | 2,451 |
| motion_iou05 | 38.88 | 42.68 | 43.44 | 2,945 |
| motion_strict | 27.44 | 28.06 | 37.74 | 4,909 |
| appearance_pretrained_weight02 * | 39.05 | 42.74 | 43.77 | 3,092 |
| appearance_finetuned_weight02 | 39.47 | 43.15 | 43.79 | 3,063 |
| appearance_finetuned_weight08 | 39.62 | 43.51 | 43.72 | 3,190 |

HOTA was the primary selection metric. The default motion config had the highest HOTA of its tested motion settings. Pretrained weight 0.2 modestly improved HOTA and IDF1 while reducing switches. Trained-head weight 0.6 narrowly led the 0.2 and 0.8 trials in HOTA; weight 0.8 led IDF1 by only 0.04 points. This selection was made before evaluating scenes 10 and 11, and no thresholds were changed afterward.

## Held-out results

All four rows below combine the same two held-out FRCNN scenes. TrackEval combines counts and matches across sequences; the aggregate is not a simple average of displayed scene percentages.

| Locked variant | HOTA | DetA | AssA | IDF1 | MOTA | ID switches |
|---|---:|---:|---:|---:|---:|---:|
| IoU | **43.62** | **49.72** | **38.57** | **47.15** | **52.44** | **402** |
| IoU + Kalman motion | 25.41 | 46.35 | 13.96 | 23.96 | 39.14 | 2,527 |
| Motion + pretrained appearance (weight 0.2) | 26.22 | 46.49 | 14.82 | 25.02 | 39.16 | 2,534 |
| Motion + trained head (weight 0.6) | 27.24 | 46.31 | 16.05 | 26.39 | 38.83 | 2,598 |

The largest drop is in AssA when motion gating is introduced. This is consistent with premature rejection of plausible associations and resulting fragmentation, but aggregate metrics alone cannot prove which individual gate failed. The trained head's advantage over frozen features is concentrated in scene 10: HOTA 29.44 versus 27.78. In scene 11 it is slightly lower, 23.55 versus 23.65. The learned head also has 2,598 switches against 2,534 for pretrained appearance. Thus its average identity accuracy gain is qualified, and no overall improvement over IoU can be claimed.

| Variant | Scene | HOTA | IDF1 | ID switches | Processing FPS |
|---|---|---:|---:|---:|---:|
| IoU | 10 | 38.58 | 42.69 | 338 | 1,369.33 |
| IoU | 11 | 49.77 | 53.34 | 64 | 5,824.09 |
| Motion | 10 | 26.64 | 27.10 | 967 | 83.55 |
| Motion | 11 | 23.41 | 19.50 | 1,560 | 187.67 |
| Pretrained appearance | 10 | 27.78 | 28.35 | 962 | 4.20 |
| Pretrained appearance | 11 | 23.65 | 20.26 | 1,572 | 5.22 |
| Trained head | 10 | 29.44 | 30.96 | 972 | 4.28 |
| Trained head | 11 | 23.55 | 19.82 | 1,626 | 5.38 |

The runs used one Apple M2 Pro MacBook Pro with 16 GB memory, macOS, Python 3.12, and CPU inference; MPS was unavailable. Processing FPS is frames divided by one measured process wall time, including detection parsing, tracking, feature inference when applicable, and result writing. It excludes optional MP4 rendering and TrackEval. Aggregate FPS across 1,554 frames was 2,458.33 (IoU), 123.10 (motion), 4.73 (pretrained), and 4.85 (trained). These are single runs without uncertainty intervals. IoU and motion do not decode frame images, while appearance does; therefore their wall times reflect different required input work. A separate short rendered-video smoke run verified the MP4 path, but its time is not mixed into this table.

## Qualitative examples

`scripts/render_examples.py` generated six short, 640 x 640 local MP4 clips from the trained-head held-out results. Red shows one target ground-truth box and green its one-to-one IoU-matched tracker box (minimum IoU 0.5); the banner reports frame, GT number, and track number. The clips and `outputs/qualitative/manifest.json` are ignored because they contain MOT17 imagery. They can be regenerated with the command below after placing the dataset locally. The matching is for case review only and does not replace official TrackEval.

| Clip | Scene, GT identity, frames | Observed temporary track ID | Interpretation |
|---|---|---|---|
| `preserved_01.mp4` | 10, GT 4, 350-380 | 320 throughout | Stable visible pedestrian track. |
| `preserved_02.mp4` | 10, GT 2, 510-540 | 413 throughout | Stable track across a moving pedestrian. |
| `preserved_03.mp4` | 11, GT 26, 170-200 | 93 throughout | Stable track in the second held-out scene. |
| `switch_01.mp4` | 10, GT 4, 555-572 | 320 to 469 at 565 | Frame 564 has a high-overlap public detection (IoU 0.84) yet no reported match; new tentative track becomes visible at 565. Association/confirmation fragmentation, not a public-detection miss at the transition. |
| `switch_02.mp4` | 11, GT 5, 97-112 | 21 to 83 at 105 | The target has visibility 1.0 and detection IoU 0.87 at unreported frame 104. A track is lost despite a strong detection, implicating association or gating. |
| `switch_03.mp4` | 11, GT 31, 397-412 | 224 to 267 at 405 | Visibility rises from 0.59 to 0.67 around frame 404, while detection IoU is 0.94. Partial occlusion may complicate tracking, but an available strong detection makes association/confirmation the immediate failure mechanism. |

For these three selected transitions, the public detector produced a box overlapping the target on the unreported frame; the result files alone do not isolate whether the Kalman gate, cost limit, or lifecycle rule rejected it. They do not establish a camera-motion cause. Elsewhere, missed public detections and occlusion remain plausible sources of error, but we do not assign a specific cause without inspecting the relevant frames and input boxes.

## Limitations and threats to validity

Only two base scenes form the held-out set, and the appearance head used 1,024 development crops and one training seed. There are no repeated run timings, confidence intervals, or cross-detector experiments. The fixed public detector limits recall and can dominate all tracker metrics. The simple constant-velocity model and narrow Mahalanobis gate are a likely mismatch for abrupt image motion; the much lower AssA and many switches justify investigating this, not declaring it proven for every event. The threshold search was small, chosen on the development set, and could miss better configurations. Matching qualitative cases at IoU 0.5 is a separate inspection method and should not be confused with TrackEval's full evaluation. We do not infer real-world person identity from any track number.

## Reproduction and artifact map

Install Python 3.12 and FFmpeg/FFprobe, then `python3.12 -m venv .venv` and `.venv/bin/python -m pip install -e '.[test,appearance,evaluation]'`. Pinned package versions are in `pyproject.toml`: PyTorch 2.9.1, TorchVision 0.24.1, Pillow 12.0.0, NumPy 2.5.4, SciPy 1.18.1, and pytest 9.0.3. Obtain the official MOT17 training archive separately and place the seven FRCNN sequences at `data/MOT17/train/MOT17-XX-FRCNN/`, each with `seqinfo.ini`, `img1`, `det/det.txt`, and `gt/gt.txt`. Obtain official TrackEval at the commit above in `data/TrackEval/`. The repository does not include benchmark frames, annotations, pretrained weights, the trained checkpoint, or local result files.

```bash
.venv/bin/python -m pytest -q
.venv/bin/pedtrack audit-data --data-root data/MOT17 --partition development
.venv/bin/pedtrack audit-data --data-root data/MOT17 --partition held_out
.venv/bin/python scripts/train_embedding.py --data-root data/MOT17 --max-samples 1024 --epochs 5 --pairs-per-epoch 2048 --batch-size 128 --device cpu
.venv/bin/python scripts/run_experiments.py --partition development --variants iou motion appearance_pretrained appearance_finetuned --device cpu
.venv/bin/pedtrack evaluate --data-root data/MOT17 --trackeval-root data/TrackEval --partition development --variants iou motion appearance_pretrained appearance_finetuned
# Run the committed development ablation configs and inspect configs/threshold_lock.json before held-out evaluation.
.venv/bin/python scripts/run_experiments.py --partition held_out --variants iou motion appearance_pretrained_weight02 appearance_finetuned --device cpu
.venv/bin/pedtrack evaluate --data-root data/MOT17 --trackeval-root data/TrackEval --partition held_out --variants iou motion appearance_pretrained_weight02 appearance_finetuned
.venv/bin/python scripts/render_examples.py
```

The detailed metric CSVs are `outputs/<variant>/metrics/<partition>/pedestrian_detailed.csv`; per-sequence provenance and runtime are `outputs/runs/<partition>/<variant>.json`. The lock file records exact selected config hashes. For a short interactive demo, `pedtrack track --sequence-dir data/MOT17/train/MOT17-02-FRCNN --config configs/iou.json --variant demo --max-frames 30 --render` produces a partial MOT text file and annotated MP4. A partial file is suffixed `-first30` so it cannot satisfy a full-sequence evaluation.

## Attribution, references, and data use

Our contribution is the input validation, association/tracking code, experiment runner, compatibility launcher, cache, renderer, and reports. The MOT17 benchmark and public detections are external data; the MobileNetV3 backbone and ImageNet-pretrained weights are external models; TrackEval supplies the official metric formulas. No benchmark imagery or weights are committed. Consult the official MOTChallenge access terms before redistributing clips or using data beyond the permitted scope. TorchVision code is BSD 3-Clause; TrackEval is MIT. ImageNet access is described for non-commercial research/education under its conditions. The local six clips remain ignored rather than redistributed.

- P. Dendorfer et al., [MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking](https://link.springer.com/article/10.1007/s11263-020-01393-0), IJCV 2021; [MOT17 data page](https://motchallenge.net/data/MOT17/).
- J. Luiten et al., [HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking](https://link.springer.com/article/10.1007/s11263-020-01375-2), IJCV 2021; [official TrackEval repository and MIT license](https://github.com/JonathonLuiten/TrackEval).
- [TorchVision MobileNetV3 Small weight documentation](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.mobilenet_v3_small.html) and [BSD 3-Clause license](https://github.com/pytorch/vision/blob/main/LICENSE).
- [ImageNet access conditions](https://www.image-net.org/download-images.php).
