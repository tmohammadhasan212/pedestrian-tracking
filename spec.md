# Project 4 — Multi-Object Pedestrian Tracking in Video

**Level:** Medium · **Indicative duration:** 5–7 weeks · **Primary skills:** detection, motion estimation, data association, appearance learning, video systems

## Project goal

Implement a tracker that assigns stable identities to pedestrians across video frames, including short occlusions and reappearances. Build the association and track-management logic yourself, then measure how motion and learned appearance features change tracking quality. The central deliverable is a reproducible tracker with a clear experimental comparison, not an imported end-to-end tracking wrapper.

## User-facing behavior

Given a video or a folder of frames plus a detector output file, the system emits a standard tracking-results file and an annotated video with pedestrian boxes, IDs, confidence, and optional trails. It reports total processing time and frames per second. The display must not suggest that the assigned IDs identify real people; they are temporary track numbers within that video.

## Dataset and split contract

- Use [MOT17](https://motchallenge.net/data/MOT17/) and its documented annotation/detection format. Start with one official public-detection variant and identify it in all result tables. Keep detector inputs fixed across tracker ablations.
- Split the released training scenes into development and held-out evaluation scenes. MOT17 includes repeated versions of the same underlying scene with different detections; group these versions together so the same scene cannot land in both splits. Commit the exact scene list and seed.
- Use identity annotations from development scenes to train or fine-tune the appearance model. Do not train or tune on identities from held-out evaluation scenes. Treat identity numbers as sequence-local labels, not global person identities.
- Do not use hidden test annotations. If you later use the public benchmark server, follow its submission rules and label results separately from local held-out results.
- Document dataset use terms, citation, and any pretrained appearance-model license. Do not commit the benchmark videos or annotations.

## Required implementation

1. **Input adapter:** parse sequence metadata, frames, detections, and ground truth. Validate box coordinates, frame order, confidence values, and image dimensions. Support deterministic filtering by detection confidence.
2. **IoU tracker baseline:** implement track creation/deletion, box overlap cost, Hungarian assignment, and unmatched detection/track handling. Make minimum track age and maximum missed frames configurable.
3. **Motion model:** implement a constant-velocity Kalman filter for box state, including predict/update steps and covariance handling. Add gating before assignment so impossible matches are rejected.
4. **Appearance model:** extract a feature vector from each detected person crop using a pretrained CNN. Fine-tune or train a compact embedding head with identity-based positive/negative pairs from development scenes; document training loss and sampling strategy. Normalize features and compare cosine or Euclidean distance.
5. **Combined association:** fuse motion/IoU and appearance costs with explicit weights. Handle missing crops, occlusions, duplicate detections, and low-confidence detections. Prevent matching a detection to more than one track.
6. **Track lifecycle:** tentative versus confirmed tracks, aging, deletion, and short-term reactivation. Track IDs must remain stable for the lifetime of a track and never be accidentally reused within a video.
7. **Output and demo:** write MOTChallenge-format result files and render annotated MP4 or equivalent. Expose `track` and `evaluate` commands. Keep evaluation separate from the live tracker code.

## Experiments and metrics

Use the [official TrackEval implementation](https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md) for HOTA, IDF1, MOTA, identity switches, and detection/association components where available. Report per-sequence and aggregate results, plus end-to-end frames per second and hardware. Keep input detections identical for these comparisons:

1. IoU association only.
2. IoU plus Kalman prediction/gating.
3. Motion plus pretrained appearance embeddings.
4. Motion plus fine-tuned appearance embeddings.

Then ablate key association thresholds on development scenes and lock them before held-out evaluation. Show at least three video clips where identities are preserved and three where they switch or fragment. Explain whether failure is caused by detector misses, occlusion, camera motion, or association logic. A detector can dominate tracking results, so do not claim a tracking improvement if detector inputs changed between runs.

## Repository contract

```text
pedestrian-tracking/
  README.md              task, demo clip, result table, limitations
  DATA.md                MOT17 access, scenes, split, detector choice
  pyproject.toml         pinned dependencies
  configs/               tracker and appearance-training variants
  src/tracker/           readers, kalman, assignment, tracks, features, IO
  scripts/               audit_data, train_embedding, track, evaluate, render
  tests/                 matching, state transitions, formats, geometry
  reports/               metric tables, ablations, failure analysis
  assets/                short clips/frames if redistribution is permitted
```

Document how to place MOT17 locally and how to run one short sequence before the full evaluation. If benchmark footage cannot be redistributed, publish a diagram, generated visualization, or link to your results instead of uploading the source video.

## Git history and final research reports

- After implementing and validating each feature, make a focused Git commit that records the completed work and its relevant tests or documentation. Use a clear, descriptive message that says what changed (for example, `feat: add Kalman prediction and gating`). Commit fixes and experiment updates when they form meaningful, reviewable units of work.
- The completed project must have at least **15 meaningful commits** across its development history. This is a minimum for real, validated progress, not a target to satisfy with empty commits, cosmetic changes, or artificial splitting of one change. Do not claim a feature is complete before its validation passes and its commit is recorded.
- Deliver both `reports/final_report.md` and `reports/final_report.pdf` at the end of the project. The PDF must faithfully present the same substantive findings as the Markdown report. Write them as research-style reports, with figures and tables where useful, that cover:
  - The project objectives, scope, success criteria, and system architecture.
  - The MOT17 dataset and annotation/detection choices, the exact development and held-out scene split and seed, and how repeated detector versions and identity labels were handled to prevent leakage.
  - The step-by-step implementation and how each milestone was achieved, supported by code, tests, experiments, or other verifiable evidence.
  - The mathematical and methodological rationale for the tracker components, including IoU, assignment, Kalman prediction and gating, appearance embeddings and training loss, association fusion, and track lifecycle decisions.
  - The evaluation protocol, fixed detector inputs, development-only tuning, experimental configurations and ablations, metrics (including HOTA, IDF1, MOTA, identity switches, and runtime where available), and per-sequence and aggregate results. Explain both improvements and regressions without overstating accuracy.
  - Qualitative successes and failures, limitations and threats to validity, and possible next steps.
  - Reproducibility details: dependency versions, configuration, seeds, hardware, dataset access and placement, commands to train, track, and evaluate, and the files needed to regenerate tables and figures.
  - References, dataset and model licenses or use terms, and honest attribution of original implementation versus pretrained models, external code, tools, and other borrowed work. Do not invent experiments, measurements, citations, or contributions.

## Definition of done

- Unit tests cover box conversions, IoU, assignment with unmatched tracks, Kalman predict/update on synthetic motion, track aging, ID uniqueness, and output format.
- A short video runs end to end from detection file to annotated video and standard results file.
- Four ablation variants use exactly the same held-out scenes and detection inputs, and TrackEval reproduces the reported metrics.
- The appearance training set is separated by scene from evaluation; repeat MOT17 detector versions of the same scene never cross the boundary.
- The report includes per-sequence HOTA/IDF1, ID switches, runtime, and annotated identity-switch examples.
- Each completed feature has a focused commit after validation, and the finished project has at least 15 meaningful, descriptive commits without filler.
- Both `reports/final_report.md` and `reports/final_report.pdf` are complete, consistent research-style reports covering the requirements above.

## Suggested milestones

1. **Week 1:** dataset parsing, scene-safe split, IoU tracker and output format.
2. **Week 2:** Kalman filter, gating, track lifecycle, synthetic tests.
3. **Weeks 3–4:** pretrained features and appearance-head training.
4. **Weeks 5–7:** TrackEval comparison, threshold locking, video demo, report.

## CV evidence to collect

Use measured results in a line like: “Implemented a pedestrian tracker with Kalman filtering, Hungarian assignment, and learned appearance features; improved [measured metric] over an IoU baseline on held-out MOT17 scenes.”
