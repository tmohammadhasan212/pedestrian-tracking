# Research progress record

The project has completed the four fixed-detection development and held-out runs, development-only threshold ablations, a committed threshold lock, official TrackEval evaluation, six local qualitative clips, and the final Markdown/PDF research report. See [final_report.md](final_report.md) for the measured results, methods, limitations, and exact reproduction commands.

Implementation milestones were validated in focused commits: MOT17 parsing and scene-safe split; IoU/Hungarian association; Kalman prediction and gating; lifecycle; pretrained features; development-only embedding training; motion/appearance fusion; renderer and CLI; official TrackEval integration; run provenance; development ablations and threshold lock; and qualitative examples. The full history has at least 15 meaningful commits. Run `.venv/bin/python -m pytest -q` for the current tests.

The core empirical finding is a regression from the IoU baseline when motion gating and appearance are added. The trained head improves on the pretrained appearance variant but does not recover the baseline's identity consistency. Do not present this as an overall tracking improvement.
