# Local MOT17 evaluation protocol

Official [TrackEval](https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md) is cloned into ignored `data/TrackEval/` at commit `12c8791b303e0a0b50f753af204249e622d0281a`. The local Python environment uses NumPy 2.5.4 and SciPy 1.18.1. This TrackEval revision still refers to the removed aliases `np.float` and `np.int`, so `scripts/run_trackeval_compat.py` restores those names to their original Python types before running the **unmodified** official metric implementation. Its CLI also interprets optional path arguments as lists; `pedtrack evaluate` uses the default seqmap location and `OUTPUT_SUB_FOLDER` instead. No metric formula or TrackEval source file is changed.

The fixed input is the official MOT17 **FRCNN** public detection file in each scene, with confidence threshold 0.3. `scripts/run_experiments.py` hashes each detector file, config, checkpoint, and result file and writes a per-sequence manifest under `outputs/runs/<partition>/<variant>.json`. It rejects detector hash or confidence-threshold mismatches across variants. Full-sequence result files use the exact sequence name; partial smoke files carry a `-firstN` suffix and cannot satisfy evaluation.

Development scenes are 02, 04, 05, 09, and 13. Held-out scenes are 10 and 11. The split is fixed in `configs/split.json`; the same base scene never appears in both partitions. Appearance training used only development annotations. Thresholds must be selected with development TrackEval results and recorded in a lock file before the held-out run.

Run variants and official metrics with:

```bash
.venv/bin/python scripts/run_experiments.py --partition development --variants iou motion appearance_pretrained appearance_finetuned --device cpu
.venv/bin/pedtrack evaluate --data-root data/MOT17 --trackeval-root data/TrackEval --partition development --variants iou motion appearance_pretrained appearance_finetuned
```

TrackEval writes `pedestrian_detailed.csv` and `pedestrian_summary.txt` under `outputs/<variant>/metrics/<partition>/`. The detailed CSV contains per-sequence and aggregate HOTA, DetA, AssA, IDF1, MOTA, and ID switches. The script reports per-sequence processing time separately. All benchmark numbers and hardware will be copied into a report only after the full run succeeds.
