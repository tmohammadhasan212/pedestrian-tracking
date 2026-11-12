# Appearance training record

The appearance backbone is TorchVision `mobilenet_v3_small` with `IMAGENET1K_V1` weights, used as a **frozen** feature extractor. We remove its classifier and globally pool to a 576-dimensional vector. Each cropped pedestrian is resized to 256×128, scaled to `[0,1]`, and normalized with ImageNet mean and standard deviation. The pretrained features and learned 128-dimensional head outputs are L2 normalized.

The backbone weights came from the [TorchVision model entry](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.mobilenet_v3_small.html), SHA-256 `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`. TorchVision code is [BSD 3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE); ImageNet data has [non-commercial research and education access terms](https://image-net.org/accessagreement). Neither the pretrained weights nor the local checkpoint are committed.

## Exact training run

```bash
.venv/bin/python scripts/train_embedding.py --data-root data/MOT17 --max-samples 1024 --epochs 5 --pairs-per-epoch 2048 --batch-size 128 --device cpu
```

Seed 1704 and the committed `configs/split.json` selected only development scenes. GT samples required pedestrian category 1, ground-truth confidence 1, visibility at least 0.5, box size at least 8×16, and intersection with the image. A maximum of 8 frames per `(sequence, identity)` was considered before the 1,024-sample cap. Identity values were sequence-local. Positive pairs used distinct frames of one identity; negative pairs used different identities in the same sequence. Each epoch drew 2,048 balanced positive/negative pairs. The head was a bias-free linear 576→128 projection trained with AdamW (learning rate 0.001, weight decay 0.0001) and cosine contrastive loss: positive `(1 - cosine)^2`; negative `max(0, cosine - 0.2)^2`. The backbone stayed frozen.

| Development scene | Sampled and valid crops |
|---|---:|
| MOT17-02-FRCNN | 139 |
| MOT17-04-FRCNN | 190 |
| MOT17-05-FRCNN | 308 |
| MOT17-09-FRCNN | 73 |
| MOT17-13-FRCNN | 314 |

All 1,024 selected crops were valid, from 862 distinct frames. No held-out scene was read for training. The checkpoint's metadata stores the training-scene list; the tracker rejects a checkpoint listing a non-development scene. PyTorch 2.9.1 ran on CPU on an Apple M2 Pro MacBook Pro (16 GB memory); MPS was unavailable in this environment.

| Epoch | Mean sampled-pair loss |
|---:|---:|
| 1 | 0.070320 |
| 2 | 0.050228 |
| 3 | 0.041197 |
| 4 | 0.037128 |
| 5 | 0.031113 |

The local checkpoint is `outputs/appearance/head.pt`, SHA-256 `ce2c89ef09955ffbc93585b41cf24ec49404556133bc5e542b2f61f53a0a6162`. Its sidecar `head.json` contains exact hyperparameters and per-scene counts. The loss values describe training pairs only; they do not measure identity preservation in video.

## Real-video execution check

Both appearance configurations were run on the first 30 frames of MOT17-02-FRCNN with the same FRCNN detections and confidence threshold 0.3. The `--max-frames 30` suffix keeps these partial outputs out of TrackEval's expected full-sequence filenames.

| Variant | Results | Tracker time | End-to-end time with MP4 |
|---|---:|---:|---:|
| Pretrained features | 392 rows | 6.491 s | 8.143 s |
| Learned head | 392 rows | 6.503 s | 8.145 s |

These were one-off smoke timings, not repeated benchmark measurements. They preceded the full evaluation, so no tracking-quality claim followed from the training loss or result-row counts. Full TrackEval results are now in [final_report.md](final_report.md); the trained head improves on pretrained appearance but remains below the IoU baseline.
