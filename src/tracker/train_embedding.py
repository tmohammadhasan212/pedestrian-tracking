"""Train a compact appearance head on frozen pretrained crop features."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .appearance import BACKBONE_NAME, EMBEDDING_DIM, FEATURE_DIM, WEIGHTS_NAME, FeatureExtractor
from .appearance_data import AppearanceSample, PairSampler, development_samples

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _feature_bank(extractor: FeatureExtractor, samples: list[AppearanceSample]) -> tuple[list[AppearanceSample], list[tuple[float, ...]]]:
    from PIL import Image

    by_image: dict[Path, list[AppearanceSample]] = defaultdict(list)
    for sample in samples:
        by_image[sample.image_path].append(sample)
    retained: list[AppearanceSample] = []
    features: list[tuple[float, ...]] = []
    for index, (path, group) in enumerate(sorted(by_image.items()), 1):
        with Image.open(path) as image:
            vectors = extractor.encode_boxes(image, [sample.box for sample in group], use_head=False)
        for sample, vector in zip(group, vectors):
            if vector is not None:
                retained.append(sample)
                features.append(vector)
        if index % 100 == 0:
            print(f"encoded {index}/{len(by_image)} frames", flush=True)
    return retained, features


def train(samples: list[AppearanceSample], features: list[tuple[float, ...]], *,
          seed: int, epochs: int, pairs_per_epoch: int, batch_size: int, learning_rate: float,
          margin: float, device: str) -> tuple[object, list[float]]:
    import torch
    import torch.nn.functional as functional

    if epochs < 1 or pairs_per_epoch < 2 or batch_size < 1 or learning_rate <= 0 or not -1 < margin < 1:
        raise ValueError("invalid training hyperparameters")
    if len(samples) != len(features) or any(len(vector) != FEATURE_DIM for vector in features):
        raise ValueError("feature bank shape does not match samples")
    torch.manual_seed(seed)
    sampler = PairSampler(samples, seed)
    bank = torch.tensor(features, dtype=torch.float32, device=device)
    head = torch.nn.Linear(FEATURE_DIM, EMBEDDING_DIM, bias=False).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        used = 0
        for start in range(0, pairs_per_epoch, batch_size):
            count = min(batch_size, pairs_per_epoch - start)
            pairs = [sampler.pair((start + i) % 2 == 0) for i in range(count)]
            first = torch.tensor([pair[0] for pair in pairs], device=device)
            second = torch.tensor([pair[1] for pair in pairs], device=device)
            target = torch.tensor([pair[2] for pair in pairs], dtype=torch.float32, device=device)
            first_embedding = functional.normalize(head(bank[first]), dim=1)
            second_embedding = functional.normalize(head(bank[second]), dim=1)
            similarity = (first_embedding * second_embedding).sum(dim=1)
            positive_loss = target * (1 - similarity).square()
            negative_loss = (1 - target) * functional.relu(similarity - margin).square()
            loss = (positive_loss + negative_loss).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * count
            used += count
        losses.append(epoch_loss / used)
        print(f"epoch {epoch + 1}/{epochs}: contrastive loss {losses[-1]:.6f}", flush=True)
    return head.cpu().eval(), losses


def main(argv: list[str] | None = None) -> None:
    import torch

    parser = argparse.ArgumentParser(prog="train_embedding")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "configs" / "split.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "appearance" / "head.pt")
    parser.add_argument("--seed", type=int, default=1704)
    parser.add_argument("--max-per-identity", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pairs-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args(argv)
    samples = development_samples(args.data_root, args.split, args.seed, args.max_per_identity,
                                  args.max_samples, args.min_visibility)
    scene_counts = Counter(sample.sequence for sample in samples)
    print("sampled development scenes:", dict(sorted(scene_counts.items())), flush=True)
    extractor = FeatureExtractor(device=args.device)
    retained, features = _feature_bank(extractor, samples)
    print(f"valid crops: {len(retained)}/{len(samples)}", flush=True)
    head, losses = train(retained, features, seed=args.seed, epochs=args.epochs,
                         pairs_per_epoch=args.pairs_per_epoch, batch_size=args.batch_size,
                         learning_rate=args.learning_rate, margin=args.margin, device=extractor.device)
    metadata = {
        "backbone": BACKBONE_NAME, "pretrained_weights": WEIGHTS_NAME,
        "pretrained_weights_sha256": hashlib.sha256(
            (PROJECT_ROOT / "data" / "torch" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth").read_bytes()
        ).hexdigest(),
        "feature_dim": FEATURE_DIM, "embedding_dim": EMBEDDING_DIM,
        "training_scenes": sorted({sample.sequence for sample in retained}),
        "samples_per_scene": dict(sorted(Counter(sample.sequence for sample in retained).items())),
        "seed": args.seed, "max_per_identity": args.max_per_identity,
        "max_samples": args.max_samples, "epochs": args.epochs, "pairs_per_epoch": args.pairs_per_epoch,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "margin": args.margin, "min_visibility": args.min_visibility, "device": extractor.device,
        "loss": "balanced cosine contrastive: positive (1-cos)^2; negative max(0,cos-margin)^2",
        "epoch_losses": losses,
        "torch_version": str(torch.__version__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "metadata": metadata}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved head: {args.output}")


if __name__ == "__main__":
    main()
