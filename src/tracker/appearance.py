"""Frozen pretrained MobileNetV3 crop features and optional trained embedding head."""

from math import ceil, floor
from pathlib import Path
from typing import Any

from .geometry import Box

BACKBONE_NAME = "mobilenet_v3_small"
WEIGHTS_NAME = "IMAGENET1K_V1"
FEATURE_DIM = 576
EMBEDDING_DIM = 128


def crop_person(image: Any, box: Box) -> Any | None:
    """Clip a MOT box to the image; skip missing or tiny crops."""
    left = max(0, floor(box.x))
    top = max(0, floor(box.y))
    right = min(image.width, ceil(box.x + box.w))
    bottom = min(image.height, ceil(box.y + box.h))
    if right - left < 4 or bottom - top < 8:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


class FeatureExtractor:
    def __init__(self, checkpoint: Path | None = None, device: str = "auto", pretrained: bool = True) -> None:
        try:
            import torch
            from torchvision import models, transforms
        except ImportError as exc:
            raise RuntimeError("install the appearance extra: pip install -e '.[appearance]'") from exc
        self.torch = torch
        self.device = ("mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        if pretrained:
            torch.hub.set_dir(str(Path(__file__).resolve().parents[2] / "data" / "torch"))
        model = models.mobilenet_v3_small(weights=weights)
        self.backbone = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten()).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)), transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.head = None
        self.metadata = None
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            metadata = payload["metadata"]
            if metadata["backbone"] != BACKBONE_NAME or metadata["feature_dim"] != FEATURE_DIM:
                raise ValueError("appearance checkpoint has incompatible backbone")
            self.head = torch.nn.Linear(FEATURE_DIM, EMBEDDING_DIM, bias=False)
            self.head.load_state_dict(payload["state_dict"])
            self.head = self.head.to(self.device).eval()
            self.metadata = metadata

    def encode_boxes(self, image: Any, boxes: list[Box], use_head: bool = True,
                     batch_size: int = 32, normalize_output: bool = True) -> list[tuple[float, ...] | None]:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        crops = [crop_person(image, box) for box in boxes]
        features: list[tuple[float, ...] | None] = [None] * len(boxes)
        valid = [(index, crop) for index, crop in enumerate(crops) if crop is not None]
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, len(valid), batch_size):
                chunk = valid[start:start + batch_size]
                batch = torch.stack([self.transform(crop) for _, crop in chunk]).to(self.device)
                vectors = self.backbone(batch)
                if use_head and self.head is not None:
                    vectors = self.head(vectors)
                if normalize_output:
                    vectors = torch.nn.functional.normalize(vectors, dim=1)
                vectors = vectors.cpu()
                for (index, _), vector in zip(chunk, vectors):
                    features[index] = tuple(float(value) for value in vector.tolist())
        return features

    def extract_frame(self, path: Path, boxes: list[Box]) -> list[tuple[float, ...] | None]:
        from PIL import Image

        with Image.open(path) as image:
            return self.encode_boxes(image, boxes)

    def project_raw(self, vectors: list[tuple[float, ...] | None]) -> list[tuple[float, ...] | None]:
        valid = [(index, vector) for index, vector in enumerate(vectors) if vector is not None]
        result: list[tuple[float, ...] | None] = [None] * len(vectors)
        with self.torch.inference_mode():
            for start in range(0, len(valid), 128):
                chunk = valid[start:start + 128]
                batch = self.torch.tensor([vector for _, vector in chunk], dtype=self.torch.float32,
                                          device=self.device)
                projected = self.head(batch) if self.head is not None else batch
                projected = self.torch.nn.functional.normalize(projected, dim=1).cpu()
                for (index, _), vector in zip(chunk, projected):
                    result[index] = tuple(float(value) for value in vector.tolist())
        return result
