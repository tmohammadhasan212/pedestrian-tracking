from collections import Counter
from pathlib import Path

import pytest

from tracker.appearance import FEATURE_DIM, FeatureExtractor, crop_person
from tracker.appearance_data import PairSampler, development_samples
from tracker.train_embedding import train
from tracker.geometry import Box
from tracker.split import load_split, validate_training_scenes


def test_sampling_uses_only_development_scenes_and_sequence_local_ids(tmp_path):
    split = Path(__file__).resolve().parents[1] / "configs" / "split.json"
    for scene in ("02", "04", "05", "09", "13"):
        name = f"MOT17-{scene}-FRCNN"
        root = tmp_path / "train" / name
        (root / "gt").mkdir(parents=True)
        (root / "seqinfo.ini").write_text(
            f"[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength=2\nimWidth=100\nimHeight=80\nimExt=.jpg\n"
        )
        (root / "gt" / "gt.txt").write_text(
            "1,1,1,1,12,20,1,1,0.9\n2,1,2,1,12,20,1,1,0.9\n"
            "1,2,40,1,12,20,1,1,0.9\n2,2,41,1,12,20,1,1,0.9\n"
            "1,3,70,1,12,20,1,7,0.9\n"  # distractor class
        )
    samples = development_samples(tmp_path, split, 1704, max_per_identity=2, max_samples=100)
    assert len(samples) == 20
    assert set(Counter(item.sequence for item in samples).values()) == {4}
    assert all(item.identity in (1, 2) for item in samples)
    assert all("MOT17-10" not in item.sequence and "MOT17-11" not in item.sequence for item in samples)
    sampler = PairSampler(samples, 1704)
    for _ in range(20):
        first, second, label = sampler.pair(True)
        assert label == 1 and first != second and samples[first].key == samples[second].key
        first, second, label = sampler.pair(False)
        assert label == 0 and samples[first].sequence == samples[second].sequence
        assert samples[first].key != samples[second].key


def test_crop_clips_and_skips_missing_regions():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (40, 40), "red")
    assert crop_person(image, Box(-2, 0, 10, 20)).size == (8, 20)
    assert crop_person(image, Box(-20, 0, 10, 20)) is None


def test_backbone_features_are_normalized_without_network():
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("torch")
    extractor = FeatureExtractor(device="cpu", pretrained=False)
    image = Image.new("RGB", (64, 128), "blue")
    features = extractor.encode_boxes(image, [Box(0, 0, 64, 128), Box(100, 0, 10, 20)])
    assert features[1] is None
    assert features[0] is not None and len(features[0]) == FEATURE_DIM
    assert sum(value * value for value in features[0]) == pytest.approx(1, abs=1e-5)


def test_head_training_and_checkpoint_load_without_network(tmp_path):
    torch = pytest.importorskip("torch")
    from tracker.appearance_data import AppearanceSample

    samples = [AppearanceSample("MOT17-02-FRCNN", frame, identity, Box(0, 0, 10, 20), tmp_path / "unused.jpg")
               for identity in (1, 2) for frame in (1, 2)]
    features = [tuple([1.0, 0.0] + [0.0] * (FEATURE_DIM - 2))] * 2 + [
        tuple([0.0, 1.0] + [0.0] * (FEATURE_DIM - 2))] * 2
    head, losses = train(samples, features, seed=17, epochs=2, pairs_per_epoch=16,
                         batch_size=8, learning_rate=1e-3, margin=0.2, device="cpu")
    assert len(losses) == 2 and all(0 <= loss < 2 for loss in losses)
    checkpoint = tmp_path / "head.pt"
    torch.save({"state_dict": head.state_dict(),
                "metadata": {"backbone": "mobilenet_v3_small", "feature_dim": FEATURE_DIM,
                             "training_scenes": ["MOT17-02-FRCNN"]}}, checkpoint)
    extractor = FeatureExtractor(checkpoint, device="cpu", pretrained=False)
    assert extractor.head is not None and extractor.head.out_features == 128


def test_checkpoint_scene_metadata_rejects_held_out_identity_training():
    split = load_split(Path(__file__).resolve().parents[1] / "configs" / "split.json")
    validate_training_scenes({"training_scenes": ["MOT17-02-FRCNN"]}, split)
    with pytest.raises(ValueError, match="non-development"):
        validate_training_scenes({"training_scenes": ["MOT17-10-FRCNN"]}, split)
    with pytest.raises(ValueError, match="no training scenes"):
        validate_training_scenes({"training_scenes": []}, split)
