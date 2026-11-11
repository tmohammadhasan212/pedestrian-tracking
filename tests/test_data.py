from pathlib import Path

import pytest

from tracker.geometry import Box, iou
from tracker.mot import read_annotations, read_detections, read_sequence_info, write_results, Result
from tracker.split import create_split, load_split, sequence_names


def sample_sequence(tmp_path: Path) -> tuple[Path, object]:
    root = tmp_path / "MOT17-02-FRCNN"
    root.mkdir()
    (root / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nframeRate=30\nseqLength=3\nimWidth=100\nimHeight=80\nimExt=.jpg\n"
    )
    return root, read_sequence_info(root)


def test_geometry_round_trip_and_iou():
    a = Box.from_mot(1, 1, 10, 10)
    assert a == Box(0, 0, 10, 10)
    assert a.to_mot() == (1, 1, 10, 10)
    assert iou(a, Box(5, 0, 10, 10)) == pytest.approx(1 / 3)
    assert iou(a, Box(20, 20, 2, 2)) == 0
    with pytest.raises(ValueError):
        Box(0, 0, 0, 10)


def test_mot_reader_filtering_and_validation(tmp_path):
    root, info = sample_sequence(tmp_path)
    path = root / "det.txt"
    path.write_text("1,-1,1,1,10,10,0.9,-1,-1,-1\n2,-1,2,1,10,10,0.1,-1,-1,-1\n")
    detections = read_detections(path, info, min_confidence=0.5)
    assert list(detections) == [1]
    assert detections[1][0].box == Box(0, 0, 10, 10)
    path.write_text("2,-1,1,1,10,10,0.9,-1,-1,-1\n1,-1,1,1,10,10,0.9,-1,-1,-1\n")
    with pytest.raises(ValueError, match="out of order"):
        read_detections(path, info)
    path.write_text("1,-1,95,1,10,10,0.9,-1,-1,-1\n")
    with pytest.raises(ValueError, match="outside image"):
        read_detections(path, info)
    path.write_text("1,-1,1,1,10,10,nan,-1,-1,-1\n")
    with pytest.raises(ValueError, match="non-finite"):
        read_detections(path, info)


def test_annotations_and_result_format(tmp_path):
    root, info = sample_sequence(tmp_path)
    path = root / "gt.txt"
    path.write_text("1,7,1,1,10,10,1,1,0.8,-1\n")
    assert read_annotations(path, info)[1][0].identity == 7
    output = root / "results.txt"
    write_results(output, [Result(1, 7, Box(0, 0, 10, 10), 0.9)])
    assert output.read_text() == "1,7,1.000,1.000,10.000,10.000,0.9000,-1,-1,-1\n"


def test_scene_split_is_fixed_and_grouped():
    split = load_split(Path(__file__).resolve().parents[1] / "configs" / "split.json")
    assert split == create_split(1704)
    assert not set(split["development_scenes"]) & set(split["held_out_scenes"])
    assert sequence_names(split, "held_out") == ["MOT17-10-FRCNN", "MOT17-11-FRCNN"]
