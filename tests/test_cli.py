import shutil
from pathlib import Path

import pytest

from tracker.cli import main


def make_sequence(tmp_path: Path) -> Path:
    root = tmp_path / "MOT17-02-FRCNN"
    (root / "img1").mkdir(parents=True)
    (root / "det").mkdir()
    (root / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nframeRate=5\nseqLength=3\nimWidth=64\nimHeight=48\nimExt=.ppm\n"
    )
    for frame in range(1, 4):
        (root / "img1" / f"{frame:06d}.ppm").write_bytes(b"P6\n64 48\n255\n" + bytes([32, 32, 32]) * 64 * 48)
    (root / "det" / "det.txt").write_text(
        "1,-1,2,2,12,20,0.9,-1,-1,-1\n"
        "2,-1,3,2,12,20,0.9,-1,-1,-1\n"
        "3,-1,4,2,12,20,0.9,-1,-1,-1\n"
    )
    return root


def test_short_sequence_end_to_end(tmp_path, capsys):
    sequence = make_sequence(tmp_path)
    output = tmp_path / "outputs"
    args = ["track", "--sequence-dir", str(sequence), "--output-root", str(output)]
    if shutil.which("ffmpeg"):
        args.append("--render")
    main(args)
    result = output / "iou" / "data" / "MOT17-02-FRCNN.txt"
    lines = result.read_text().splitlines()
    assert len(lines) == 2  # min_hits=2
    assert [line.split(",")[:2] for line in lines] == [["2", "1"], ["3", "1"]]
    printed = capsys.readouterr().out
    assert "tracker FPS" in printed and "end-to-end time" in printed
    if shutil.which("ffmpeg"):
        video = output / "iou" / "MOT17-02-FRCNN.mp4"
        assert video.is_file() and video.stat().st_size > 0


def test_evaluation_fails_before_trackeval_when_result_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="MOT17-10-FRCNN"):
        main(["evaluate", "--data-root", str(tmp_path), "--trackeval-root", str(tmp_path),
              "--output-root", str(tmp_path / "outputs"), "--partition", "held_out", "--variants", "iou"])


def test_evaluation_uses_exact_held_out_seqmap(tmp_path):
    output = tmp_path / "outputs"
    for name in ("MOT17-10-FRCNN", "MOT17-11-FRCNN"):
        result = output / "iou" / "data" / f"{name}.txt"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text("")
    script = tmp_path / "TrackEval" / "scripts" / "run_mot_challenge.py"
    script.parent.mkdir(parents=True)
    script.write_text("import pathlib, sys\npathlib.Path(sys.argv[1 + sys.argv.index('--SEQMAP_FILE')]).with_suffix('.called').write_text(' '.join(sys.argv))\n")
    main(["evaluate", "--data-root", str(tmp_path / "MOT17"), "--trackeval-root", str(script.parent.parent),
          "--output-root", str(output), "--partition", "held_out", "--variants", "iou"])
    seqmap = output / "seqmaps" / "held_out.txt"
    assert seqmap.read_text().splitlines() == ["name", "MOT17-10-FRCNN", "MOT17-11-FRCNN"]
    assert "--SKIP_SPLIT_FOL True" in seqmap.with_suffix(".called").read_text()
