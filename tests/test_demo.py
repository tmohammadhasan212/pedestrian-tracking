import json
import shutil
import subprocess
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest

from scripts.prepare_demo_video import prepare
from tracker.demo import DemoSession, make_server
from tracker.render import probe_video


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
def test_supplied_video_streams_annotated_frames(tmp_path: Path):
    video = tmp_path / "short.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=gray:s=64x48:r=5:d=0.6", "-frames:v", "3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    detections = tmp_path / "detections.txt"
    detections.write_text("1,-1,2,2,12,20,0.9\n2,-1,3,2,12,20,0.9\n3,-1,4,2,12,20,0.9\n")
    config = tmp_path / "motion.json"
    config.write_text(json.dumps({"min_iou": 0.3, "min_hits": 2, "max_missed": 15,
                                  "duplicate_iou": 0.9, "gate_threshold": 9.488}))
    session = DemoSession(video, detections, config, trails=True)
    frames = list(session.frames())
    assert len(frames) == 3
    assert all(frame.startswith(b"\xff\xd8") for frame in frames)
    assert frames[0] != frames[1]

    try:
        server = make_server(session, 0)
    except PermissionError:
        pytest.skip("local sockets are unavailable in this environment")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with urlopen(url + "/", timeout=5) as response:
            assert b"Pedestrian tracking live demo" in response.read()
        with urlopen(url + "/health", timeout=5) as response:
            assert json.load(response) == {"ready": True, "frames": 3}
        with urlopen(url + "/stream", timeout=5) as response:
            assert response.headers["Content-Type"].startswith("multipart/x-mixed-replace")
            assert response.read(128).startswith(b"--frame\r\nContent-Type: image/jpeg")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
def test_prepare_demo_video_keeps_detections_in_clip(tmp_path: Path):
    sequence = tmp_path / "MOT17-02-FRCNN"
    (sequence / "img1").mkdir(parents=True)
    (sequence / "det").mkdir()
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nframeRate=5\n"
        "seqLength=4\nimWidth=64\nimHeight=48\nimExt=.ppm\n"
    )
    for frame in range(1, 5):
        (sequence / "img1" / f"{frame:06d}.ppm").write_bytes(
            b"P6\n64 48\n255\n" + bytes([frame * 20, 32, 32]) * 64 * 48
        )
    (sequence / "det" / "det.txt").write_text(
        "1,-1,2,2,12,20,0.9\n2,-1,3,2,12,20,0.9\n"
        "3,-1,4,2,12,20,0.9\n4,-1,5,2,12,20,0.9\n"
    )
    video, detections = prepare(sequence, 3, tmp_path / "assets")
    assert probe_video(video)[3] == 3
    assert [line.split(",")[0] for line in detections.read_text().splitlines()] == ["1", "2", "3"]
    config = tmp_path / "motion.json"
    config.write_text(json.dumps({"gate_threshold": 9.488}))
    assert len(list(DemoSession(video, detections, config).frames())) == 3
