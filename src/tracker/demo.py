"""Local browser demo that tracks a supplied video as frames are streamed."""

import html
import io
import json
import time
import webbrowser
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .appearance import FeatureExtractor
from .mot import SequenceInfo, read_detections
from .render import probe_video
from .split import load_split, validate_training_scenes
from .tracks import AppearanceConfig, AppearanceTracker, IoUTracker, MotionConfig, MotionTracker


class DemoSession:
    def __init__(self, video: Path, detection_path: Path, config_path: Path,
                 min_confidence: float = 0.3, max_frames: int | None = None,
                 device: str = "cpu", trails: bool = False, split_path: Path | None = None) -> None:
        from .cli import PROJECT_ROOT, _config

        if not video.is_file():
            raise FileNotFoundError(video)
        if not detection_path.is_file():
            raise FileNotFoundError(detection_path)
        width, height, fps, length = probe_video(video)
        self.video = video
        self.info = SequenceInfo(video.stem, length, width, height, fps, video.parent, "")
        self.detections = read_detections(detection_path, self.info, min_confidence)
        self.config, checkpoint = _config(config_path)
        self.config_path = config_path
        self.frame_limit = min(max_frames or length, length)
        if self.frame_limit < 1:
            raise ValueError("--max-frames must be positive")
        self.trails = trails
        self.extractor = None
        if isinstance(self.config, AppearanceConfig):
            self.extractor = FeatureExtractor(checkpoint, device=device)
            if self.extractor.metadata is not None:
                validate_training_scenes(self.extractor.metadata,
                                         load_split(split_path or PROJECT_ROOT / "configs" / "split.json"))

    def _tracker(self):
        if isinstance(self.config, AppearanceConfig):
            return AppearanceTracker(self.config)
        if isinstance(self.config, MotionConfig):
            return MotionTracker(self.config)
        return IoUTracker(self.config)

    def frames(self):
        from PIL import ImageDraw, ImageFont
        from .cli import _video_images

        tracker = self._tracker()
        trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=20))
        display_width = min(self.info.width, 1000)
        scale = display_width / self.info.width
        display_height = round(self.info.height * scale)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
        except OSError:
            font = ImageFont.load_default()
        images = _video_images(self.video, self.info)
        started = time.perf_counter()
        try:
            for frame in range(1, self.frame_limit + 1):
                image = next(images)
                detections = self.detections.get(frame, [])
                features = (self.extractor.encode_boxes(image, [item.box for item in detections])
                            if self.extractor and detections else None)
                results = tracker.step(frame, detections, features)
                if scale != 1:
                    image = image.resize((display_width, display_height))
                draw = ImageDraw.Draw(image)
                for result in results:
                    color = ((result.identity * 73) % 180 + 75,
                             (result.identity * 137) % 180 + 75,
                             (result.identity * 199) % 180 + 75)
                    box = result.box
                    x0, y0 = round(box.x * scale), round(box.y * scale)
                    x1, y1 = round((box.x + box.w) * scale), round((box.y + box.h) * scale)
                    draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
                    if self.trails:
                        center = (round((box.x + box.w / 2) * scale),
                                  round((box.y + box.h) * scale))
                        trails[result.identity].append(center)
                        if len(trails[result.identity]) > 1:
                            draw.line(list(trails[result.identity]), fill=color, width=2)
                    label = f"#{result.identity}  {result.confidence:.2f}"
                    bounds = draw.textbbox((0, 0), label, font=font)
                    label_y = max(38, y0 - (bounds[3] - bounds[1]) - 7)
                    draw.rectangle((x0, label_y, x0 + bounds[2] + 8, label_y + bounds[3] + 5), fill=color)
                    draw.text((x0 + 4, label_y + 2), label, fill="black", font=font)
                fps = frame / max(time.perf_counter() - started, 1e-9)
                draw.rectangle((0, 0, display_width, 36), fill=(16, 22, 30))
                draw.text((10, 8), f"Frame {frame}/{self.frame_limit}  |  Tracks {len(results)}  |  {fps:.1f} FPS",
                          fill="white", font=font)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=82)
                yield buffer.getvalue()
        finally:
            images.close()


def make_server(session: DemoSession, port: int = 8765) -> ThreadingHTTPServer:
    title = html.escape(session.video.name)
    config_name = html.escape(session.config_path.name)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pedestrian tracking demo</title>
<style>body{{font:16px system-ui;background:#101820;color:#f0f5f8;margin:0}}
main{{max-width:1100px;margin:30px auto;padding:0 20px}}h1{{margin-bottom:8px}}
p{{color:#b6c8d2;line-height:1.5}}button{{background:#35c58a;border:0;border-radius:8px;
padding:12px 18px;font-size:16px;font-weight:650;cursor:pointer;margin:12px 0}}
img{{display:block;width:100%;background:#060b10;border-radius:10px;min-height:220px;object-fit:contain}}
small{{color:#b6c8d2}}</style></head><body><main><h1>Pedestrian tracking live demo</h1>
<p><strong>Video:</strong> {title} &nbsp; <strong>Tracker:</strong> {config_name}</p>
<p>Frames are decoded and tracked as they stream. Colored numbers are temporary IDs within this video.</p>
<button onclick="document.getElementById('feed').src='/stream?run='+Date.now()">Start / restart</button>
<img id="feed" alt="Live annotated pedestrian tracking stream">
<p><small>Uses the supplied detection file. The stream ends after {session.frame_limit} frames;
click Start / restart to replay. Stop the server with Ctrl-C in the terminal.</small></p></main></body></html>"""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                content = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            elif path == "/health":
                content = json.dumps({"ready": True, "frames": session.frame_limit}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            elif path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                stream = session.frames()
                deadline = time.monotonic()
                try:
                    for jpeg in stream:
                        deadline += 1 / session.info.fps
                        delay = deadline - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                         str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    stream.close()
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            return

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def run_demo(session: DemoSession, port: int = 8765, open_browser: bool = True) -> None:
    server = make_server(session, port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Live demo: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
