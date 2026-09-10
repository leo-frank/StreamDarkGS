from __future__ import annotations

import json
import queue
import ssl
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def _capture_page(capture_fps: float, jpeg_quality: int) -> bytes:
    interval_ms = max(round(1000.0 / max(capture_fps, 0.1)), 1)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>StreamDarkGS 手机采集</title>
<style>
body{{font-family:system-ui;margin:0;background:#111;color:#eee;text-align:center}}
video{{width:100%;max-height:68vh;background:#000}}
.viewer{{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:10px}}
.viewer img{{width:100%;background:#222;min-height:120px;object-fit:contain}}
.bar{{padding:14px}}button{{font-size:18px;padding:12px 22px;margin:6px}}
#status{{margin:8px;color:#9fd}}
</style></head><body>
<h2>StreamDarkGS 手机采集</h2><video id="video" autoplay playsinline muted></video>
<div class="bar"><button id="start">开始采集</button><button id="finish" disabled>结束并重建</button>
<div id="status">等待启动摄像头</div></div>
<div class="viewer"><div><b>最新 Albedo</b><img id="albedo"></div>
<div><b>实时点亮</b><img id="relit"></div></div>
<div class="bar">光源 X <input id="lx" type="range" min="-1" max="1" step="0.05" value="-0.25">
光源 Y <input id="ly" type="range" min="-1" max="1" step="0.05" value="-0.35">
环境光 <input id="ambient" type="range" min="0" max="0.5" step="0.01" value="0.05"></div>
<canvas id="canvas" hidden></canvas>
<script>
const video=document.querySelector('#video'), canvas=document.querySelector('#canvas');
const statusEl=document.querySelector('#status'), startBtn=document.querySelector('#start');
const finishBtn=document.querySelector('#finish'), lx=document.querySelector('#lx');
const ly=document.querySelector('#ly'), ambient=document.querySelector('#ambient');
const albedo=document.querySelector('#albedo'), relit=document.querySelector('#relit');
let timer=null, busy=false, sent=0;
let previewVersion=0;
async function updateSettings(){{await fetch('/viewer/settings',{{method:'POST',headers:{{'Content-Type':'application/json'}},
 body:JSON.stringify({{light_x:+lx.value,light_y:+ly.value,ambient:+ambient.value}})}});}}
[lx,ly,ambient].forEach(e=>e.oninput=updateSettings);
setInterval(async()=>{{try{{const s=await (await fetch('/status')).json();
 if(s.preview_version!==previewVersion){{previewVersion=s.preview_version;
 albedo.src='/viewer/albedo.jpg?v='+previewVersion; relit.src='/viewer/relit.jpg?v='+previewVersion;}}
 }}catch(e){{}}}},1000);
async function sendFrame(){{
  if(busy||!video.videoWidth)return; busy=true;
  canvas.width=video.videoWidth; canvas.height=video.videoHeight;
  canvas.getContext('2d').drawImage(video,0,0);
  const blob=await new Promise(r=>canvas.toBlob(r,'image/jpeg',{jpeg_quality / 100.0:.2f}));
  try{{const r=await fetch('/frame',{{method:'POST',headers:{{'Content-Type':'image/jpeg'}},body:blob}});
    if(!r.ok)throw new Error(await r.text()); sent++; statusEl.textContent=`已发送 ${{sent}} 帧`;
  }}catch(e){{statusEl.textContent='上传失败：'+e.message;}}finally{{busy=false;}}
}}
startBtn.onclick=async()=>{{
 try{{const stream=await navigator.mediaDevices.getUserMedia({{video:{{facingMode:{{ideal:'environment'}}}},audio:false}});
 video.srcObject=stream; await video.play(); timer=setInterval(sendFrame,{interval_ms});
 startBtn.disabled=true; finishBtn.disabled=false; statusEl.textContent='采集中';
 }}catch(e){{statusEl.textContent='无法打开摄像头：'+e.message;}}
}};
finishBtn.onclick=async()=>{{clearInterval(timer); finishBtn.disabled=true;
 if(video.srcObject)video.srcObject.getTracks().forEach(t=>t.stop());
 await fetch('/finish',{{method:'POST'}}); statusEl.textContent=`采集结束，共发送 ${{sent}} 帧，服务器正在处理尾帧`;
}};
</script></body></html>""".encode("utf-8")


class CameraStreamServer:
    """Receive JPEG frames over HTTP and expose saved filenames as an iterator."""

    def __init__(
        self,
        image_dir: str | Path,
        *,
        host: str,
        port: int,
        queue_size: int,
        max_frame_bytes: int,
        capture_fps: float,
        jpeg_quality: int,
        certfile: str = "",
        keyfile: str = "",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.port = int(port)
        self.max_frame_bytes = max(int(max_frame_bytes), 1024)
        self._queue: queue.Queue[str] = queue.Queue(maxsize=max(int(queue_size), 1))
        self._finished = threading.Event()
        self._counter = 0
        self._counter_lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._previews: dict[str, bytes] = {}
        self._preview_version = 0
        self._viewer_settings = {"light_x": -0.25, "light_y": -0.35, "ambient": 0.05}
        self._page = _capture_page(capture_fps, jpeg_quality)
        self.certfile = certfile
        self.keyfile = keyfile
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                route = self.path.split("?", 1)[0]
                if route == "/":
                    self._reply(HTTPStatus.OK, owner._page, "text/html; charset=utf-8")
                elif route == "/status":
                    body = json.dumps(
                        {"received": owner._counter, "finished": owner._finished.is_set(),
                         "preview_version": owner._preview_version}
                    ).encode("utf-8")
                    self._reply(HTTPStatus.OK, body, "application/json")
                elif route in {"/viewer/albedo.jpg", "/viewer/relit.jpg"}:
                    key = "albedo" if "albedo" in route else "relit"
                    with owner._preview_lock:
                        preview = owner._previews.get(key)
                    if preview is None:
                        self._reply(HTTPStatus.NOT_FOUND, b"preview not ready", "text/plain")
                    else:
                        self._reply(HTTPStatus.OK, preview, "image/jpeg")
                else:
                    self._reply(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

            def do_POST(self) -> None:  # noqa: N802
                if self.path == "/viewer/settings":
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        values = json.loads(self.rfile.read(length))
                        with owner._preview_lock:
                            for key in ("light_x", "light_y", "ambient"):
                                if key in values:
                                    owner._viewer_settings[key] = float(values[key])
                    except (ValueError, TypeError, json.JSONDecodeError):
                        self._reply(HTTPStatus.BAD_REQUEST, b"invalid settings", "text/plain")
                        return
                    self._reply(HTTPStatus.OK, b"ok", "text/plain")
                    return
                if self.path == "/finish":
                    owner._finished.set()
                    self._reply(HTTPStatus.OK, b"finishing", "text/plain")
                    return
                if self.path != "/frame":
                    self._reply(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                    return
                if owner._finished.is_set():
                    self._reply(HTTPStatus.CONFLICT, b"stream already finished", "text/plain")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if length <= 0 or length > owner.max_frame_bytes:
                    self._reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, b"invalid frame size", "text/plain")
                    return
                payload = self.rfile.read(length)
                decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    self._reply(HTTPStatus.BAD_REQUEST, b"invalid JPEG", "text/plain")
                    return
                with owner._counter_lock:
                    index = owner._counter
                    owner._counter += 1
                name = f"stream_{time.time_ns()}_{index:08d}.jpg"
                path = owner.image_dir / name
                temporary = path.with_suffix(".jpg.tmp")
                temporary.write_bytes(payload)
                temporary.replace(path)
                try:
                    owner._queue.put_nowait(name)
                except queue.Full:
                    path.unlink(missing_ok=True)
                    self._reply(HTTPStatus.TOO_MANY_REQUESTS, b"server input queue is full", "text/plain")
                    return
                self._reply(HTTPStatus.OK, name.encode("utf-8"), "text/plain")

            def log_message(self, format: str, *args: object) -> None:
                return

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        if self.certfile:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self.certfile, self.keyfile or None)
            self._httpd.socket = context.wrap_socket(self._httpd.socket, server_side=True)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def frames(self) -> Iterator[str]:
        while True:
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._finished.is_set() and self._queue.empty():
                    return

    def viewer_settings(self) -> dict[str, float]:
        with self._preview_lock:
            return dict(self._viewer_settings)

    def publish_previews(self, *, albedo_jpeg: bytes, relit_jpeg: bytes) -> None:
        with self._preview_lock:
            self._previews = {"albedo": albedo_jpeg, "relit": relit_jpeg}
            self._preview_version += 1

    def close(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
