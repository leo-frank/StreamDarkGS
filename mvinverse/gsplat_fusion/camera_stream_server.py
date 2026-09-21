from __future__ import annotations

import json
import queue
import ssl
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class StreamFrame:
    filename: str
    received_at: float


def _capture_page(capture_fps: float, jpeg_quality: int) -> bytes:
    interval_ms = max(round(1000.0 / max(capture_fps, 0.1)), 1)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>StreamDarkGS 手机采集</title>
<style>
html,body{{height:100%;overflow:hidden}}
body{{font-family:system-ui;margin:0;background:#111;color:#eee;text-align:center;display:flex;flex-direction:column}}
h2{{font-size:18px;line-height:24px;margin:4px 0 2px;flex:none}}
video{{width:100%;height:34dvh;min-height:0;background:#000;object-fit:cover;flex:none}}
.bar{{padding:2px 4px;line-height:28px;flex:none}}
button{{font-size:15px;padding:5px 12px;margin:2px}}
#status{{margin:2px;color:#9fd;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.preview-grid{{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:4px;padding:4px;min-height:0;flex:1}}
.preview-card{{position:relative;min-width:0;min-height:0;background:#222;border-radius:3px;overflow:hidden}}
.preview-card b{{position:absolute;z-index:1;top:2px;left:4px;padding:1px 4px;background:#0009;border-radius:3px;font-size:12px}}
.preview-card img{{display:block;width:100%;height:100%;background:#222;object-fit:contain}}
#prediction-frame,#light-controls{{display:none}}
</style></head><body>
<h2>StreamDarkGS 手机采集</h2><video id="video" autoplay playsinline muted></video>
<div class="bar"><button id="start">开始采集</button><button id="finish" disabled>结束并重建</button>
<div id="status">等待启动摄像头</div></div>
<div class="preview-grid">
<div class="preview-card"><b>实时点亮</b><img id="relit"></div>
<div class="preview-card"><b>地图 Albedo</b><img id="albedo"></div>
<div class="preview-card"><b>MVInverse Albedo</b><img id="mvinverse_albedo"></div>
<div class="preview-card"><b>MVInverse Normal</b><img id="mvinverse_normal"></div>
</div>
<div id="prediction-frame">等待 MVInverse 预测</div>
<div id="interactive" hidden><h3>交互式点亮浏览</h3>
<p>单指旋转 · 双指缩放/平移</p><button id="reset-view">重置视角</button>
<img id="orbit-image" style="width:100%;max-height:80vh;object-fit:contain;touch-action:none" draggable="false"></div>
<div id="light-controls" class="bar">光源 X <input id="lx" type="range" min="-1" max="1" step="0.05" value="-0.25">
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
let predictionVersion=0;
let orbitReady=false, orbitBusy=false, orbitDirty=false, orbitSerial=0;
const orbitImage=document.querySelector('#orbit-image');
let orbit={{yaw:0,pitch:0,zoom:0,pan_x:0,pan_y:0,dragging:0}};
const touches=new Map();
function gesture(){{const p=[...touches.values()];return {{x:p.reduce((s,v)=>s+v.x,0)/p.length,y:p.reduce((s,v)=>s+v.y,0)/p.length,
 distance:p.length>1?Math.hypot(p[0].x-p[1].x,p[0].y-p[1].y):0,n:p.length}};}}
orbitImage.onpointerdown=e=>{{orbitImage.setPointerCapture(e.pointerId);touches.set(e.pointerId,{{x:e.clientX,y:e.clientY}});}};
orbitImage.onpointermove=e=>{{if(!touches.has(e.pointerId))return;const a=gesture();touches.set(e.pointerId,{{x:e.clientX,y:e.clientY}});const b=gesture();
 if(b.n===1){{orbit.yaw+=(b.x-a.x)*0.006;orbit.pitch=Math.max(-1.4,Math.min(1.4,orbit.pitch+(b.y-a.y)*0.006));}}
 else{{orbit.pan_x-=(b.x-a.x)*0.002;orbit.pan_y-=(b.y-a.y)*0.002;if(a.distance>0&&b.distance>0)orbit.zoom=Math.max(-2,Math.min(2,orbit.zoom+Math.log(a.distance/b.distance)));}}
 orbit.dragging=1;orbitDirty=true;}};
function endGesture(e){{touches.delete(e.pointerId);orbit.dragging=touches.size?1:0;orbitDirty=true;}}
orbitImage.onpointerup=endGesture;orbitImage.onpointercancel=endGesture;
orbitImage.onwheel=e=>{{e.preventDefault();orbit.zoom=Math.max(-2,Math.min(2,orbit.zoom+e.deltaY*0.001));orbit.dragging=0;orbitDirty=true;}};
document.querySelector('#reset-view').onclick=()=>{{orbit={{yaw:0,pitch:0,zoom:0,pan_x:0,pan_y:0,dragging:0}};orbitDirty=true;}};
setInterval(async()=>{{if(!orbitReady||orbitBusy||!orbitDirty)return;orbitBusy=true;orbitDirty=false;
 try{{const r=await fetch('/viewer/settings',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(orbit)}});if(!r.ok)orbitDirty=true;}}
 catch(e){{orbitDirty=true;}}finally{{orbitBusy=false;}}}},60);
setInterval(async()=>{{if(!orbitReady)return;try{{const s=await(await fetch('/status')).json();if(s.preview_version===orbitSerial)return;
 orbitSerial=s.preview_version;orbitImage.src='/viewer/relit.jpg?v='+orbitSerial;}}catch(e){{}}}},100);
async function updateSettings(){{await fetch('/viewer/settings',{{method:'POST',headers:{{'Content-Type':'application/json'}},
 body:JSON.stringify({{light_x:+lx.value,light_y:+ly.value,ambient:+ambient.value}})}});}}
[lx,ly,ambient].forEach(e=>e.oninput=updateSettings);
setInterval(async()=>{{try{{const s=await (await fetch('/status')).json();
 orbitReady=!!s.interactive_ready;document.querySelector('#interactive').hidden=!orbitReady;
 if(s.preview_version!==previewVersion){{previewVersion=s.preview_version;
 albedo.src='/viewer/albedo.jpg?v='+previewVersion; relit.src='/viewer/relit.jpg?v='+previewVersion;}}
 if(s.prediction_version && s.prediction_version!==predictionVersion){{predictionVersion=s.prediction_version;
 document.querySelector('#mvinverse_normal').src='/viewer/mvinverse_normal.jpg?v='+predictionVersion;
 document.querySelector('#mvinverse_albedo').src='/viewer/mvinverse_albedo.jpg?v='+predictionVersion;
 document.querySelector('#prediction-frame').textContent='MVInverse 帧：'+s.prediction_frame;}}
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
        self._queue: queue.Queue[StreamFrame] = queue.Queue(
            maxsize=max(int(queue_size), 1)
        )
        self._finished = threading.Event()
        self._counter = 0
        self._counter_lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._previews: dict[str, bytes] = {}
        self._preview_version = 0
        self._preview_metadata: dict[int, dict] = {}
        self._prediction_version = 0
        self._prediction_frame = ""
        self.interactive_ready = False
        self._settings_version = 0
        self._viewer_settings = {"light_x": -0.25, "light_y": -0.35, "ambient": 0.05}
        self._page = _capture_page(capture_fps, jpeg_quality)
        self.certfile = certfile
        self.keyfile = keyfile
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, status: int, body: bytes, content_type: str, headers=None) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for key, value in (headers or {}).items():
                    self.send_header(key, str(value))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                route = self.path.split("?", 1)[0]
                if route == "/":
                    self._reply(HTTPStatus.OK, owner._page, "text/html; charset=utf-8")
                elif route == "/profile":
                    page = Path(__file__).with_name("profile_viewer.html").read_bytes()
                    self._reply(HTTPStatus.OK, page, "text/html; charset=utf-8")
                elif route == "/status":
                    body = json.dumps(
                        {"received": owner._counter, "finished": owner._finished.is_set(),
                         "preview_version": owner._preview_version,
                         "prediction_version": owner._prediction_version,
                         "prediction_frame": owner._prediction_frame,
                         "interactive_ready": owner.interactive_ready,
                         "queue_depth": owner.queue_depth()}
                    ).encode("utf-8")
                    self._reply(HTTPStatus.OK, body, "application/json")
                elif route in {"/viewer/albedo.jpg", "/viewer/relit.jpg",
                               "/viewer/mvinverse_normal.jpg", "/viewer/mvinverse_albedo.jpg"}:
                    key = route.rsplit("/", 1)[1][:-4]
                    with owner._preview_lock:
                        preview = owner._previews.get(key)
                        preview_version = owner._preview_version
                    if preview is None:
                        self._reply(HTTPStatus.NOT_FOUND, b"preview not ready", "text/plain")
                    else:
                        self._reply(HTTPStatus.OK, preview, "image/jpeg", {"X-Preview-Version": preview_version})
                else:
                    self._reply(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

            def do_POST(self) -> None:  # noqa: N802
                if self.path == "/profile/ack":
                    try:
                        import math
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 4096:
                            raise ValueError("invalid record size")
                        values = json.loads(self.rfile.read(length))
                        version = int(values["version"])
                        timings = {key: float(values[key]) for key in
                                   ("fetch_ms", "decode_ms", "draw_wait_ms", "fetch_to_draw_ms", "poll_ms")}
                        if not all(math.isfinite(v) and v >= 0 for v in timings.values()):
                            raise ValueError("invalid duration")
                    except (ValueError, TypeError, KeyError):
                        self._reply(HTTPStatus.BAD_REQUEST, b"invalid timing", "text/plain")
                        return
                    with owner._preview_lock:
                        meta = dict(owner._preview_metadata.get(version, {}))
                    now = time.perf_counter()
                    # Same server clock, including the ACK's return trip. These
                    # are upper bounds to display latency, not cross-clock deltas.
                    for name, key in (("publish_to_ack_ms", "published_at"),
                                      ("receive_to_ack_ms", "received_at")):
                        if meta.get(key) is not None:
                            timings[name] = (now - meta[key]) * 1000
                    fields = " ".join(f"{k}={v:.3f}" for k, v in timings.items())
                    print(f"[viewer-client-timing] version={version} frame={meta.get('frame', 'unknown')} {fields}", flush=True)
                    self._reply(HTTPStatus.OK, b"ok", "text/plain")
                    return
                if self.path == "/viewer/settings":
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        values = json.loads(self.rfile.read(length))
                        with owner._preview_lock:
                            for key in ("light_x", "light_y", "ambient", "yaw", "pitch", "zoom", "pan_x", "pan_y", "dragging"):
                                if key in values:
                                    value = float(values[key])
                                    if not (-100 <= value <= 100):
                                        raise ValueError("settings out of range")
                                    owner._viewer_settings[key] = value
                            owner._settings_version += 1
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
                    owner._queue.put_nowait(StreamFrame(name, time.perf_counter()))
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

    def frames(self) -> Iterator[StreamFrame]:
        while True:
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._finished.is_set() and self._queue.empty():
                    return

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def viewer_settings(self) -> dict[str, float]:
        with self._preview_lock:
            return dict(self._viewer_settings, revision=self._settings_version)

    def publish_previews(self, *, albedo_jpeg: bytes, relit_jpeg: bytes, metadata=None) -> None:
        with self._preview_lock:
            self._previews.update({"albedo": albedo_jpeg, "relit": relit_jpeg})
            self._preview_version += 1
            self._preview_metadata[self._preview_version] = dict(metadata or {}, published_at=time.perf_counter())
            if len(self._preview_metadata) > 512:
                self._preview_metadata.pop(next(iter(self._preview_metadata)))

    def publish_predictions(self, *, frame: str, normal_jpeg: bytes, albedo_jpeg: bytes) -> None:
        with self._preview_lock:
            self._previews.update({"mvinverse_normal": normal_jpeg,
                                   "mvinverse_albedo": albedo_jpeg})
            self._prediction_frame = frame
            self._prediction_version += 1

    def close(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
