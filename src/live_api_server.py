r"""Live API server for MJPEG stream and WebSocket events.

Run:
    .\venv\Scripts\python.exe .\src\live_api_server.py --source 0 --encodings .\models\known_faces.pkl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import face_recognition
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from ultralytics import YOLO

from liveness import AdaptiveLiveness, BlinkLiveness, SimpleFaceTracker, mean_ear


Box = Tuple[int, int, int, int]


def load_known(path: Path):
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("encodings", []), data.get("names", [])


def identify(encodings, names, face_enc, thr=0.58, margin=0.0):
    if len(encodings) == 0:
        return "necunoscut", 1.0
    d = face_recognition.face_distance(encodings, face_enc)
    i = int(np.argmin(d))
    best_name = names[i]
    best_dist = float(d[i])
    if best_dist >= thr:
        return "necunoscut", best_dist

    if margin > 0:
        other = [float(di) for di, n in zip(d, names) if n != best_name]
        if other:
            second_best_other = min(other)
            if (second_best_other - best_dist) < margin:
                return "necunoscut", best_dist

    return best_name, best_dist


@dataclass
class PersonEvent:
    type: str
    ts: float
    name: str
    dist: float


class Hub:
    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: Dict):
        async with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        msg = json.dumps(payload, ensure_ascii=False)
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                # client is probably gone; let handler clean it up
                pass


class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self.jpeg_bytes: Optional[bytes] = None
        self.ts: float = 0.0

    def set(self, jpg: bytes) -> None:
        with self._lock:
            self.jpeg_bytes = jpg
            self.ts = time.time()

    def get(self) -> Optional[bytes]:
        with self._lock:
            return self.jpeg_bytes

    def get_age_s(self) -> Optional[float]:
        with self._lock:
            if self.jpeg_bytes is None or self.ts <= 0:
                return None
            return time.time() - self.ts


class LatestEvent:
    def __init__(self):
        self._lock = threading.Lock()
        self.payload: Optional[Dict] = None
        self.ts: float = 0.0

    def set(self, payload: Dict) -> None:
        with self._lock:
            self.payload = payload
            self.ts = time.time()

    def get(self) -> Optional[Dict]:
        with self._lock:
            return self.payload


class PushNotifier:
    def __init__(self, service_account_path: str, registry_path: str, title_prefix: str = "FaceClient"):
        self._lock = threading.Lock()
        self._registry_path = Path(registry_path)
        self._title_prefix = title_prefix
        self._tokens: Set[str] = set()
        self._enabled = False
        self._messaging = None

        self._load_tokens()

        service_account_path = (service_account_path or "").strip()
        if not service_account_path:
            print("Push disabled: --fcm-service-account not provided")
            return

        sa_path = Path(service_account_path)
        if not sa_path.exists():
            print(f"Push disabled: FCM service account file not found: {sa_path}")
            return

        try:
            import firebase_admin  # pyright: ignore[reportMissingImports]
            from firebase_admin import credentials, messaging  # pyright: ignore[reportMissingImports]

            if not firebase_admin._apps:
                firebase_admin.initialize_app(credentials.Certificate(str(sa_path)))
            self._messaging = messaging
            self._enabled = True
            print(f"Push enabled: loaded Firebase service account from {sa_path}")
        except Exception as e:
            print(f"Push disabled: Firebase init failed: {e}")

    def _load_tokens(self) -> None:
        with self._lock:
            try:
                if not self._registry_path.exists():
                    return
                raw = json.loads(self._registry_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    vals = raw.get("tokens", [])
                elif isinstance(raw, list):
                    vals = raw
                else:
                    vals = []
                self._tokens = {str(t).strip() for t in vals if str(t).strip()}
            except Exception:
                self._tokens = set()

    def _save_tokens(self) -> None:
        with self._lock:
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"tokens": sorted(self._tokens)}
            self._registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def count(self) -> int:
        with self._lock:
            return len(self._tokens)

    def is_enabled(self) -> bool:
        return self._enabled

    def state(self) -> Dict:
        return {
            "enabled": self._enabled,
            "devices": self.count(),
            "registry": str(self._registry_path),
        }

    def register(self, token: str) -> bool:
        tok = (token or "").strip()
        if not tok:
            return False
        with self._lock:
            before = len(self._tokens)
            self._tokens.add(tok)
            changed = len(self._tokens) != before
        if changed:
            self._save_tokens()
        return changed

    def unregister(self, token: str) -> bool:
        tok = (token or "").strip()
        if not tok:
            return False
        removed = False
        with self._lock:
            if tok in self._tokens:
                self._tokens.remove(tok)
                removed = True
        if removed:
            self._save_tokens()
        return removed

    async def send_person_event(self, name: str, dist: float, ts: float) -> Dict:
        if not self._enabled:
            return {"ok": False, "reason": "push_disabled", "sent": 0, "invalid_removed": 0, "devices": self.count()}
        return await asyncio.to_thread(self._send_person_event_sync, name, dist, ts)

    async def send_test(self, title: str, body: str) -> Dict:
        if not self._enabled:
            return {"ok": False, "reason": "push_disabled"}
        return await asyncio.to_thread(self._send_test_sync, title, body)

    def _send_person_event_sync(self, name: str, dist: float, ts: float) -> Dict:
        if not self._enabled or self._messaging is None:
            return {"ok": False, "reason": "push_disabled", "sent": 0, "invalid_removed": 0, "devices": self.count()}

        with self._lock:
            tokens = list(self._tokens)
        if not tokens:
            return {"ok": False, "reason": "no_devices", "sent": 0, "invalid_removed": 0, "devices": 0}

        invalid: Set[str] = set()
        sent = 0
        for token in tokens:
            msg = self._messaging.Message(
                token=token,
                notification=self._messaging.Notification(
                    title=f"{self._title_prefix}: {name}",
                    body=f"Detected LIVE ({dist:.2f})",
                ),
                data={
                    "type": "person_detected",
                    "name": name,
                    "dist": f"{dist:.3f}",
                    "ts": f"{ts:.3f}",
                },
                android=self._messaging.AndroidConfig(priority="high"),
            )
            try:
                self._messaging.send(msg)
                sent += 1
            except Exception as e:
                text = str(e)
                print(f"Push send failed for token (person_event): {text}")
                if "registration token" in text.lower() or "requested entity was not found" in text.lower():
                    invalid.add(token)

        if invalid:
            with self._lock:
                self._tokens = {t for t in self._tokens if t not in invalid}
            self._save_tokens()
        return {
            "ok": sent > 0,
            "reason": "sent" if sent > 0 else "send_failed",
            "sent": sent,
            "invalid_removed": len(invalid),
            "devices": self.count(),
        }

    def _send_test_sync(self, title: str, body: str) -> Dict:
        with self._lock:
            tokens = list(self._tokens)
        if not tokens:
            return {"ok": False, "reason": "no_devices"}

        invalid: Set[str] = set()
        sent = 0
        for token in tokens:
            msg = self._messaging.Message(
                token=token,
                notification=self._messaging.Notification(title=title, body=body),
                data={"type": "push_test"},
                android=self._messaging.AndroidConfig(priority="high"),
            )
            try:
                self._messaging.send(msg)
                sent += 1
            except Exception as e:
                text = str(e)
                print(f"Push send failed for token (push_test): {text}")
                if "registration token" in text.lower() or "requested entity was not found" in text.lower():
                    invalid.add(token)

        if invalid:
            with self._lock:
                self._tokens = {t for t in self._tokens if t not in invalid}
            self._save_tokens()

        return {"ok": sent > 0, "sent": sent, "invalid_removed": len(invalid), "devices": self.count()}


def build_app(args) -> FastAPI:
    app = FastAPI(title="Live Face ID Stream", version="1.0")
    hub = Hub()
    latest = LatestFrame()
    latest_event = LatestEvent()
    push = PushNotifier(args.fcm_service_account, args.device_registry, args.push_title_prefix)
    event_q: "queue.Queue[PersonEvent]" = queue.Queue(maxsize=200)

    stop_event = threading.Event()
    det_thread: Optional[threading.Thread] = None
    dispatcher_task: Optional[asyncio.Task] = None

    @dataclass
    class PresenceState:
        last_seen_ts: float
        last_notified_ts: float

    presence: Dict[str, PresenceState] = {}

    @app.get("/")
    async def index():
        html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Stream</title>
  <style>
    body { font-family: system-ui, Arial; margin: 12px; }
    img { width: 100%; max-width: 900px; border-radius: 10px; border: 1px solid #ddd; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .card { padding: 10px 12px; border: 1px solid #eee; border-radius: 10px; }
    #log { max-height: 220px; overflow: auto; font-family: ui-monospace, monospace; font-size: 12px; }
  </style>
</head>
<body>
  <h2>Live Stream + Events</h2>
  <div class="row">
    <div class="card">
      <div><b>Stream</b></div>
      <img src="/stream.mjpg" />
    </div>
    <div class="card" style="min-width: 320px; flex: 1;">
      <div><b>Events</b></div>
      <div id="log"></div>
    </div>
  </div>

<script>
  const log = document.getElementById('log');
  function addLine(s){
    const d = document.createElement('div');
    d.textContent = s;
    log.prepend(d);
  }
  const wsProto = (location.protocol === 'https:') ? 'wss' : 'ws';
  const ws = new WebSocket(`${wsProto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    try {
      const obj = JSON.parse(ev.data);
            if (obj.type === 'person_detected') {
                addLine(`${new Date(obj.ts*1000).toLocaleTimeString()} | ${obj.name} | ${obj.dist.toFixed(2)}`);
            }
    } catch(e) {}
  };
</script>
</body>
</html>"""
        return HTMLResponse(html)

    def _token_ok(token: Optional[str]) -> bool:
        configured = (args.auth_token or "").strip()
        if not configured:
            return True
        return (token or "") == configured

    def _require_token_http(request: Request) -> Optional[Response]:
        if _token_ok(request.query_params.get("token")):
            return None
        return Response(status_code=401, content=b"Unauthorized")

    @app.get("/snapshot.jpg")
    async def snapshot(request: Request):
        deny = _require_token_http(request)
        if deny is not None:
            return deny
        jpg = latest.get()
        if jpg is None:
            return Response(status_code=503)
        return Response(content=jpg, media_type="image/jpeg")

    @app.get("/health")
    async def health():
        age = latest.get_age_s()
        return {
            "ok": True,
            "has_frame": age is not None,
            "frame_age_s": age,
        }

    @app.get("/last_event")
    async def last_event():
        payload = latest_event.get()
        if payload is None:
            return Response(status_code=204)
        return Response(content=json.dumps(payload, ensure_ascii=False), media_type="application/json")

    @app.post("/register_device")
    async def register_device(request: Request):
        deny = _require_token_http(request)
        if deny is not None:
            return deny
        try:
            payload = await request.json()
        except Exception:
            return Response(status_code=400, content=b"Invalid JSON")

        fcm_token = str(payload.get("fcm_token", "")).strip()
        if not fcm_token:
            return Response(status_code=400, content=b"Missing fcm_token")

        changed = push.register(fcm_token)
        out = {
            "ok": True,
            "registered": True,
            "new": changed,
            "devices": push.count(),
        }
        return Response(content=json.dumps(out), media_type="application/json")

    @app.post("/unregister_device")
    async def unregister_device(request: Request):
        deny = _require_token_http(request)
        if deny is not None:
            return deny
        try:
            payload = await request.json()
        except Exception:
            return Response(status_code=400, content=b"Invalid JSON")

        fcm_token = str(payload.get("fcm_token", "")).strip()
        if not fcm_token:
            return Response(status_code=400, content=b"Missing fcm_token")

        removed = push.unregister(fcm_token)
        out = {
            "ok": True,
            "removed": removed,
            "devices": push.count(),
        }
        return Response(content=json.dumps(out), media_type="application/json")

    @app.get("/push_status")
    async def push_status(request: Request):
        deny = _require_token_http(request)
        if deny is not None:
            return deny
        return Response(content=json.dumps({"ok": True, **push.state()}, ensure_ascii=False), media_type="application/json")

    @app.post("/push_test")
    async def push_test(request: Request):
        deny = _require_token_http(request)
        if deny is not None:
            return deny
        result = await push.send_test("FaceClient test", "Test push from server")
        return Response(content=json.dumps(result, ensure_ascii=False), media_type="application/json")

    @app.get("/stream.mjpg")
    async def stream(request: Request):
        deny = _require_token_http(request)
        if deny is not None:
            return deny
        async def gen():
            boundary = b"frame"
            min_dt = 1.0 / float(args.max_stream_fps) if args.max_stream_fps and args.max_stream_fps > 0 else 0.0
            last_sent = 0.0
            while True:
                jpg = latest.get()
                if jpg is None:
                    await asyncio.sleep(0.05)
                    continue

                if min_dt > 0:
                    now = time.time()
                    dt = now - last_sent
                    if dt < min_dt:
                        await asyncio.sleep(min_dt - dt)
                    last_sent = time.time()

                yield b"--" + boundary + b"\r\n"
                yield b"Content-Type: image/jpeg\r\n"
                yield f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                yield jpg
                yield b"\r\n"

                # Yield control even if FPS cap is disabled
                await asyncio.sleep(0)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        token = ws.query_params.get("token")
        if not _token_ok(token):
            await ws.accept()
            await ws.close(code=1008)
            return

        await hub.connect(ws)
        try:
            while True:
                # Keep alive; clients may send pings/anything
                await ws.receive_text()
        except WebSocketDisconnect:
            await hub.disconnect(ws)
        except Exception:
            await hub.disconnect(ws)

    async def dispatcher():
        while not stop_event.is_set():
            try:
                ev = await asyncio.to_thread(event_q.get, True, 0.5)
            except queue.Empty:
                continue
            try:
                latest_event.set(asdict(ev))
                await hub.broadcast(asdict(ev))
                push_result = await push.send_person_event(ev.name, ev.dist, ev.ts)
                print(
                    "PUSH status: "
                    f"reason={push_result.get('reason')} "
                    f"sent={push_result.get('sent')} "
                    f"devices={push_result.get('devices')} "
                    f"invalid_removed={push_result.get('invalid_removed')}"
                )
            except Exception as e:
                print(f"Dispatcher error: {e}")

    def detection_loop():
        # Disable Ultralytics anonymous analytics thread.
        # Without this, Ctrl+C can race interpreter shutdown and raise:
        # "RuntimeError: can't create new thread at interpreter shutdown".
        try:
            from ultralytics.utils.events import events as ul_events

            ul_events.enabled = False
        except Exception:
            pass

        encs, names = load_known(Path(args.encodings))
        try:
            uniq = len(set(names))
        except Exception:
            uniq = 0
        print(f"Loaded encodings: {len(encs)} (persons: {uniq}) from {args.encodings}")
        if len(encs) == 0:
            print("No known encodings loaded. Events will be mostly 'necunoscut'.")
        model = YOLO(args.yolo)

        cap_src = 0 if args.source == "0" else args.source
        cap = cv2.VideoCapture(cap_src)
        if not cap.isOpened():
            print("Cannot open source")
            return

        tracker = SimpleFaceTracker(max_center_dist_px=90.0, max_missing_s=1.5)
        if args.liveness_algo == "v2":
            liveness = AdaptiveLiveness(
                grace_seconds=args.liveness_grace,
                live_ttl_seconds=args.liveness_ttl,
                min_blinks=args.liveness_min_blinks,
                min_actions=args.liveness_min_actions,
                head_turn_range=args.head_turn_range,
                min_ear_samples_before_spoof=args.liveness_min_samples,
            )
        else:
            liveness = BlinkLiveness(
                ear_threshold=args.ear_thr,
                min_closed_seconds=args.blink_min_close,
                grace_seconds=args.liveness_grace,
                live_ttl_seconds=args.liveness_ttl,
                min_blinks=args.liveness_min_blinks,
                min_actions=args.liveness_min_actions,
                head_turn_range=args.head_turn_range,
                min_ear_samples_before_spoof=args.liveness_min_samples,
            )

        last_emit: Dict[str, float] = {}

        frame_i = 0
        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                frame_i += 1
                if args.frame_skip > 1 and (frame_i % args.frame_skip) != 0:
                    continue

                # Optional downscale to reduce CPU load (YOLO + face_recognition + JPEG)
                if args.max_width and args.max_width > 0:
                    h0, w0 = frame.shape[:2]
                    if w0 > args.max_width:
                        scale = float(args.max_width) / float(w0)
                        frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

                try:
                    yolo_res = model(frame, conf=args.conf, verbose=False)[0]
                except RuntimeError as e:
                    # If we're stopping, exit quietly (Windows Ctrl+C / interpreter shutdown race)
                    if stop_event.is_set() or "interpreter shutdown" in str(e).lower():
                        break
                    raise

                boxes = []
                if yolo_res.boxes is not None:
                    for b in yolo_res.boxes:
                        cls_id = int(b.cls[0]) if hasattr(b.cls[0], "item") is False else int(b.cls[0].item())
                        if cls_id != 0:
                            continue
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        boxes.append((x1, y1, x2, y2))

                now = time.time()
                for (x1, y1, x2, y2) in boxes:
                    crop = frame[max(0, y1) : y2, max(0, x1) : x2]
                    if crop.size == 0:
                        continue

                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb, model="hog")
                    face_encs = face_recognition.face_encodings(rgb, locs)

                    for (top, right, bottom, left), fe in zip(locs, face_encs):
                        name, dist = identify(encs, names, fe, thr=args.thr, margin=args.margin)

                        fx1, fy1, fx2, fy2 = (x1 + left, y1 + top, x1 + right, y1 + bottom)

                        live_label = "OFF"
                        if args.liveness:
                            tid = tracker.assign((fx1, fy1, fx2, fy2), now)
                            st = tracker.get_state(tid)
                            if st is not None:
                                lm_list = face_recognition.face_landmarks(rgb, [(top, right, bottom, left)])
                                ear = mean_ear(lm_list[0]) if lm_list else None
                                live_label = liveness.update(
                                    st,
                                    ear,
                                    lm_list[0] if lm_list else None,
                                    now,
                                    box=(fx1, fy1, fx2, fy2),
                                )

                        # Draw overlay
                        color = (0, 255, 0)
                        if live_label == "SPOOF":
                            color = (0, 0, 255)
                        elif live_label == "CHECKING":
                            color = (0, 200, 255)

                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
                        text = f"{name} | {live_label} | {dist:.2f}" if live_label != "SPOOF" else f"SPOOF | {live_label}"
                        cv2.putText(frame, text, (fx1, max(12, fy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # Emit events (debounced)
                        # Events/notifications should only be emitted for real persons.
                        # With liveness enabled, only emit when status is LIVE (skip CHECKING + SPOOF).
                        liveness_ok = True
                        if args.liveness and args.notify_only_live:
                            liveness_ok = (live_label == "LIVE")
                        elif args.liveness:
                            liveness_ok = (live_label != "SPOOF")

                        eligible = liveness_ok and (args.emit_unknown or name != "necunoscut")
                        if eligible:
                            key = f"{name}"
                            st = presence.get(key)
                            missing_gap = (now - st.last_seen_ts) if st is not None else 1e9
                            reappeared = st is None or missing_gap >= args.realert_after_missing

                            # Update presence (seen now)
                            if st is None:
                                st = PresenceState(last_seen_ts=now, last_notified_ts=0.0)
                                presence[key] = st
                            else:
                                st.last_seen_ts = now

                            if reappeared:
                                prev_emit = last_emit.get(key, 0.0)
                                if (now - prev_emit) >= args.emit_cooldown:
                                    last_emit[key] = now
                                    st.last_notified_ts = now
                                    ev = PersonEvent(
                                        type="person_detected",
                                        ts=now,
                                        name=name,
                                        dist=float(dist),
                                    )
                                    try:
                                        event_q.put_nowait(ev)
                                        print(f"EVENT emitted: name={name} dist={dist:.3f} liveness={live_label}")
                                    except queue.Full:
                                        pass

                # Garbage-collect presence entries that haven't been seen in a long time
                if len(presence) > 0:
                    to_del = []
                    for k, st in presence.items():
                        if (now - st.last_seen_ts) > max(10.0, args.realert_after_missing * 3.0):
                            to_del.append(k)
                    for k in to_del:
                        presence.pop(k, None)

                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                if ok2:
                    latest.set(buf.tobytes())
        finally:
            try:
                cap.release()
            except Exception:
                pass

    @app.on_event("startup")
    async def on_startup():
        nonlocal det_thread, dispatcher_task
        det_thread = threading.Thread(target=detection_loop, daemon=True)
        det_thread.start()
        dispatcher_task = asyncio.create_task(dispatcher())

    @app.on_event("shutdown")
    async def on_shutdown():
        stop_event.set()
        if dispatcher_task is not None:
            dispatcher_task.cancel()
            try:
                await dispatcher_task
            except asyncio.CancelledError:
                pass
        if det_thread is not None and det_thread.is_alive():
            # Best-effort: don't hang shutdown.
            det_thread.join(timeout=2.0)

    return app


def parse_args():
    p = argparse.ArgumentParser(description="Live MJPEG + WS server for FaceID")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="Auto-reload server on code changes (dev)")
    p.add_argument("--source", default="0", help='0 for webcam, path to video file, or RTSP URL')
    p.add_argument("--encodings", default=str(Path("models") / "known_faces.pkl"))
    p.add_argument("--yolo", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--thr", type=float, default=0.58)
    p.add_argument("--margin", type=float, default=0.0, help="Min distance margin vs next different identity (0 disables margin check)")

    p.add_argument(
        "--auth-token",
        default="",
        help="Optional shared token required as query param ?token=... for /snapshot.jpg, /stream.mjpg and /ws. Keep empty to disable.",
    )

    p.add_argument("--liveness", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--liveness-algo", choices=["v1", "v2"], default="v2", help="v2 is adaptive and usually better at low FPS")
    p.add_argument(
        "--notify-only-live",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit events/notifications only when liveness is LIVE (skip CHECKING/SPOOF)",
    )
    p.add_argument("--ear-thr", type=float, default=0.21)
    p.add_argument("--blink-min-close", type=float, default=0.06, help="Minimum eye-closed time in seconds to count as a blink")
    p.add_argument("--liveness-min-blinks", type=int, default=1, help="Minimum number of blink events for liveness")
    p.add_argument("--head-turn-range", type=float, default=0.20, help="Required yaw variation for head-turn liveness cue")
    p.add_argument("--liveness-min-actions", type=int, default=1, help="Minimum liveness actions required (blink and/or head turn)")
    p.add_argument("--liveness-min-samples", type=int, default=6, help="Minimum valid EAR samples before SPOOF is allowed")
    p.add_argument("--liveness-grace", type=float, default=10.0)
    p.add_argument("--liveness-ttl", type=float, default=20.0)

    p.add_argument("--emit-cooldown", type=float, default=10.0, help="Safety debounce (seconds) for repeated events per person")
    p.add_argument(
        "--realert-after-missing",
        type=float,
        default=30.0,
        help="Notify again only if the person was missing for at least N seconds",
    )
    p.add_argument("--emit-unknown", action=argparse.BooleanOptionalAction, default=False,
                   help="Emit events also for unknown faces (debug). Default: only known names.")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--max-width", type=int, default=960, help="Downscale frames to this width for speed (0 disables)")
    p.add_argument("--frame-skip", type=int, default=1, help="Process only every Nth frame (>=1)")
    p.add_argument("--max-stream-fps", type=float, default=12.0, help="Cap stream FPS (0 disables)")
    p.add_argument("--fcm-service-account", default="", help="Path to Firebase service-account JSON for push notifications")
    p.add_argument("--device-registry", default=str(Path("models") / "device_tokens.json"), help="Path for stored FCM device tokens")
    p.add_argument("--push-title-prefix", default="FaceClient", help="Prefix for push notification title")
    return p.parse_args()


def main():
    args = parse_args()

    # Fail fast if port is already in use, to avoid confusing partial startup logs.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((args.host, args.port))
        except OSError:
            print(f"Portul {args.port} este deja ocupat. Probabil exista deja un server pornit.")
            try:
                import psutil

                owners = []
                for c in psutil.net_connections(kind="inet"):
                    laddr = getattr(c, "laddr", None)
                    if not laddr:
                        continue
                    if getattr(laddr, "port", None) == args.port and c.status == psutil.CONN_LISTEN:
                        owners.append(c.pid)
                owners = sorted({pid for pid in owners if pid})
                if owners:
                    print(f"PID listening pe portul {args.port}: {', '.join(str(p) for p in owners)}")
                    print("Opreste-l cu: Stop-Process -Id <PID> -Force")
            except Exception:
                pass
            sys.exit(1)

    app = build_app(args)

    # Helpful startup hints: show IPs on active adapters
    try:
        import psutil

        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        ips = []
        for ifname, lst in addrs.items():
            st = stats.get(ifname)
            if st is None or not st.isup:
                continue
            for a in lst:
                # AF_INET == 2 on Windows; keep it simple to avoid importing socket
                if getattr(a, "family", None) == 2 and getattr(a, "address", None):
                    ip = a.address
                    if ip.startswith("127.") or ip.startswith("169.254."):
                        continue
                    ips.append((ifname, ip))

        if ips:
            print("\nServer URLs (use from phone/browser):")
            for ifname, ip in ips:
                print(f"  - {ifname}: http://{ip}:{args.port}/")
            print("")
    except Exception:
        pass

    import uvicorn

    if args.reload:
        # Note: uvicorn's reload mode requires the app to be passed as an import string.
        # Since we run this file as a script (python .\src\live_api_server.py), we'll just
        # ignore reload and ask for a manual restart.
        print("--reload ignored (restart the server manually after code changes)")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
