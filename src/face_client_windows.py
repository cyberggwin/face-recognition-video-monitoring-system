import queue
import subprocess
import sys
import threading
import time
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, X, Y, Button, Frame, Label, StringVar, Text, Tk, Toplevel
from tkinter.scrolledtext import ScrolledText
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image, ImageTk


IS_FROZEN = bool(getattr(sys, "frozen", False))
ROOT_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else Path(__file__).resolve().parents[1]

if IS_FROZEN:
    APP_DATA_ROOT = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "FaceClient"
else:
    APP_DATA_ROOT = ROOT_DIR

APP_DATA_ROOT.mkdir(parents=True, exist_ok=True)

def _seed_tree(src: Path, dst: Path) -> None:
    if (not src.exists()) or dst.exists():
        return
    shutil.copytree(src, dst)

def _seed_file(src: Path, dst: Path) -> None:
    if (not src.exists()) or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

# In installed mode, keep mutable data under LocalAppData to avoid Program Files write restrictions.
if IS_FROZEN:
    _seed_tree(ROOT_DIR / "dataset", APP_DATA_ROOT / "dataset")
    _seed_tree(ROOT_DIR / "models", APP_DATA_ROOT / "models")
    _seed_tree(ROOT_DIR / "credentials", APP_DATA_ROOT / "credentials")
    _seed_file(ROOT_DIR / "credentials.json", APP_DATA_ROOT / "credentials.json")
    _seed_file(ROOT_DIR / "token.json", APP_DATA_ROOT / "token.json")

PYTHON_EXE = ROOT_DIR / "venv311" / "Scripts" / "python.exe"
START_SCRIPT = ROOT_DIR / "Start-Server.ps1"
STOP_SCRIPT = ROOT_DIR / "Stop-Server.ps1"
SYNC_SCRIPT = ROOT_DIR / "src" / "sync_drive_dataset.py"
ENROLL_SCRIPT = ROOT_DIR / "src" / "enroll_faces.py"

SERVER_EXE = ROOT_DIR / "FaceServer.exe"
SYNC_EXE = ROOT_DIR / "SyncDataset.exe"
ENROLL_EXE = ROOT_DIR / "EnrollFaces.exe"
ENCODINGS_PATH = APP_DATA_ROOT / "models" / "known_faces.pkl"
YOLO_PATH = ROOT_DIR / "yolov8n.pt"
FCM_SERVICE_ACCOUNT = APP_DATA_ROOT / "credentials" / "fcm-service-account.json"
DEVICE_REGISTRY_PATH = APP_DATA_ROOT / "models" / "device_tokens.json"
DATASET_PATH = APP_DATA_ROOT / "dataset"
RECORDINGS_DIR = APP_DATA_ROOT / "recordings"


@dataclass
class EventItem:
    ts: float
    name: str
    dist: float


class FaceClientDesktop:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Face Client")
        self.root.geometry("1200x780")
        self.root.configure(bg="#102340")

        self.server_url = "http://127.0.0.1:8000"
        self.snapshot_url = f"{self.server_url}/snapshot.jpg"
        self.last_event_url = f"{self.server_url}/last_event"

        self.server_proc = None
        self.server_monitor_thread: threading.Thread | None = None
        self.stream_running = False
        self.recording = False
        self.video_writer = None
        self.recording_path = None

        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.last_event_ts = 0.0
        self.events: list[EventItem] = []

        self.status_var = StringVar(value="Ready")
        self.server_var = StringVar(value="Server: stopped")
        self.record_var = StringVar(value="Recording: OFF")
        self.notif_var = StringVar(value="Notifications sent: 0")
        self._last_stream_error_log_ts = 0.0

        self._build_ui()

        self.log(f"Data folder: {APP_DATA_ROOT}")

        self.stream_running = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()

        self.event_thread = threading.Thread(target=self._event_loop, daemon=True)
        self.event_thread.start()

        self.root.after(100, self._drain_log_queue)
        self.root.after(33, self._render_loop)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        top = Frame(self.root, bg="#0c1a30")
        top.pack(fill=X, padx=12, pady=10)

        btn_style = {
            "bg": "#1f4b88",
            "fg": "white",
            "activebackground": "#2f67b8",
            "activeforeground": "white",
            "relief": "flat",
            "padx": 12,
            "pady": 8,
        }

        self.start_btn = Button(top, text="Start Server", command=self.start_server, **btn_style)
        self.start_btn.pack(side=LEFT, padx=4)
        self.stop_btn = Button(top, text="Stop Server", command=self.stop_server, **btn_style)
        self.stop_btn.pack(side=LEFT, padx=4)
        self.update_btn = Button(top, text="Update Database", command=self.update_database, **btn_style)
        self.update_btn.pack(side=LEFT, padx=4)
        self.record_btn = Button(top, text="Start Recording", command=self.toggle_recording, **btn_style)
        self.record_btn.pack(side=LEFT, padx=4)
        Button(top, text="Errors", command=self.show_errors, **btn_style).pack(side=LEFT, padx=4)
        Button(top, text="Notifications Sent", command=self.show_notifications, **btn_style).pack(side=LEFT, padx=4)

        status_bar = Frame(self.root, bg="#0c1a30")
        status_bar.pack(fill=X, padx=12, pady=(0, 10))

        Label(status_bar, textvariable=self.status_var, bg="#0c1a30", fg="#cde3ff").pack(side=LEFT, padx=8)
        Label(status_bar, textvariable=self.server_var, bg="#0c1a30", fg="#cde3ff").pack(side=LEFT, padx=8)
        Label(status_bar, textvariable=self.record_var, bg="#0c1a30", fg="#cde3ff").pack(side=LEFT, padx=8)
        Label(status_bar, textvariable=self.notif_var, bg="#0c1a30", fg="#cde3ff").pack(side=RIGHT, padx=8)

        center = Frame(self.root, bg="#102340")
        center.pack(fill=BOTH, expand=True, padx=12, pady=(0, 12))

        self.stream_label = Label(center, bg="#152d52")
        self.stream_label.pack(fill=BOTH, expand=True)

        footer = Frame(self.root, bg="#0c1a30")
        footer.pack(fill=X, padx=12, pady=(0, 10))
        Label(
            footer,
            text="Live stream centered. Last 5 events available in Notifications Sent.",
            bg="#0c1a30",
            fg="#9dc4f5",
        ).pack(side=LEFT, padx=8, pady=6)

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{ts}] {msg}")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._last_logs.append(line)
                if len(self._last_logs) > 400:
                    self._last_logs = self._last_logs[-400:]
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)

    @property
    def _last_logs(self) -> list[str]:
        if not hasattr(self, "__last_logs"):
            self.__last_logs = []
        return self.__last_logs

    @_last_logs.setter
    def _last_logs(self, value: list[str]) -> None:
        self.__last_logs = value

    def start_server(self) -> None:
        if self.server_proc and self.server_proc.poll() is None:
            self.status_var.set("Server already running")
            return

        self.log("Starting server...")
        self.status_var.set("Starting server...")

        if not ENCODINGS_PATH.exists():
            self.log(f"WARNING: encodings missing: {ENCODINGS_PATH}")
            self.log("Run Update Database before Start Server.")

        if IS_FROZEN:
            if not SERVER_EXE.exists():
                self.status_var.set("FaceServer.exe missing")
                self.log(f"ERROR: missing {SERVER_EXE}")
                return

            cmd = [
                str(SERVER_EXE),
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--source",
                "0",
                "--encodings",
                str(ENCODINGS_PATH),
                "--yolo",
                str(YOLO_PATH),
                "--device-registry",
                str(DEVICE_REGISTRY_PATH),
            ]
            if FCM_SERVICE_ACCOUNT.exists():
                cmd.extend(["--fcm-service-account", str(FCM_SERVICE_ACCOUNT)])
        else:
            if not START_SCRIPT.exists():
                self.status_var.set("Start-Server.ps1 not found")
                self.log("ERROR: Start-Server.ps1 not found")
                return

            cmd = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(START_SCRIPT),
                "-ForceRestart",
            ]

        try:
            self.server_proc = subprocess.Popen(
                cmd,
                cwd=str(APP_DATA_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            threading.Thread(target=self._read_server_output, daemon=True).start()
            self.server_monitor_thread = threading.Thread(target=self._monitor_server_process, daemon=True)
            self.server_monitor_thread.start()
            self.server_var.set("Server: running")
            self.status_var.set("Server started")
        except Exception as e:
            self.server_var.set("Server: stopped")
            self.status_var.set("Server start failed")
            self.log(f"ERROR: failed to start server: {e}")

    def _monitor_server_process(self) -> None:
        proc = self.server_proc
        if proc is None:
            return
        try:
            rc = proc.wait()
            if self.stream_running:
                if rc == 0:
                    self.log("Server process exited normally")
                    self.status_var.set("Server stopped")
                else:
                    self.log(f"ERROR: server process exited with code {rc}")
                    self.status_var.set(f"Server failed (code {rc})")
                self.server_var.set("Server: stopped")
        except Exception as e:
            self.log(f"ERROR: server monitor failed: {e}")

    def _read_server_output(self) -> None:
        if not self.server_proc or not self.server_proc.stdout:
            return
        try:
            for line in self.server_proc.stdout:
                text = line.rstrip()
                if text:
                    self.log(text)
        except Exception as e:
            self.log(f"ERROR: reading server output failed: {e}")

    def stop_server(self) -> None:
        self.log("Stopping server...")
        self.status_var.set("Stopping server...")

        if (not IS_FROZEN) and STOP_SCRIPT.exists():
            cmd = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(STOP_SCRIPT),
            ]
            try:
                result = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True, timeout=15)
                if result.stdout.strip():
                    self.log(result.stdout.strip())
                if result.stderr.strip():
                    self.log(f"ERROR: {result.stderr.strip()}")
            except Exception as e:
                self.log(f"ERROR: stop script failed: {e}")

        if self.server_proc and self.server_proc.poll() is None:
            try:
                self.server_proc.terminate()
                self.server_proc.wait(timeout=5)
            except Exception:
                try:
                    self.server_proc.kill()
                except Exception:
                    pass

        self.server_proc = None
        self.server_var.set("Server: stopped")
        self.status_var.set("Server stopped")

    def update_database(self) -> None:
        self.update_btn.configure(state="disabled")
        threading.Thread(target=self._update_database_worker, daemon=True).start()

    def _update_database_worker(self) -> None:
        self.status_var.set("Updating database...")

        try:
            if IS_FROZEN:
                if not SYNC_EXE.exists() or not ENROLL_EXE.exists():
                    self.status_var.set("Update tools missing")
                    self.log("ERROR: SyncDataset.exe or EnrollFaces.exe missing")
                    return

                if not (APP_DATA_ROOT / "credentials.json").exists():
                    self.status_var.set("Update database failed (credentials)")
                    self.log(f"ERROR: missing {(APP_DATA_ROOT / 'credentials.json')}")
                    return

                sync_cmd = [
                    str(SYNC_EXE),
                    "--drive-folder",
                    "proiectlicenta/dataset",
                    "--local",
                    str(DATASET_PATH),
                ]
            else:
                if not PYTHON_EXE.exists():
                    self.status_var.set("Python venv missing")
                    self.log(f"ERROR: missing {PYTHON_EXE}")
                    return

                sync_cmd = [
                    str(PYTHON_EXE),
                    str(SYNC_SCRIPT),
                    "--drive-folder",
                    "proiectlicenta/dataset",
                    "--local",
                    str(DATASET_PATH),
                ]

            self.log("Running sync_drive_dataset.py...")
            sync_res = subprocess.run(sync_cmd, cwd=str(APP_DATA_ROOT), capture_output=True, text=True, timeout=900)
            if sync_res.stdout.strip():
                self.log(sync_res.stdout.strip())
            if sync_res.returncode != 0:
                self.log(sync_res.stderr.strip() or "Unknown sync error")
                self.status_var.set("Update database failed (sync)")
                return

            if IS_FROZEN:
                enroll_cmd = [
                    str(ENROLL_EXE),
                    "--dataset",
                    str(DATASET_PATH),
                    "--output",
                    str(ENCODINGS_PATH),
                    "--jitters",
                    "2",
                ]
            else:
                enroll_cmd = [
                    str(PYTHON_EXE),
                    str(ENROLL_SCRIPT),
                    "--dataset",
                    str(DATASET_PATH),
                    "--output",
                    str(ENCODINGS_PATH),
                    "--jitters",
                    "2",
                ]
            self.log("Running enroll_faces.py...")
            enroll_res = subprocess.run(enroll_cmd, cwd=str(APP_DATA_ROOT), capture_output=True, text=True, timeout=900)
            if enroll_res.stdout.strip():
                self.log(enroll_res.stdout.strip())
            if enroll_res.returncode != 0:
                self.log(enroll_res.stderr.strip() or "Unknown enroll error")
                self.status_var.set("Update database failed (enroll)")
                return

            self.status_var.set("Database updated successfully")
            self.log("Database update completed")
        except Exception as e:
            self.status_var.set("Update database failed")
            self.log(f"ERROR: update database exception: {e}")
        finally:
            self.update_btn.configure(state="normal")

    def toggle_recording(self) -> None:
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.mp4")
        self.recording_path = RECORDINGS_DIR / filename
        self.recording = True
        self.record_btn.configure(text="Stop Recording")
        self.record_var.set(f"Recording: ON ({filename})")
        self.status_var.set("Recording started")
        self.log(f"Recording started: {self.recording_path}")

    def _stop_recording(self) -> None:
        self.recording = False
        self.record_btn.configure(text="Start Recording")
        self.record_var.set("Recording: OFF")
        self.status_var.set("Recording stopped")
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.log("Recording stopped")

    def _stream_loop(self) -> None:
        while self.stream_running:
            frame = self._fetch_snapshot_frame()
            if frame is None:
                time.sleep(0.25)
                continue

            if self.recording:
                self._write_recording_frame(frame)

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
            time.sleep(0.03)

    def _fetch_snapshot_frame(self):
        try:
            with urlopen(self.snapshot_url, timeout=2) as resp:
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                now = time.time()
                if (now - self._last_stream_error_log_ts) > 5.0:
                    self.log("ERROR: received invalid snapshot frame")
                    self._last_stream_error_log_ts = now
            return frame
        except URLError:
            now = time.time()
            if (now - self._last_stream_error_log_ts) > 5.0:
                self.log("ERROR: snapshot unavailable (server down or not ready)")
                self._last_stream_error_log_ts = now
            return None
        except Exception:
            now = time.time()
            if (now - self._last_stream_error_log_ts) > 5.0:
                self.log("ERROR: snapshot read failed")
                self._last_stream_error_log_ts = now
            return None

    def _write_recording_frame(self, frame: np.ndarray) -> None:
        try:
            if self.video_writer is None and self.recording_path is not None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(str(self.recording_path), fourcc, 15.0, (w, h))
            if self.video_writer is not None:
                self.video_writer.write(frame)
        except Exception as e:
            self.log(f"ERROR: recording write failed: {e}")

    def _render_loop(self) -> None:
        frame = None
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            max_w = max(500, self.stream_label.winfo_width())
            max_h = max(350, self.stream_label.winfo_height())
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            tk_img = ImageTk.PhotoImage(img)
            self.stream_label.configure(image=tk_img)
            self.stream_label.image = tk_img

        self.root.after(33, self._render_loop)

    def _event_loop(self) -> None:
        while self.stream_running:
            try:
                with urlopen(self.last_event_url, timeout=2) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                payload = self._parse_json(raw)
                if payload and payload.get("type") == "person_detected":
                    ts = float(payload.get("ts", 0.0))
                    if ts > self.last_event_ts:
                        self.last_event_ts = ts
                        item = EventItem(
                            ts=ts,
                            name=str(payload.get("name", "unknown")),
                            dist=float(payload.get("dist", 1.0)),
                        )
                        self.events.insert(0, item)
                        if len(self.events) > 200:
                            self.events = self.events[:200]
                        self.notif_var.set(f"Notifications sent: {len(self.events)}")
                        self.log(f"Notification event: {item.name} ({item.dist:.2f})")
            except Exception:
                pass
            time.sleep(1.0)

    @staticmethod
    def _parse_json(raw: str):
        import json

        try:
            return json.loads(raw)
        except Exception:
            return None

    def show_errors(self) -> None:
        win = Toplevel(self.root)
        win.title("Errors / Logs")
        win.geometry("900x500")
        txt = ScrolledText(win, wrap="word")
        txt.pack(fill=BOTH, expand=True)

        lines = self._last_logs or ["No logs yet"]
        txt.insert("1.0", "\n".join(lines))
        txt.configure(state="disabled")

    def show_notifications(self) -> None:
        win = Toplevel(self.root)
        win.title("Notifications sent")
        win.geometry("700x420")
        txt = Text(win, wrap="none")
        txt.pack(fill=BOTH, expand=True)

        if not self.events:
            txt.insert("1.0", "No notification events yet")
        else:
            txt.insert("1.0", "Last events (newest first)\n\n")
            for ev in self.events[:100]:
                hhmmss = datetime.fromtimestamp(ev.ts).strftime("%H:%M:%S")
                txt.insert("end", f"{hhmmss} | {ev.name} | dist={ev.dist:.2f}\n")
        txt.configure(state="disabled")

    def on_close(self) -> None:
        self.stream_running = False
        if self.recording:
            self._stop_recording()
        self.stop_server()
        self.root.destroy()


def main() -> None:
    app = Tk()
    FaceClientDesktop(app)
    app.mainloop()


if __name__ == "__main__":
    main()
