"""Blink-based liveness utilities using EAR from landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


Point = Tuple[int, int]
Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(eye: Iterable[Point]) -> Optional[float]:
    """Compute Eye Aspect Ratio (EAR) from 6 eye landmark points.

    Standard ordering used by dlib/face_recognition:
    - eye has 6 points around the eye contour.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Returns None if not enough points.
    """
    pts = list(eye)
    if len(pts) != 6:
        return None

    p1 = np.array(pts[0], dtype=np.float32)
    p2 = np.array(pts[1], dtype=np.float32)
    p3 = np.array(pts[2], dtype=np.float32)
    p4 = np.array(pts[3], dtype=np.float32)
    p5 = np.array(pts[4], dtype=np.float32)
    p6 = np.array(pts[5], dtype=np.float32)

    a = _euclidean(p2, p6)
    b = _euclidean(p3, p5)
    c = _euclidean(p1, p4)
    if c <= 1e-6:
        return None
    return (a + b) / (2.0 * c)


def mean_ear(landmarks: Dict[str, List[Point]]) -> Optional[float]:
    """Compute mean EAR from landmarks dict returned by face_recognition.face_landmarks."""
    left = landmarks.get("left_eye")
    right = landmarks.get("right_eye")
    if not left or not right:
        return None

    left_ear = eye_aspect_ratio(left)
    right_ear = eye_aspect_ratio(right)
    if left_ear is None or right_ear is None:
        return None
    return (left_ear + right_ear) / 2.0


def yaw_proxy(landmarks: Dict[str, List[Point]]) -> Optional[float]:
    """Approximate face yaw from landmarks.

    Returns a normalized value around 0 for frontal faces.
    Negative values suggest looking left, positive looking right.
    """
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")
    nose_tip = landmarks.get("nose_tip")
    if not left_eye or not right_eye or not nose_tip:
        return None

    lx = float(np.mean([p[0] for p in left_eye]))
    rx = float(np.mean([p[0] for p in right_eye]))
    nx = float(np.mean([p[0] for p in nose_tip]))

    eye_dist = abs(rx - lx)
    if eye_dist <= 1e-6:
        return None

    mid = (lx + rx) / 2.0
    return (nx - mid) / eye_dist


@dataclass
class TrackState:
    track_id: int
    first_seen_ts: float
    last_seen_ts: float
    last_box: Box

    # Blink tracking
    eye_closed_since_ts: Optional[float] = None
    ear_samples: int = 0
    total_blinks: int = 0
    last_blink_ts: float = 0.0

    # Head turn tracking
    yaw_min: float = 1e9
    yaw_max: float = -1e9
    head_turn_done: bool = False

    # Adaptive v2 tracking
    ear_ema: float = 0.0
    ear_baseline: float = 0.0
    eye_closed: bool = False
    eye_closed_started_ts: float = 0.0


class SimpleFaceTracker:
    """Very simple face tracker based on nearest center distance.

    This is intentionally lightweight: it assigns a stable ID to face boxes
    across frames so we can accumulate blink evidence.
    """

    def __init__(self, max_center_dist_px: float = 80.0, max_missing_s: float = 1.5):
        self._next_id = 1
        self._tracks: Dict[int, TrackState] = {}
        self.max_center_dist_px = float(max_center_dist_px)
        self.max_missing_s = float(max_missing_s)

    @staticmethod
    def _center(box: Box) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _purge(self, now_ts: float) -> None:
        to_del = [tid for tid, st in self._tracks.items() if (now_ts - st.last_seen_ts) > self.max_missing_s]
        for tid in to_del:
            del self._tracks[tid]

    def assign(self, box: Box, now_ts: float) -> int:
        """Return a stable track id for the given box."""
        self._purge(now_ts)

        cx, cy = self._center(box)
        best_id: Optional[int] = None
        best_dist = 1e9

        for tid, st in self._tracks.items():
            pcx, pcy = self._center(st.last_box)
            dist = float(np.hypot(cx - pcx, cy - pcy))
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id is not None and best_dist <= self.max_center_dist_px:
            st = self._tracks[best_id]
            st.last_seen_ts = now_ts
            st.last_box = box
            return best_id

        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = TrackState(track_id=tid, first_seen_ts=now_ts, last_seen_ts=now_ts, last_box=box)
        return tid

    def get_state(self, track_id: int) -> Optional[TrackState]:
        return self._tracks.get(track_id)


class BlinkLiveness:
    """Streaming blink-based liveness.

    States:
    - CHECKING: within grace period and no blink yet
    - LIVE: at least one blink observed
    - SPOOF: grace period elapsed without a blink

    You can explain this in thesis as a temporal physiological cue.
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        min_closed_seconds: float = 0.06,
        grace_seconds: float = 4.0,
        live_ttl_seconds: float = 20.0,
        min_blinks: int = 1,
        min_actions: int = 1,
        head_turn_range: float = 0.20,
        min_ear_samples_before_spoof: int = 8,
    ):
        self.ear_threshold = float(ear_threshold)
        self.min_closed_seconds = float(min_closed_seconds)
        self.grace_seconds = float(grace_seconds)
        self.live_ttl_seconds = float(live_ttl_seconds)
        self.min_blinks = int(min_blinks)
        self.min_actions = int(min_actions)
        self.head_turn_range = float(head_turn_range)
        self.min_ear_samples_before_spoof = int(min_ear_samples_before_spoof)

    def update(
        self,
        st: TrackState,
        ear: Optional[float],
        landmarks: Optional[Dict[str, List[Point]]],
        now_ts: float,
        box: Optional[Box] = None,
    ) -> str:
        """Update track state with current EAR and return liveness label."""
        if ear is not None:
            st.ear_samples += 1

            if ear < self.ear_threshold:
                if st.eye_closed_since_ts is None:
                    st.eye_closed_since_ts = now_ts
            else:
                # Rising edge: counts as blink if eye stayed closed long enough.
                if st.eye_closed_since_ts is not None and (now_ts - st.eye_closed_since_ts) >= self.min_closed_seconds:
                    st.total_blinks += 1
                    st.last_blink_ts = now_ts
                st.eye_closed_since_ts = None

        if landmarks is not None:
            y = yaw_proxy(landmarks)
            if y is not None:
                st.yaw_min = min(st.yaw_min, y)
                st.yaw_max = max(st.yaw_max, y)
                if (st.yaw_max - st.yaw_min) >= self.head_turn_range:
                    st.head_turn_done = True

        return self.label(st, now_ts)

    def label(self, st: TrackState, now_ts: float) -> str:
        blink_ok = st.total_blinks >= self.min_blinks
        action_count = (1 if blink_ok else 0) + (1 if st.head_turn_done else 0)

        # LIVE if enough liveness actions were observed.
        if action_count >= self.min_actions:
            if self.live_ttl_seconds <= 0:
                return "LIVE"
            if (now_ts - st.last_blink_ts) <= self.live_ttl_seconds:
                return "LIVE"
            # If LIVE is also backed by head movement, keep it LIVE even without fresh blink.
            if st.head_turn_done:
                return "LIVE"

        # With sparse/unstable frames, wait for enough valid EAR samples before deciding SPOOF.
        if st.ear_samples < self.min_ear_samples_before_spoof:
            return "CHECKING"

        # Otherwise CHECKING until grace expires, then SPOOF.
        if (now_ts - st.first_seen_ts) <= self.grace_seconds:
            return "CHECKING"
        return "SPOOF"


class AdaptiveLiveness:
    """Adaptive liveness for low FPS / glasses scenarios.

    Compared to v1, this variant adapts EAR thresholds per tracked face,
    which usually reduces false SPOOF for real users.
    """

    def __init__(
        self,
        grace_seconds: float = 8.0,
        live_ttl_seconds: float = 25.0,
        min_blinks: int = 1,
        min_actions: int = 2,
        min_ear_samples_before_spoof: int = 8,
        close_ratio: float = 0.73,
        open_ratio: float = 0.84,
        ema_alpha: float = 0.35,
        min_blink_seconds: float = 0.05,
        max_blink_seconds: float = 0.8,
        head_turn_range: float = 0.16,
    ):
        self.grace_seconds = float(grace_seconds)
        self.live_ttl_seconds = float(live_ttl_seconds)
        self.min_blinks = int(min_blinks)
        self.min_actions = int(min_actions)
        self.min_ear_samples_before_spoof = int(min_ear_samples_before_spoof)
        self.close_ratio = float(close_ratio)
        self.open_ratio = float(open_ratio)
        self.ema_alpha = float(ema_alpha)
        self.min_blink_seconds = float(min_blink_seconds)
        self.max_blink_seconds = float(max_blink_seconds)
        self.head_turn_range = float(head_turn_range)

    def update(
        self,
        st: TrackState,
        ear: Optional[float],
        landmarks: Optional[Dict[str, List[Point]]],
        now_ts: float,
        box: Optional[Box] = None,
    ) -> str:
        if ear is not None:
            st.ear_samples += 1

            if st.ear_ema <= 0:
                st.ear_ema = ear
            else:
                st.ear_ema = (1.0 - self.ema_alpha) * st.ear_ema + self.ema_alpha * ear

            if st.ear_baseline <= 0:
                st.ear_baseline = st.ear_ema
            else:
                # Update baseline only when likely open-eye state.
                if st.ear_ema > (st.ear_baseline * 0.80):
                    st.ear_baseline = 0.94 * st.ear_baseline + 0.06 * st.ear_ema

            close_thr = st.ear_baseline * self.close_ratio
            open_thr = st.ear_baseline * self.open_ratio

            if (not st.eye_closed) and st.ear_ema < close_thr:
                st.eye_closed = True
                st.eye_closed_started_ts = now_ts
            elif st.eye_closed and st.ear_ema > open_thr:
                dur = now_ts - st.eye_closed_started_ts
                if self.min_blink_seconds <= dur <= self.max_blink_seconds:
                    st.total_blinks += 1
                    st.last_blink_ts = now_ts
                st.eye_closed = False
                st.eye_closed_started_ts = 0.0

        if landmarks is not None:
            y = yaw_proxy(landmarks)
            if y is not None:
                st.yaw_min = min(st.yaw_min, y)
                st.yaw_max = max(st.yaw_max, y)
                if (st.yaw_max - st.yaw_min) >= self.head_turn_range:
                    st.head_turn_done = True

        return self.label(st, now_ts)

    def label(self, st: TrackState, now_ts: float) -> str:
        blink_ok = st.total_blinks >= self.min_blinks
        action_count = (1 if blink_ok else 0) + (1 if st.head_turn_done else 0)

        if action_count >= self.min_actions:
            # Keep LIVE for a TTL window after last blink/head-turn evidence.
            if self.live_ttl_seconds <= 0:
                return "LIVE"
            if blink_ok and (now_ts - st.last_blink_ts) <= self.live_ttl_seconds:
                return "LIVE"
            if st.head_turn_done:
                return "LIVE"

        if st.ear_samples < self.min_ear_samples_before_spoof:
            return "CHECKING"
        if (now_ts - st.first_seen_ts) <= self.grace_seconds:
            return "CHECKING"
        return "SPOOF"
