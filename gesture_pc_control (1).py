"""
Gesture PC Control
==================
Hand-tracked mouse/cursor controller using MediaPipe. Move your cursor,
click, drag, scroll, and zoom, all with your webcam and no extra hardware.

Controls / gestures:
  INDEX FINGER UP (only)              = Move cursor
  THUMB + INDEX pinch                 = Left click (quick pinch) / drag (hold + move)
  THUMB + MIDDLE pinch                = Right click
  PEACE SIGN (index+middle up)        = Scroll (move hand up/down)
  BOTH HANDS OPEN, move apart/together = Zoom (Ctrl + scroll)
  OPEN PALM, swipe quickly left/right  = Swipe (default: browser/back-forward)
  P                                   = Pause / resume gesture control
  Q / Esc                             = Quit

Everything tunable lives in gesture_config.json next to this script — it's
created automatically on first run.

Install deps first:
    pip install opencv-python mediapipe pyautogui numpy pynput
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import json
import platform
import threading
from collections import deque
from pathlib import Path

import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

pyautogui.FAILSAFE = False   # we're intentionally driving the mouse ourselves
pyautogui.PAUSE = 0.0        # don't let pyautogui sleep between calls; we pace the loop instead

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("gesture_config.json")

DEFAULT_CONFIG = {
    "camera_index": 0,
    "flip_camera": True,
    "target_width": 640,
    "target_height": 480,
    "min_detection_conf": 0.6,
    "min_tracking_conf": 0.5,
    "grace_frames": 8,

    # --- Cursor mapping ---
    # The "active box" is a region within the camera frame (as fractions of
    # width/height) that maps to the FULL screen. Keeping it smaller than the
    # whole frame means small hand movements cover the whole screen, so you
    # don't have to reach to the edges of the camera view.
    "active_box": [0.15, 0.15, 0.85, 0.85],   # [x1, y1, x2, y2] as fractions

    # One Euro Filter for the cursor. min_cutoff: lower = smoother/more lag
    # when the hand is nearly still. beta: how much a fast movement is
    # allowed to cut through that smoothing — higher = snappier on quick
    # moves, but a bit more jitter on quick moves too.
    "one_euro_min_cutoff": 1.2,
    "one_euro_beta": 0.02,
    "one_euro_d_cutoff": 1.0,

    # --- Gestures ---
    # Pinch (click) distance is normalized by hand size (wrist-to-middle-
    # knuckle distance) AND self-calibrated per hand via a live-tracked
    # open/closed envelope (see PinchEnvelope) — no fixed threshold to
    # mistune for a given hand/camera. These fractions say where, within
    # that live-observed open<->closed span, "closed enough to click" and
    # "open enough to release" sit.
    "pinch_close_frac": 0.30,       # fraction of the open<->closed span that counts as "pinched"
    "pinch_release_frac": 0.50,     # must open back past this fraction to release (hysteresis)
    "pinch_envelope_attack": 0.15,  # how fast the envelope chases a new extreme (open/closed)
    "pinch_envelope_release": 0.01, # how fast it relaxes back if that extreme wasn't seen again
    "pinch_distance_smoothing": 0.5,  # 0..1 EMA on the pinch distance itself; lower = smoother, less shake-prone
    "pinch_confirm_frames": 3,        # consecutive frames the pinch must hold before it counts (debounce for shaky hands)
    "click_cooldown": 0.35,           # min seconds between separate click/right-click firings
    "click_max_hold_time": 0.35,  # pinch shorter than this = click; longer = drag
    "open_finger_thresh": 4,      # fingers extended to count hand as "open" (0-5)

    # --- Scroll ---
    # --- Scroll (joystick-style: hold below/above center to scroll continuously) ---
    "scroll_neutral_y": 0.5,          # normalized y (0=top, 1=bottom) treated as "center, no scroll"
    "scroll_deadzone_norm": 0.06,     # +/- this far from neutral = still zone, no scroll
    "scroll_max_offset_norm": 0.32,   # offset at which scroll speed reaches its max
    "scroll_max_speed": 45.0,         # scroll ticks per second at full deflection

    # --- Zoom ---
    "zoom_sensitivity": 60.0,     # ctrl-scroll ticks per normalized unit of wrist-distance change
    "zoom_deadzone": 0.004,

    # --- Swipe (open palm, swiped quickly left/right) ---
    "swipe_enabled": True,
    "swipe_distance_threshold": 0.16,   # min horizontal travel (fraction of frame width) to count as a swipe
    "swipe_time_window": 0.45,          # swipe must complete within this many seconds
    "swipe_cooldown": 0.7,              # ignore new swipes for this long after one fires
    # Hotkey sent for each direction (pyautogui.hotkey(*keys)). Defaults to
    # browser/file-explorer back-forward. For Windows virtual-desktop
    # switching instead, use ["ctrl", "winleft", "left"] / ["...", "right"].
    "swipe_left_hotkey": ["alt", "left"],
    "swipe_right_hotkey": ["alt", "right"],

    "show_camera_window": True,
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"[WARN] Could not read {CONFIG_PATH.name}, using defaults: {e}")
    else:
        save_config(cfg)
        print(f"[INFO] Created {CONFIG_PATH.name} — edit it to tune the app.")
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save config: {e}")


CFG = load_config()
SCREEN_W, SCREEN_H = pyautogui.size()

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

# Colors (BGR) — dark glass HUD, same family as the steering-wheel app
CLR_BG        = (26, 18, 10)
CLR_PANEL     = (48, 34, 18)
CLR_PANEL_ED  = (130, 90, 40)
CLR_TEXT      = (245, 245, 250)
CLR_TEXT_DIM  = (170, 165, 180)
CLR_ACCENT    = (255, 210, 60)
CLR_MOVE      = (255, 220, 90)
CLR_CLICK     = (110, 255, 130)
CLR_RCLICK    = (90, 90, 255)
CLR_SCROLL    = (255, 140, 220)
CLR_ZOOM      = (120, 255, 210)
CLR_SWIPE     = (255, 180, 60)
CLR_IDLE      = (170, 165, 180)
CLR_PAUSED    = (90, 90, 255)


# ---------------------------------------------------------------------------
# Hand geometry helpers
# ---------------------------------------------------------------------------
def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def hand_scale(hand_landmarks):
    """Wrist-to-middle-knuckle distance — a stand-in for "how big the hand
    looks right now", i.e. how close it is to the camera. Used to normalize
    pinch distances so click detection doesn't depend on hand distance."""
    return max(_dist(hand_landmarks[0], hand_landmarks[9]), 1e-6)


def finger_extended_flags(hand_landmarks):
    """Rotation-invariant per-finger extended/folded flags.

    Compares each fingertip's distance from the wrist to its knuckle (MCP)'s
    distance from the wrist, so it stays accurate regardless of hand tilt
    (unlike a simple y-coordinate comparison).
    """
    wrist = hand_landmarks[0]
    scale = hand_scale(hand_landmarks)
    TIPS = [4, 8, 12, 16, 20]
    MCPS = [2, 5, 9, 13, 17]
    flags = []
    for tip_i, mcp_i in zip(TIPS, MCPS):
        tip_dist = _dist(wrist, hand_landmarks[tip_i])
        mcp_dist = _dist(wrist, hand_landmarks[mcp_i])
        flags.append(tip_dist > mcp_dist * 1.15)
    return flags  # [thumb, index, middle, ring, pinky]


def is_open_hand(flags, thresh):
    return sum(flags) >= thresh


def classify_pose(flags):
    thumb, index, middle, ring, pinky = flags
    if index and not middle and not ring and not pinky:
        return "POINT"
    if index and middle and not ring and not pinky:
        return "PEACE"
    if sum(flags) >= 4:
        return "OPEN"
    if sum(flags) == 0:
        return "FIST"
    return "OTHER"


# ---------------------------------------------------------------------------
# One Euro Filter — adaptive smoothing for cursor x/y.
#
# A plain low-pass filter (or Kalman filter with fixed noise terms) has to
# pick one point on the jitter-vs-lag tradeoff and live with it everywhere.
# The One Euro Filter instead widens its cutoff (less smoothing, less lag)
# when the signal is moving fast, and narrows it (more smoothing, less
# jitter) when the signal is nearly still — which is exactly the behavior
# you want from a hand-tracked cursor: rock steady when you're holding it in
# place, snappy when you're actually moving it.
# ---------------------------------------------------------------------------
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        dt = max(t - self.t_prev, 1e-4)

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def reset(self, x, t):
        self.x_prev = x
        self.dx_prev = 0.0
        self.t_prev = t


# ---------------------------------------------------------------------------
# Pinch envelope — self-calibrating open/closed distance tracker.
#
# A single fixed pinch_threshold can't be right for everyone: hand size,
# camera distance, and MediaPipe's own landmark noise all shift where "open"
# and "closed" actually sit for a given person's setup. This tracks a
# running "how open does this hand get" / "how closed does this hand get"
# envelope (attack/release, like audio auto-gain) and derives the click
# threshold as a fraction of that live-observed span, so clicking works
# consistently without hand-tuned magic numbers.
# ---------------------------------------------------------------------------
class PinchEnvelope:
    def __init__(self, attack=0.15, release=0.01, init_min=0.12, init_max=0.9):
        self.env_min = init_min
        self.env_max = init_max
        self.attack = attack
        self.release = release

    def update(self, dist):
        if dist > self.env_max:
            self.env_max += (dist - self.env_max) * self.attack
        else:
            self.env_max += (dist - self.env_max) * self.release
        if dist < self.env_min:
            self.env_min += (dist - self.env_min) * self.attack
        else:
            self.env_min += (dist - self.env_min) * self.release
        # Keep a sane minimum spread so the thresholds can't collapse onto
        # each other if the hand hasn't moved much yet (e.g. right at startup).
        if self.env_max - self.env_min < 0.08:
            mid = (self.env_max + self.env_min) / 2
            self.env_min, self.env_max = mid - 0.04, mid + 0.04

    def thresholds(self, close_frac, release_frac):
        span = self.env_max - self.env_min
        return self.env_min + close_frac * span, self.env_min + release_frac * span


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def open_camera(cfg):
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg["camera_index"], backend)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        cap.release()

    print("[INFO] Configured camera index failed, scanning for a working one...")
    for i in range(5):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"[INFO] Found working camera at index {i}")
                cfg["camera_index"] = i
                save_config(cfg)
                return cap
        cap.release()
    return None


class ThreadedCamera:
    """Background-thread capture so read() always returns the newest frame
    instead of blocking / lagging behind, same idea as in the steering app."""

    def __init__(self, cap):
        self.cap = cap
        self._frame = None
        self._ok = False
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stopped:
            ok, frame = self.cap.read()
            with self._lock:
                self._ok = ok
                self._frame = frame
            if not ok:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def set(self, prop, value):
        return self.cap.set(prop, value)

    def release(self):
        self._stopped = True
        self._thread.join(timeout=1.0)
        self.cap.release()


# ---------------------------------------------------------------------------
# Controller — turns gestures into real mouse/keyboard actions
# ---------------------------------------------------------------------------
class GestureController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fx_filter = OneEuroFilter(cfg["one_euro_min_cutoff"], cfg["one_euro_beta"], cfg["one_euro_d_cutoff"])
        self.fy_filter = OneEuroFilter(cfg["one_euro_min_cutoff"], cfg["one_euro_beta"], cfg["one_euro_d_cutoff"])
        self.cursor_x, self.cursor_y = SCREEN_W // 2, SCREEN_H // 2

        self.left_pinch_active = False
        self.left_pinch_start_t = 0.0
        self.dragging = False
        self.left_confirm_count = 0
        self.left_release_count = 0
        self.last_left_click_time = 0.0
        self.left_envelope = PinchEnvelope(cfg["pinch_envelope_attack"], cfg["pinch_envelope_release"])

        self.right_pinch_active = False
        self.right_confirm_count = 0
        self.right_release_count = 0
        self.last_right_click_time = 0.0
        self.right_envelope = PinchEnvelope(cfg["pinch_envelope_attack"], cfg["pinch_envelope_release"])

        self._pinch_dist_ema = {}  # smoothed pinch distances, keyed by "left"/"right"
        self.last_pinch_debug = {}  # last (dist, close_th, release_th) per hand, for the HUD

        self.scroll_accum = 0.0
        self.zoom_last_dist = None

        self.swipe_positions = deque(maxlen=30)   # (timestamp, normalized_x)
        self.last_swipe_time = 0.0

        self.paused = False

    # -- cursor -------------------------------------------------------
    def move_cursor(self, norm_x, norm_y, now):
        x1, y1, x2, y2 = self.cfg["active_box"]
        # Map the active box -> [0,1], clamped, then -> screen pixels.
        fx = (norm_x - x1) / max(x2 - x1, 1e-6)
        fy = (norm_y - y1) / max(y2 - y1, 1e-6)
        fx = min(1.0, max(0.0, fx))
        fy = min(1.0, max(0.0, fy))

        # A single adaptive filter, not several stacked ones — each extra
        # smoothing stage adds its own bit of lag, and it adds up fast.
        fx_f = self.fx_filter.filter(fx, now)
        fy_f = self.fy_filter.filter(fy, now)

        self.cursor_x = fx_f * SCREEN_W
        self.cursor_y = fy_f * SCREEN_H
        pyautogui.moveTo(int(self.cursor_x), int(self.cursor_y))

    def reset_cursor_filter(self, norm_x, norm_y, now):
        """Call when the hand reappears after being lost, so the filter
        doesn't try to smoothly interpolate across the gap (which would
        look like an extra burst of lag)."""
        x1, y1, x2, y2 = self.cfg["active_box"]
        fx = min(1.0, max(0.0, (norm_x - x1) / max(x2 - x1, 1e-6)))
        fy = min(1.0, max(0.0, (norm_y - y1) / max(y2 - y1, 1e-6)))
        self.fx_filter.reset(fx, now)
        self.fy_filter.reset(fy, now)

    # -- click / drag ---------------------------------------------------
    def _smoothed(self, key, raw_dist):
        alpha = self.cfg["pinch_distance_smoothing"]
        prev = self._pinch_dist_ema.get(key)
        smoothed = raw_dist if prev is None else prev + alpha * (raw_dist - prev)
        self._pinch_dist_ema[key] = smoothed
        return smoothed

    def update_left_pinch(self, raw_dist, now):
        """Left click / drag. A shaky hand can make the raw distance bounce
        across the threshold several times a second, so this smooths the
        distance (EMA) and additionally requires it to stay past the
        threshold for a few consecutive frames before actually flipping
        state — a single noisy frame can no longer register as a full
        click or an early release.

        `raw_dist` is expected to already be normalized by hand size (see
        `hand_scale`). The click/release thresholds themselves come from
        `left_envelope`, which learns this hand's actual open/closed range
        live rather than relying on one fixed number for everyone."""
        cfg = self.cfg
        dist = self._smoothed("left", raw_dist)
        self.left_envelope.update(dist)
        close_th, release_th = self.left_envelope.thresholds(cfg["pinch_close_frac"], cfg["pinch_release_frac"])
        self.last_pinch_debug["left"] = (dist, close_th, release_th)
        confirm_n = cfg["pinch_confirm_frames"]

        if not self.left_pinch_active:
            self.left_confirm_count = self.left_confirm_count + 1 if dist < close_th else 0
            if self.left_confirm_count >= confirm_n:
                self.left_pinch_active = True
                self.left_confirm_count = 0
                self.left_pinch_start_t = now
                pyautogui.mouseDown(button="left")
            return None

        # currently pinching (mouse down)
        if dist > release_th:
            self.left_release_count += 1
        else:
            self.left_release_count = 0

        if self.left_release_count >= confirm_n:
            self.left_pinch_active = False
            self.left_release_count = 0
            held = now - self.left_pinch_start_t
            pyautogui.mouseUp(button="left")
            was_dragging = self.dragging
            self.dragging = False
            return "CLICK" if held < cfg["click_max_hold_time"] and not was_dragging else "DRAG-END"

        held = now - self.left_pinch_start_t
        if held >= cfg["click_max_hold_time"] and not self.dragging:
            self.dragging = True
            return "DRAGGING"
        return None

    def update_right_pinch(self, raw_dist, now):
        """Right click: a one-shot tap with its own cooldown, so a shaky
        hand hovering near the threshold can't fire several right-clicks
        in a row even if the smoothed distance still wobbles a little.

        `raw_dist` is expected to already be normalized by hand size; the
        thresholds come from `right_envelope`, self-calibrated the same way
        as the left pinch."""
        cfg = self.cfg
        dist = self._smoothed("right", raw_dist)
        self.right_envelope.update(dist)
        close_th, release_th = self.right_envelope.thresholds(cfg["pinch_close_frac"], cfg["pinch_release_frac"])
        self.last_pinch_debug["right"] = (dist, close_th, release_th)
        confirm_n = cfg["pinch_confirm_frames"]

        if not self.right_pinch_active:
            self.right_confirm_count = self.right_confirm_count + 1 if dist < close_th else 0
            if (self.right_confirm_count >= confirm_n
                    and now - self.last_right_click_time >= cfg["click_cooldown"]):
                self.right_pinch_active = True
                self.right_confirm_count = 0
                self.last_right_click_time = now
                pyautogui.click(button="right")
                return "RIGHT-CLICK"
            return None

        # currently "held" — wait for a clean, confirmed release before rearming
        if dist > release_th:
            self.right_release_count += 1
        else:
            self.right_release_count = 0
        if self.right_release_count >= confirm_n:
            self.right_pinch_active = False
            self.right_release_count = 0
        return None

    # -- scroll -------------------------------------------------------------
    # Joystick-style: the hand's vertical offset from a neutral zone sets a
    # scroll SPEED, not a one-shot amount — so holding the gesture below
    # center keeps scrolling continuously for as long as you hold it,
    # instead of stopping the moment your hand stops moving (or hits the
    # edge of frame). Fractional ticks are accumulated across frames so the
    # scroll rate is smooth and frame-rate independent.
    def update_scroll(self, norm_y, dt):
        cfg = self.cfg
        neutral = cfg["scroll_neutral_y"]
        deadzone = cfg["scroll_deadzone_norm"]
        max_offset = cfg["scroll_max_offset_norm"]
        max_speed = cfg["scroll_max_speed"]  # ticks / second at full deflection

        offset = norm_y - neutral  # >0 => hand below neutral => scroll down
        if abs(offset) < deadzone:
            self.scroll_accum = 0.0
            return 0

        extra = min(1.0, (abs(offset) - deadzone) / max(max_offset - deadzone, 1e-6))
        speed = extra * max_speed
        signed_speed = -speed if offset > 0 else speed  # pyautogui: +up / -down

        self.scroll_accum += signed_speed * dt
        ticks = int(self.scroll_accum)
        self.scroll_accum -= ticks
        if ticks != 0:
            pyautogui.scroll(ticks)
        return ticks

    def reset_scroll(self):
        self.scroll_accum = 0.0

    # -- zoom ---------------------------------------------------------------
    def update_zoom(self, wrist_dist):
        cfg = self.cfg
        if self.zoom_last_dist is None:
            self.zoom_last_dist = wrist_dist
            return 0
        d = wrist_dist - self.zoom_last_dist
        self.zoom_last_dist = wrist_dist
        if abs(d) < cfg["zoom_deadzone"]:
            return 0
        ticks = int(d * cfg["zoom_sensitivity"] * 40)
        if ticks != 0:
            pyautogui.keyDown("ctrl")
            pyautogui.scroll(ticks)
            pyautogui.keyUp("ctrl")
        return ticks

    def reset_zoom(self):
        self.zoom_last_dist = None

    # -- swipe (open palm, moved quickly left/right) -----------------------
    def update_swipe(self, norm_x, now):
        cfg = self.cfg
        if not cfg.get("swipe_enabled", True):
            return None

        self.swipe_positions.append((now, norm_x))
        # Drop samples older than the swipe time window.
        window = cfg["swipe_time_window"]
        while self.swipe_positions and now - self.swipe_positions[0][0] > window:
            self.swipe_positions.popleft()

        if now - self.last_swipe_time < cfg["swipe_cooldown"]:
            return None
        if len(self.swipe_positions) < 3:
            return None

        oldest_t, oldest_x = self.swipe_positions[0]
        dx = norm_x - oldest_x
        if abs(dx) < cfg["swipe_distance_threshold"]:
            return None

        # Require the motion to be mostly monotonic (a real swipe, not a
        # back-and-forth wobble) by checking the midpoint sits between the ends.
        mid_x = self.swipe_positions[len(self.swipe_positions) // 2][1]
        if dx > 0 and not (oldest_x <= mid_x <= norm_x + 1e-6):
            return None
        if dx < 0 and not (oldest_x >= mid_x >= norm_x - 1e-6):
            return None

        self.last_swipe_time = now
        self.swipe_positions.clear()

        if dx > 0:
            pyautogui.hotkey(*cfg["swipe_right_hotkey"])
            return "SWIPE RIGHT"
        else:
            pyautogui.hotkey(*cfg["swipe_left_hotkey"])
            return "SWIPE LEFT"

    def reset_swipe(self):
        self.swipe_positions.clear()

    def release_all(self):
        if self.left_pinch_active:
            pyautogui.mouseUp(button="left")
        self.left_pinch_active = False
        self.dragging = False
        self.left_confirm_count = 0
        self.left_release_count = 0
        self.right_pinch_active = False
        self.right_confirm_count = 0
        self.right_release_count = 0
        self._pinch_dist_ema.clear()
        self.reset_scroll()
        self.reset_zoom()
        self.reset_swipe()


# ---------------------------------------------------------------------------
# Drawing — lightweight glass HUD
# ---------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_DUPLEX


def rounded_rect(frame, pt1, pt2, radius, color, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(frame, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def glass_panel(frame, pt1, pt2, radius=16, base=CLR_PANEL, alpha=0.7, edge=None):
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = frame.shape[:2]
    x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return
    roi = frame[y1c:y2c, x1c:x2c]
    overlay = roi.copy()
    rounded_rect(overlay, (0, 0), (x2c - x1c, y2c - y1c), radius, base, -1)
    frame[y1c:y2c, x1c:x2c] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
    if edge:
        rounded_rect(frame, (x1, y1), (x2, y2), radius, edge, 1)


def pill_badge(frame, center, text, color, scale=0.5, pad_x=14, pad_y=8):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    cx, cy = center
    w = tw + pad_x * 2 + 16
    h = th + pad_y * 2
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2
    glass_panel(frame, (x1, y1), (x2, y2), radius=h // 2, base=(22, 16, 10), alpha=0.75, edge=(70, 60, 45))
    cv2.circle(frame, (x1 + pad_x + 4, cy), 5, color, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x1 + pad_x + 16, y2 - pad_y - 2), FONT, scale, color, 1, cv2.LINE_AA)


def draw_hand_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for connection in HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (90, 70, 50), 1, cv2.LINE_AA)
    for p in points:
        cv2.circle(frame, p, 2, (210, 200, 235), -1, cv2.LINE_AA)


def draw_active_box(frame, active_box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = active_box
    cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)),
                  (90, 80, 60), 1, cv2.LINE_AA)


def draw_scroll_guide(frame, cfg, hand_norm_y):
    """Visual reference for the joystick-style scroll zone: a still band
    around neutral (no scroll), and how far the hand currently is from it."""
    h, w = frame.shape[:2]
    neutral = cfg["scroll_neutral_y"]
    dz = cfg["scroll_deadzone_norm"]
    max_off = cfg["scroll_max_offset_norm"]

    def y_px(norm_y):
        return int(min(1.0, max(0.0, norm_y)) * h)

    top_dz, bot_dz = y_px(neutral - dz), y_px(neutral + dz)
    top_max, bot_max = y_px(neutral - max_off), y_px(neutral + max_off)

    cv2.line(frame, (0, top_max), (w, top_max), (90, 70, 50), 1, cv2.LINE_AA)
    cv2.line(frame, (0, bot_max), (w, bot_max), (90, 70, 50), 1, cv2.LINE_AA)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top_dz), (w, bot_dz), (60, 200, 200), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.line(frame, (0, y_px(neutral)), (w, y_px(neutral)), (60, 200, 200), 1, cv2.LINE_AA)

    hy = y_px(hand_norm_y)
    cv2.line(frame, (0, hy), (w, hy), CLR_SCROLL, 1, cv2.LINE_AA)
    cv2.circle(frame, (w - 20, hy), 5, CLR_SCROLL, -1, cv2.LINE_AA)


def draw_hud(frame, pose_label, pose_color, paused, fps, last_action, action_flash_t, pinch_debug=None):
    h, w = frame.shape[:2]

    status = "PAUSED (P to resume)" if paused else "ACTIVE (P to pause)"
    status_color = CLR_PAUSED if paused else CLR_CLICK
    pill_badge(frame, (100, 26), status, status_color, scale=0.45)

    fps_color = CLR_CLICK if fps > 25 else (CLR_ACCENT if fps > 15 else CLR_RCLICK)
    pill_badge(frame, (w - 70, 26), f"{fps:.0f} FPS", fps_color, scale=0.45)

    panel_top = h - 90
    glass_panel(frame, (10, panel_top), (w - 10, h - 10), radius=18, base=CLR_PANEL, alpha=0.72, edge=CLR_PANEL_ED)
    cv2.putText(frame, f"GESTURE: {pose_label}", (26, panel_top + 34), FONT, 0.6, pose_color, 1, cv2.LINE_AA)

    # Live pinch calibration readout: current smoothed distance vs. the
    # self-learned "click" threshold, so you can see click detection is
    # actually tracking your hand instead of guessing at a hidden number.
    if pinch_debug and "left" in pinch_debug:
        dist, close_th, _ = pinch_debug["left"]
        debug_txt = f"pinch: {dist:.2f} / click < {close_th:.2f}"
        debug_color = CLR_CLICK if dist < close_th else CLR_TEXT_DIM
        cv2.putText(frame, debug_txt, (26, panel_top + 58), FONT, 0.45, debug_color, 1, cv2.LINE_AA)

    if last_action and action_flash_t > 0:
        alpha = min(1.0, action_flash_t / 0.5)
        overlay = frame.copy()
        (tw, th), _ = cv2.getTextSize(last_action, FONT, 0.9, 2)
        cv2.putText(overlay, last_action, (w // 2 - tw // 2, panel_top - 16), FONT, 0.9, CLR_ACCENT, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, "Q  quit", (w - 90, h - 16), FONT, 0.4, CLR_TEXT_DIM, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    cfg = CFG
    raw_cap = open_camera(cfg)
    if raw_cap is None:
        print("[ERROR] Could not open any camera.")
        print("  -> Check gesture_config.json and your camera permissions.")
        return

    raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["target_width"])
    raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["target_height"])
    raw_cap.set(cv2.CAP_PROP_FPS, 60)
    raw_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap = ThreadedCamera(raw_cap)

    if not MODEL_PATH.exists():
        print(f"[INFO] Downloading hand model to {MODEL_PATH}...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=cfg["min_detection_conf"],
            min_hand_presence_confidence=cfg["min_tracking_conf"],
            min_tracking_confidence=cfg["min_tracking_conf"],
        )
    )

    controller = GestureController(cfg)

    prev_time = time.time()
    start_time = prev_time
    last_ts_ms = -1
    lost_frames = 0
    last_action = ""
    action_flash_t = 0.0

    print("=" * 55)
    print("  Gesture PC Control  |  Q to quit, P to pause")
    print("=" * 55)
    print("  Point (index only)      -> move cursor")
    print("  Thumb+Index pinch       -> click / drag")
    print("  Thumb+Middle pinch      -> right click")
    print("  Peace sign, move up/dn  -> scroll")
    print("  Both hands open, apart  -> zoom")
    print("  One hand open, swipe    -> back/forward (Alt+Left/Right)")
    print(f"  Screen resolution: {SCREEN_W}x{SCREEN_H}")
    print("=" * 55)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            if cfg["flip_camera"]:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = int((time.time() - start_time) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms
            results = hands.detect_for_video(mp.Image(mp.ImageFormat.SRGB, rgb), ts_ms)

            pose_label = "NO HAND"
            pose_color = CLR_IDLE
            draw_active_box(frame, cfg["active_box"])

            if results.hand_landmarks and results.handedness and not controller.paused:
                hand_data = {}
                for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                    label = handedness[0].display_name or handedness[0].category_name
                    draw_hand_landmarks(frame, hand_landmarks)
                    flags = finger_extended_flags(hand_landmarks)
                    hand_data[label] = (hand_landmarks, flags)

                was_lost = lost_frames > 0
                lost_frames = 0
                now = time.time()
                dt = max(1e-4, min(0.2, now - prev_time))  # clamp to avoid spikes on hitches

                # --- Two-hand gesture: zoom (both hands open) ---
                if "Left" in hand_data and "Right" in hand_data:
                    (l_lms, l_flags) = hand_data["Left"]
                    (r_lms, r_flags) = hand_data["Right"]
                    if is_open_hand(l_flags, cfg["open_finger_thresh"]) and is_open_hand(r_flags, cfg["open_finger_thresh"]):
                        lw, rw = l_lms[0], r_lms[0]
                        wrist_dist = math.hypot(lw.x - rw.x, lw.y - rw.y)
                        ticks = controller.update_zoom(wrist_dist)
                        pose_label, pose_color = "ZOOM", CLR_ZOOM
                        controller.reset_swipe()
                        if ticks:
                            last_action, action_flash_t = ("ZOOM IN" if ticks > 0 else "ZOOM OUT"), 0.5
                    else:
                        controller.reset_zoom()
                else:
                    controller.reset_zoom()

                # --- Single-hand gestures: use whichever hand is present (prefer Right) ---
                primary = hand_data.get("Right") or hand_data.get("Left")
                if primary and pose_label == "NO HAND":
                    lms, flags = primary
                    pose = classify_pose(flags)
                    index_tip, thumb_tip, middle_tip = lms[8], lms[4], lms[12]
                    scale = hand_scale(lms)  # wrist-to-knuckle distance, for scale-normalized pinches

                    if pose == "POINT":
                        if was_lost:
                            controller.reset_cursor_filter(index_tip.x, index_tip.y, now)
                        controller.move_cursor(index_tip.x, index_tip.y, now)
                        pose_label, pose_color = "MOVE", CLR_MOVE
                        controller.reset_scroll()
                        controller.reset_swipe()
                    elif pose == "PEACE":
                        mid_y = (index_tip.y + middle_tip.y) / 2
                        draw_scroll_guide(frame, cfg, mid_y)
                        ticks = controller.update_scroll(mid_y, dt)
                        pose_label, pose_color = "SCROLL", CLR_SCROLL
                        if ticks:
                            last_action, action_flash_t = ("SCROLL UP" if ticks > 0 else "SCROLL DOWN"), 0.4
                        controller.reset_swipe()
                    elif pose == "OPEN":
                        controller.reset_scroll()
                        wrist = lms[0]
                        result = controller.update_swipe(wrist.x, now)
                        pose_label, pose_color = "SWIPE READY", CLR_SWIPE
                        if result:
                            last_action, action_flash_t = result, 0.5
                            pose_label = result
                    else:
                        controller.reset_scroll()
                        controller.reset_swipe()
                        pose_label, pose_color = pose, CLR_IDLE

                    # Pinch detection works regardless of overall pose, using thumb
                    # distance normalized by hand scale (so it's consistent whether
                    # the hand is close to or far from the camera). Smoothing,
                    # hysteresis, and shake debouncing all happen inside the
                    # controller — this just supplies the normalized measurement.
                    li_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y) / scale
                    lm_dist = math.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y) / scale

                    result = controller.update_left_pinch(li_dist, now)
                    if result:
                        last_action, action_flash_t = result, 0.5
                        if result == "DRAGGING":
                            pose_label, pose_color = "DRAG", CLR_CLICK

                    result = controller.update_right_pinch(lm_dist, now)
                    if result:
                        last_action, action_flash_t = result, 0.5
                        pose_label, pose_color = "RIGHT CLICK", CLR_RCLICK
            else:
                lost_frames += 1
                if lost_frames >= cfg["grace_frames"]:
                    controller.release_all()

            now = time.time()
            frame_dt = now - prev_time
            fps = 1.0 / max(frame_dt, 1e-6)
            prev_time = now
            if action_flash_t > 0:
                action_flash_t = max(0.0, action_flash_t - frame_dt)

            if cfg.get("show_camera_window", True):
                draw_hud(frame, pose_label, pose_color, controller.paused, fps, last_action, action_flash_t,
                         controller.last_pinch_debug)
                cv2.imshow("Gesture PC Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('p'), ord('P')):
                controller.paused = not controller.paused
                if controller.paused:
                    controller.release_all()
                print(f"[INFO] {'Paused' if controller.paused else 'Resumed'}")

    finally:
        controller.release_all()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Stopped. All mouse buttons released.")


if __name__ == "__main__":
    main()