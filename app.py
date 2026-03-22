"""
VisionSound — AI Depth & Spatial Audio Assist
Two-thread architecture:
  Thread 1: reads raw camera frames as fast as possible (30+ FPS)
  Thread 2: runs MiDaS depth inference (~5-15 FPS on CPU)
  Thread 3: voice TTS alert thread (non-blocking, debounced)
  MJPEG stream: overlays latest depth result onto latest raw frame → smooth video
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import torch
import numpy as np
import threading
import time
import queue
import logging

# ── OPTIONAL: pyttsx3 for offline TTS ────────────────────────────────
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("pyttsx3 not found — install with: pip install pyttsx3")

# ── CONFIG ────────────────────────────────────────────────────────────
CAMERA_INDEX     = 1       # USB camera index
DANGER_THRESHOLD = 150     # depth 0–255, above = obstacle alert
WARN_THRESHOLD   = 100
STREAM_WIDTH     = 640
STREAM_HEIGHT    = 480
INFER_WIDTH      = 256     # resize to this before MiDaS (much faster)
INFER_HEIGHT     = 192
JPEG_QUALITY     = 75

# ── VOICE CONFIG ──────────────────────────────────────────────────────
VOICE_ENABLED        = True          # master switch
VOICE_COOLDOWN_SEC   = 3.0           # minimum seconds between voice alerts
VOICE_RATE           = 160           # words-per-minute (pyttsx3)
VOICE_VOLUME         = 1.0           # 0.0 – 1.0

# ── HAPTIC CONFIG ─────────────────────────────────────────────────────
HAPTIC_ENABLED         = True   # master switch (synced with frontend toggle)
HAPTIC_PULSE_MIN_MS    = 40     # vibration duration (ms) at danger threshold
HAPTIC_PULSE_MAX_MS    = 220    # vibration duration (ms) at max depth (255)
HAPTIC_INTERVAL_MIN_MS = 80     # fastest repeat interval (ms) — very close
HAPTIC_INTERVAL_MAX_MS = 700    # slowest repeat interval (ms) — just past threshold
HAPTIC_GAP_MS          = 60     # silence gap between pulses in a burst

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── SHARED STATE ─────────────────────────────────────────────────────
frame_lock  = threading.Lock()
depth_lock  = threading.Lock()

raw_frame   = None   # latest BGR frame from camera (numpy)
depth_frame = None   # latest coloured depth overlay (numpy)
latest_data = {"left": 0.0, "center": 0.0, "right": 0.0,
               "alert": False, "alert_direction": "", "fps": 0.0,
               "stream_fps": 0.0, "voice_enabled": VOICE_ENABLED,
               "haptic_enabled": HAPTIC_ENABLED,
               "haptic_pattern": [], "haptic_interval_ms": 0}
running     = True
cap         = None
cam_lock    = threading.Lock()

# ── VOICE ALERT STATE ─────────────────────────────────────────────────
speech_queue      = queue.Queue(maxsize=1)   # only keep the latest message
last_spoken_time  = 0.0
voice_lock        = threading.Lock()

# ── MODEL ─────────────────────────────────────────────────────────────
log.info("Loading MiDaS small …")
device = "cuda" if torch.cuda.is_available() else "cpu"
midas  = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas.eval()
if device == "cuda":
    midas = midas.cuda()
_t     = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
midas_transform = _t.small_transform
log.info(f"MiDaS ready on {device}")

# ── HELPERS ───────────────────────────────────────────────────────────
def depth_to_distance_label(value: float) -> str:
    """
    Converts a depth map value (0–255) to a human-readable distance label.
    Higher MiDaS values = closer object (inverse depth).
    """
    if value >= 220:
        return "extremely close"
    elif value >= 180:
        return "very close"
    elif value >= DANGER_THRESHOLD:
        return "close"
    elif value >= WARN_THRESHOLD:
        return "at medium distance"
    else:
        return "far away"

def build_voice_message(left: float, center: float, right: float,
                         direction: str, max_val: float) -> str:
    """Build a natural-language voice alert string."""
    label = depth_to_distance_label(max_val)

    if direction == "CENTER":
        return f"Warning! Obstacle directly ahead, {label}."
    elif direction == "LEFT":
        return f"Warning! Obstacle on your left, {label}."
    elif direction == "RIGHT":
        return f"Warning! Obstacle on your right, {label}."
    else:
        return f"Warning! Obstacle detected, {label}."

# ── HAPTIC HELPERS ────────────────────────────────────────────────────
def compute_haptic_params(max_val: float) -> tuple[int, int]:
    """
    Given the peak depth value (0–255), return (pulse_duration_ms, interval_ms).
    Both scale linearly between threshold and 255:
      pulse_duration: HAPTIC_PULSE_MIN_MS → HAPTIC_PULSE_MAX_MS  (longer = closer)
      interval:       HAPTIC_INTERVAL_MAX_MS → HAPTIC_INTERVAL_MIN_MS (faster = closer)
    Returns (0, 0) if below danger threshold.
    """
    if max_val <= DANGER_THRESHOLD:
        return 0, 0
    ratio        = min(1.0, (max_val - DANGER_THRESHOLD) / (255 - DANGER_THRESHOLD))
    pulse_dur    = int(HAPTIC_PULSE_MIN_MS    + ratio * (HAPTIC_PULSE_MAX_MS    - HAPTIC_PULSE_MIN_MS))
    interval_ms  = int(HAPTIC_INTERVAL_MAX_MS - ratio * (HAPTIC_INTERVAL_MAX_MS - HAPTIC_INTERVAL_MIN_MS))
    return pulse_dur, interval_ms

def build_haptic_pattern(direction: str, pulse_dur: int) -> list[int]:
    """
    Encode obstacle direction as a W3C Vibration API pattern (list of ints).
    The frontend passes this directly to navigator.vibrate().
      LEFT   → double pulse  [dur, gap, dur]
      CENTER → single pulse  [dur]
      RIGHT  → triple pulse  [dur, gap, dur, gap, dur]
    Gap scales with pulse_dur so the pattern feels proportional.
    """
    gap = max(30, pulse_dur // 2)
    if direction == "LEFT":
        return [pulse_dur, gap, pulse_dur]
    if direction == "RIGHT":
        return [pulse_dur, gap, pulse_dur, gap, pulse_dur]
    return [pulse_dur]   # CENTER — single decisive buzz

# ── THREAD 3: VOICE / TTS ─────────────────────────────────────────────
def voice_thread():
    """
    Dedicated thread that drains speech_queue and speaks via pyttsx3.
    pyttsx3 is NOT thread-safe — must be created inside this thread.
    """
    if not TTS_AVAILABLE:
        log.warning("Voice thread exiting — pyttsx3 not installed.")
        return

    engine = pyttsx3.init()
    engine.setProperty("rate",   VOICE_RATE)
    engine.setProperty("volume", VOICE_VOLUME)

    # Pick a clear female/male voice if available
    voices = engine.getProperty("voices")
    for v in voices:
        if "english" in v.name.lower() or "zira" in v.name.lower() \
                or "david" in v.name.lower():
            engine.setProperty("voice", v.id)
            break

    log.info("Voice thread ready.")

    while running:
        try:
            message = speech_queue.get(timeout=0.5)
            if message is None:            # shutdown sentinel
                break
            log.info(f"[VOICE] → {message}")
            engine.say(message)
            engine.runAndWait()
        except queue.Empty:
            continue
        except Exception as e:
            log.error(f"TTS error: {e}")

def enqueue_voice(message: str):
    """
    Thread-safe: push a message only if cooldown has passed.
    Drops old pending message and replaces with newer one (maxsize=1).
    """
    global last_spoken_time

    if not VOICE_ENABLED or not TTS_AVAILABLE:
        return

    now = time.time()
    with voice_lock:
        if now - last_spoken_time < VOICE_COOLDOWN_SEC:
            return                         # still in cooldown, skip
        last_spoken_time = now

    # Drain stale entry (if any) and push new message
    try:
        speech_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        speech_queue.put_nowait(message)
    except queue.Full:
        pass                               # already has one pending

# ── CAMERA ────────────────────────────────────────────────────────────
def open_camera(index: int):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]  # 0 = CAP_ANY
    for backend in backends:
        try:
            c = cv2.VideoCapture(index, backend) if backend != 0 else cv2.VideoCapture(index)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_WIDTH)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
                c.set(cv2.CAP_PROP_FPS, 30)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(3):  # drain buffer
                    c.read()
                log.info(f"Camera {index} opened (backend={backend})")
                return c
            c.release()
        except Exception as e:
            log.warning(f"Backend {backend} index {index} failed: {e}")
    log.error(f"Could not open camera {index}")
    return None

def get_available_cameras():
    available = []
    for i in range(5):
        try:
            c = cv2.VideoCapture(i)
            if c.isOpened():
                available.append(i)
            c.release()
        except Exception:
            pass
    return available if available else [0, 1]

# ── THREAD 1: CAMERA READER ───────────────────────────────────────────
def camera_thread():
    global raw_frame, running, cap
    fps_count = 0
    fps_time  = time.time()

    while running:
        with cam_lock:
            local_cap = cap

        if local_cap is None or not local_cap.isOpened():
            time.sleep(0.5)
            with cam_lock:
                new = open_camera(CAMERA_INDEX)
                if new:
                    cap = new
            continue

        ret, frame = local_cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        with frame_lock:
            raw_frame = frame

        fps_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            with depth_lock:
                latest_data["stream_fps"] = round(fps_count / (now - fps_time), 1)
            fps_count = 0
            fps_time  = now

# ── THREAD 2: DEPTH INFERENCE ─────────────────────────────────────────
@torch.inference_mode()
def run_midas(bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(bgr, (INFER_WIDTH, INFER_HEIGHT))
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    inp   = midas_transform(rgb)
    if device == "cuda":
        inp = inp.cuda()
    pred = midas(inp)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=(bgr.shape[0], bgr.shape[1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    d = pred.cpu().numpy()
    d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    return d

def inference_thread():
    global depth_frame, running
    fps_count = 0
    fps_time  = time.time()
    ifps      = 0.0

    while running:
        with frame_lock:
            frame = raw_frame

        if frame is None:
            time.sleep(0.05)
            continue

        frame = frame.copy()

        try:
            depth_map = run_midas(frame)
        except Exception as e:
            log.error(f"MiDaS error: {e}")
            time.sleep(0.1)
            continue

        h, w = depth_map.shape
        left   = float(np.mean(depth_map[:, :w // 3]))
        center = float(np.mean(depth_map[:, w // 3: 2 * w // 3]))
        right  = float(np.mean(depth_map[:, 2 * w // 3:]))

        max_val   = max(left, center, right)
        is_danger = max_val > DANGER_THRESHOLD
        direction = ""
        if is_danger:
            if left >= center and left >= right:
                direction = "LEFT"
            elif right >= center and right >= left:
                direction = "RIGHT"
            else:
                direction = "CENTER"

            # ── VOICE ALERT ───────────────────────────────────────────
            msg = build_voice_message(left, center, right, direction, max_val)
            enqueue_voice(msg)
            # ─────────────────────────────────────────────────────────

        # ── HAPTIC DATA (computed every frame, sent to frontend) ──────
        pulse_dur, haptic_interval = compute_haptic_params(max_val)
        haptic_pat = build_haptic_pattern(direction, pulse_dur) if is_danger else []

        fps_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            ifps = fps_count / (now - fps_time)
            fps_count = 0
            fps_time  = now

        # Build overlay: blend raw + depth colourmap
        coloured = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        coloured = cv2.addWeighted(frame, 0.3, coloured, 0.7, 0)

        cv2.line(coloured, (w // 3, 0),     (w // 3, h),     (255, 255, 255), 1)
        cv2.line(coloured, (2 * w // 3, 0), (2 * w // 3, h), (255, 255, 255), 1)

        def put(txt, x, y, col=(255, 255, 255)):
            cv2.putText(coloured, txt, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        lc = (60, 60, 255) if left   > DANGER_THRESHOLD else (80, 220, 80)
        cc = (60, 60, 255) if center > DANGER_THRESHOLD else (80, 220, 80)
        rc = (60, 60, 255) if right  > DANGER_THRESHOLD else (80, 220, 80)

        # Distance label on each zone
        put(f"L:{left:.0f} ({depth_to_distance_label(left)})",   8,  30, lc)
        put(f"C:{center:.0f} ({depth_to_distance_label(center)})", w // 2 - 70, 30, cc)
        put(f"R:{right:.0f} ({depth_to_distance_label(right)})",  2 * w // 3 + 8, 30, rc)
        put(f"DEPTH {ifps:.0f}fps", w - 105, h - 10, (180, 180, 180))

        # Status badges bottom-left: VOICE | HAPTIC
        v_col  = (0, 220, 80) if VOICE_ENABLED  else (100, 100, 100)
        h_col  = (0, 180, 255) if HAPTIC_ENABLED else (100, 100, 100)
        put("VOICE ON"  if VOICE_ENABLED  else "VOICE OFF",  8, h - 25, v_col)
        put("HAPTIC ON" if HAPTIC_ENABLED else "HAPTIC OFF", 8, h - 10, h_col)

        # Haptic intensity badge when actively vibrating
        if is_danger and HAPTIC_ENABLED:
            ratio_pct = int(min(100, (max_val - DANGER_THRESHOLD) / (255 - DANGER_THRESHOLD) * 100))
            put(f"VIBRATE {ratio_pct}%  {haptic_interval}ms", w // 2 - 70, h - 10, (0, 180, 255))

        if is_danger:
            cv2.rectangle(coloured, (0, 0), (w, h), (0, 0, 255), 4)
            put(f"!! OBSTACLE {direction} !!", w // 2 - 100, h // 2, (0, 0, 255))

        with depth_lock:
            depth_frame = coloured
            latest_data.update({
                "left":               round(left,   1),
                "center":             round(center, 1),
                "right":              round(right,  1),
                "alert":              is_danger,
                "alert_direction":    direction,
                "fps":                round(ifps, 1),
                "voice_enabled":      VOICE_ENABLED,
                "haptic_enabled":     HAPTIC_ENABLED,
                "haptic_pattern":     haptic_pat,        # e.g. [120, 60, 120]
                "haptic_interval_ms": haptic_interval,   # e.g. 350
                "distance_label":     depth_to_distance_label(max_val) if is_danger else "",
            })

# ── MJPEG GENERATOR ───────────────────────────────────────────────────
def mjpeg_generator():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    while True:
        with depth_lock:
            frame = depth_frame
        if frame is None:
            with frame_lock:
                frame = raw_frame
        if frame is None:
            time.sleep(0.03)
            continue
        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1 / 30)

# ── ROUTES ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/data")
def data():
    with depth_lock:
        return jsonify(latest_data)

@app.route("/cameras")
def cameras():
    return jsonify(get_available_cameras())

@app.route("/set_camera", methods=["POST"])
def set_camera():
    global cap, CAMERA_INDEX
    try:
        index = int(request.json["index"])
        new_cap = open_camera(index)
        if not new_cap:
            return jsonify({"status": "error", "message": f"Cannot open camera {index}"}), 400
        with cam_lock:
            if cap:
                cap.release()
            cap = new_cap
            CAMERA_INDEX = index
        return jsonify({"status": "ok", "camera": index})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/toggle_voice", methods=["POST"])
def toggle_voice():
    """Enable or disable voice alerts at runtime."""
    global VOICE_ENABLED
    body = request.get_json(silent=True) or {}
    if "enabled" in body:
        VOICE_ENABLED = bool(body["enabled"])
    else:
        VOICE_ENABLED = not VOICE_ENABLED          # simple toggle
    with depth_lock:
        latest_data["voice_enabled"] = VOICE_ENABLED
    log.info(f"Voice alerts {'ENABLED' if VOICE_ENABLED else 'DISABLED'}")
    return jsonify({"status": "ok", "voice_enabled": VOICE_ENABLED})

@app.route("/toggle_haptic", methods=["POST"])
def toggle_haptic():
    """Enable or disable haptic feedback at runtime (mirrors /toggle_voice)."""
    global HAPTIC_ENABLED
    body = request.get_json(silent=True) or {}
    if "enabled" in body:
        HAPTIC_ENABLED = bool(body["enabled"])
    else:
        HAPTIC_ENABLED = not HAPTIC_ENABLED
    with depth_lock:
        latest_data["haptic_enabled"] = HAPTIC_ENABLED
    log.info(f"Haptic feedback {'ENABLED' if HAPTIC_ENABLED else 'DISABLED'}")
    return jsonify({"status": "ok", "haptic_enabled": HAPTIC_ENABLED})

@app.route("/set_haptic_thresholds", methods=["POST"])
def set_haptic_thresholds():
    """
    Tune haptic intensity at runtime.
    Body (all optional):
      pulse_min_ms    — vibration duration at danger threshold (default 40)
      pulse_max_ms    — vibration duration at max depth        (default 220)
      interval_min_ms — fastest repeat interval, very close   (default 80)
      interval_max_ms — slowest repeat interval, at threshold (default 700)
      gap_ms          — silence gap between directional pulses (default 60)
    """
    global HAPTIC_PULSE_MIN_MS, HAPTIC_PULSE_MAX_MS
    global HAPTIC_INTERVAL_MIN_MS, HAPTIC_INTERVAL_MAX_MS, HAPTIC_GAP_MS
    try:
        body = request.get_json(silent=True) or {}
        if "pulse_min_ms"    in body: HAPTIC_PULSE_MIN_MS    = max(10,  int(body["pulse_min_ms"]))
        if "pulse_max_ms"    in body: HAPTIC_PULSE_MAX_MS    = max(50,  int(body["pulse_max_ms"]))
        if "interval_min_ms" in body: HAPTIC_INTERVAL_MIN_MS = max(50,  int(body["interval_min_ms"]))
        if "interval_max_ms" in body: HAPTIC_INTERVAL_MAX_MS = max(200, int(body["interval_max_ms"]))
        if "gap_ms"          in body: HAPTIC_GAP_MS          = max(20,  int(body["gap_ms"]))
        return jsonify({"status": "ok",
                        "pulse_min_ms":    HAPTIC_PULSE_MIN_MS,
                        "pulse_max_ms":    HAPTIC_PULSE_MAX_MS,
                        "interval_min_ms": HAPTIC_INTERVAL_MIN_MS,
                        "interval_max_ms": HAPTIC_INTERVAL_MAX_MS,
                        "gap_ms":          HAPTIC_GAP_MS})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/set_voice_cooldown", methods=["POST"])
def set_voice_cooldown():
    """Change how often voice speaks (seconds between alerts)."""
    global VOICE_COOLDOWN_SEC
    try:
        val = float(request.json["cooldown"])
        VOICE_COOLDOWN_SEC = max(0.5, min(val, 30.0))
        return jsonify({"status": "ok", "cooldown": VOICE_COOLDOWN_SEC})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/health")
def health():
    with depth_lock:
        d = dict(latest_data)
    return jsonify({"status": "ok", "device": device,
                    "tts_available": TTS_AVAILABLE,
                    "haptic_pulse_min_ms":    HAPTIC_PULSE_MIN_MS,
                    "haptic_pulse_max_ms":    HAPTIC_PULSE_MAX_MS,
                    "haptic_interval_min_ms": HAPTIC_INTERVAL_MIN_MS,
                    "haptic_interval_max_ms": HAPTIC_INTERVAL_MAX_MS,
                    **d})

# ── STARTUP ───────────────────────────────────────────────────────────
def startup():
    global cap
    log.info(f"Opening USB camera at index {CAMERA_INDEX} …")
    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        log.warning("Index 1 failed — trying index 0 …")
        cap = open_camera(0)

    threading.Thread(target=camera_thread,    daemon=True, name="CamReader").start()
    threading.Thread(target=inference_thread, daemon=True, name="DepthInfer").start()
    threading.Thread(target=voice_thread,     daemon=True, name="VoiceTTS").start()
    log.info("All threads started — visit http://localhost:5000")

startup()

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000,
                debug=False, threaded=True, use_reloader=False)
    finally:
        running = False
        speech_queue.put_nowait(None)   # graceful shutdown sentinel for TTS
        if cap:
            cap.release()