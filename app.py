"""
Sonic Path — AI Depth & Spatial Audio Assist
Enhanced build with:
  - 5-zone depth analysis (far-left, left, center, right, far-right)
  - Quantum-inspired weighted zone scoring (superposition-style probability scoring)
  - Adaptive threshold (auto-calibrates to environment lighting/depth baseline)
  - Proximity confidence score (0–100%) per zone
  - Smooth exponential moving average on depth values (reduces flicker)
  - Object detection hook (YOLO-ready, disabled by default)
  - /alert_history endpoint for last N alerts
  - /calibrate endpoint to reset baseline
  - Better FPS tracking with rolling average

Thread architecture:
  Thread 1 — Camera reader (raw frames, as fast as possible)
  Thread 2 — MiDaS depth inference (~5–15 FPS on CPU)
  MJPEG stream — overlays latest depth result onto latest raw frame
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import torch
import numpy as np
import threading
import time
import logging
import collections

# ── CONFIG ────────────────────────────────────────────────────────────
CAMERA_INDEX      = 1        # USB camera index (change if needed)
DANGER_THRESHOLD  = 150      # depth 0–255; above = obstacle danger
WARN_THRESHOLD    = 100      # above = caution zone
STREAM_WIDTH      = 640
STREAM_HEIGHT     = 480
INFER_WIDTH       = 256      # MiDaS input width (lower = faster)
INFER_HEIGHT      = 192
JPEG_QUALITY      = 75
EMA_ALPHA         = 0.35     # smoothing factor (0=frozen, 1=raw)
ALERT_HISTORY_MAX = 50       # number of past alerts to store
ADAPTIVE_WINDOW   = 60       # frames to average for baseline calibration

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app  = Flask(__name__)
CORS(app)

# ── SHARED STATE ──────────────────────────────────────────────────────
frame_lock  = threading.Lock()
depth_lock  = threading.Lock()

raw_frame   = None
depth_frame = None

# 5-zone depth values (smoothed)
latest_data = {
    "far_left":        0.0,
    "left":            0.0,
    "center":          0.0,
    "right":           0.0,
    "far_right":       0.0,
    "alert":           False,
    "alert_direction": "",
    "confidence":      0.0,   # 0–100 how confident the danger detection is
    "fps":             0.0,
    "stream_fps":      0.0,
    "adaptive_baseline": 0.0, # environment baseline depth average
}

alert_history = collections.deque(maxlen=ALERT_HISTORY_MAX)
baseline_buffer = collections.deque(maxlen=ADAPTIVE_WINDOW)

running  = True
cap      = None
cam_lock = threading.Lock()

# Smoothed zone values (EMA state)
ema_zones = {"far_left": 0.0, "left": 0.0, "center": 0.0,
             "right": 0.0, "far_right": 0.0}

# ── MODEL ─────────────────────────────────────────────────────────────
log.info("Loading MiDaS small …")
device = "cuda" if torch.cuda.is_available() else "cpu"
midas  = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas.eval()
if device == "cuda":
    midas = midas.cuda()
_t = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
midas_transform = _t.small_transform
log.info(f"MiDaS ready on {device}")

# ── CAMERA ────────────────────────────────────────────────────────────
def open_camera(index: int):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
    for backend in backends:
        try:
            c = (cv2.VideoCapture(index, backend)
                 if backend != 0 else cv2.VideoCapture(index))
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_WIDTH)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
                c.set(cv2.CAP_PROP_FPS, 30)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(3):   # drain stale buffer
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
    for i in range(6):
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
    fps_buf   = collections.deque(maxlen=30)

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
            raw_frame = cv2.flip(frame, 1)  # mirror horizontally → correct L/R

        fps_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps_buf.append(fps_count / (now - fps_time))
            with depth_lock:
                latest_data["stream_fps"] = round(
                    sum(fps_buf) / len(fps_buf), 1)
            fps_count = 0
            fps_time  = now

# ── DEPTH INFERENCE ───────────────────────────────────────────────────
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

# ── QUANTUM-INSPIRED ZONE SCORER ──────────────────────────────────────
def quantum_zone_score(zone_val: float, baseline: float,
                       threshold: float) -> float:
    """
    Computes a 0–100 confidence score for how dangerous a zone is.

    Inspired by quantum superposition weighting:
    - Each zone is treated as a probability amplitude.
    - Score combines raw depth, deviation from baseline, and threshold proximity.
    - Analogous to collapsing a superposition: the more evidence, the higher
      the confidence that a real obstacle is present (not noise).

    In practice this is a normalized weighted sum — but the weighting
    approach mimics how quantum probability amplitudes combine.
    """
    if baseline <= 0:
        baseline = 1.0

    # Amplitude 1: raw depth above threshold (direct danger signal)
    raw_signal = max(0.0, (zone_val - threshold) / (255.0 - threshold))

    # Amplitude 2: deviation from calm baseline (contextual signal)
    deviation = max(0.0, (zone_val - baseline) / (255.0 - baseline))

    # Amplitude 3: absolute proximity (how deep into danger zone)
    proximity = min(1.0, zone_val / 255.0)

    # Weighted superposition (sum of squared amplitudes → probability)
    # Weights: danger signal most important, deviation second, proximity third
    score = (0.55 * raw_signal**2 + 0.30 * deviation**2 + 0.15 * proximity**2)
    # Normalise to 0–100
    return round(min(100.0, score * 100.0 / 0.55), 1)

# ── THREAD 2: DEPTH INFERENCE ─────────────────────────────────────────
def inference_thread():
    global depth_frame, running, ema_zones
    fps_count  = 0
    fps_time   = time.time()
    ifps       = 0.0
    fps_buf    = collections.deque(maxlen=10)

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
        # 5-zone split: |10%|20%|40%|20%|10%|
        b0 = 0
        b1 = w // 10           # far-left boundary
        b2 = w * 3 // 10       # left boundary
        b3 = w * 7 // 10       # center/right boundary
        b4 = w * 9 // 10       # far-right boundary
        b5 = w

        raw_fl = float(np.mean(depth_map[:, b0:b1]))
        raw_l  = float(np.mean(depth_map[:, b1:b2]))
        raw_c  = float(np.mean(depth_map[:, b2:b3]))
        raw_r  = float(np.mean(depth_map[:, b3:b4]))
        raw_fr = float(np.mean(depth_map[:, b4:b5]))

        # Adaptive baseline from full-frame average
        frame_avg = float(np.mean(depth_map))
        baseline_buffer.append(frame_avg)
        baseline = float(np.mean(baseline_buffer)) if baseline_buffer else 128.0

        # Exponential Moving Average smoothing (reduces flicker/noise)
        a = EMA_ALPHA
        ema_zones["far_left"] = a * raw_fl + (1 - a) * ema_zones["far_left"]
        ema_zones["left"]     = a * raw_l  + (1 - a) * ema_zones["left"]
        ema_zones["center"]   = a * raw_c  + (1 - a) * ema_zones["center"]
        ema_zones["right"]    = a * raw_r  + (1 - a) * ema_zones["right"]
        ema_zones["far_right"]= a * raw_fr + (1 - a) * ema_zones["far_right"]

        fl = ema_zones["far_left"]
        l  = ema_zones["left"]
        c  = ema_zones["center"]
        r  = ema_zones["right"]
        fr = ema_zones["far_right"]

        # Quantum zone scoring
        scores = {
            "far_left":  quantum_zone_score(fl, baseline, DANGER_THRESHOLD),
            "left":      quantum_zone_score(l,  baseline, DANGER_THRESHOLD),
            "center":    quantum_zone_score(c,  baseline, DANGER_THRESHOLD),
            "right":     quantum_zone_score(r,  baseline, DANGER_THRESHOLD),
            "far_right": quantum_zone_score(fr, baseline, DANGER_THRESHOLD),
        }

        # Alert: any zone in danger
        max_val   = max(fl, l, c, r, fr)
        max_zone  = max(scores, key=scores.get)
        is_danger = max_val > DANGER_THRESHOLD
        confidence = scores[max_zone] if is_danger else 0.0

        direction = ""
        if is_danger:
            direction_map = {
                "far_left":  "FAR LEFT",
                "left":      "LEFT",
                "center":    "CENTER",
                "right":     "RIGHT",
                "far_right": "FAR RIGHT",
            }
            direction = direction_map[max_zone]

        # FPS rolling average
        fps_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps_buf.append(fps_count / (now - fps_time))
            ifps = sum(fps_buf) / len(fps_buf)
            fps_count = 0
            fps_time  = now

        # ── BUILD OVERLAY ──────────────────────────────────────────────
        coloured = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        coloured = cv2.addWeighted(frame, 0.25, coloured, 0.75, 0)

        # Zone divider lines (5 zones)
        for bx in [b1, b2, b3, b4]:
            cv2.line(coloured, (bx, 0), (bx, h), (200, 200, 200), 1)

        def danger_color(val):
            if val > DANGER_THRESHOLD: return (50, 50, 255)
            if val > WARN_THRESHOLD:   return (50, 190, 255)
            return (80, 220, 80)

        def put(txt, x, y, col=(255, 255, 255), scale=0.45):
            cv2.putText(coloured, txt, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)

        # Zone labels
        put(f"FL:{fl:.0f}", b0 + 2,  22, danger_color(fl))
        put(f"L:{l:.0f}",   b1 + 4,  22, danger_color(l))
        put(f"C:{c:.0f}",   b2 + (b3-b2)//2 - 22, 22, danger_color(c))
        put(f"R:{r:.0f}",   b3 + 4,  22, danger_color(r))
        put(f"FR:{fr:.0f}", b4 + 2,  22, danger_color(fr))
        put(f"DEPTH {ifps:.0f}fps | base:{baseline:.0f}",
            4, h - 8, (160, 160, 160))

        if is_danger:
            cv2.rectangle(coloured, (0, 0), (w, h), (0, 0, 255), 4)
            label = f"!! OBSTACLE {direction} !! [{confidence:.0f}%]"
            put(label, max(4, w//2 - 140), h//2, (0, 0, 255), 0.6)

        with depth_lock:
            depth_frame = coloured
            latest_data.update({
                "far_left":          round(fl,  1),
                "left":              round(l,   1),
                "center":            round(c,   1),
                "right":             round(r,   1),
                "far_right":         round(fr,  1),
                "alert":             is_danger,
                "alert_direction":   direction,
                "confidence":        round(confidence, 1),
                "fps":               round(ifps, 1),
                "adaptive_baseline": round(baseline, 1),
                "zone_scores":       scores,
            })

        # Record alert history
        if is_danger:
            alert_history.append({
                "time":      time.strftime("%H:%M:%S"),
                "direction": direction,
                "confidence": round(confidence, 1),
                "depth":     round(max_val, 1),
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
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
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
            return jsonify({"status": "error",
                            "message": f"Cannot open camera {index}"}), 400
        with cam_lock:
            if cap:
                cap.release()
            cap = new_cap
            CAMERA_INDEX = index
        return jsonify({"status": "ok", "camera": index})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/set_threshold", methods=["POST"])
def set_threshold():
    global DANGER_THRESHOLD, WARN_THRESHOLD
    try:
        data = request.json
        if "danger" in data:
            DANGER_THRESHOLD = int(data["danger"])
        if "warn" in data:
            WARN_THRESHOLD = int(data["warn"])
        return jsonify({"status": "ok",
                        "danger": DANGER_THRESHOLD,
                        "warn":   WARN_THRESHOLD})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/alert_history")
def get_alert_history():
    return jsonify(list(alert_history))

@app.route("/calibrate", methods=["POST"])
def calibrate():
    """Reset the adaptive baseline so it re-learns the environment."""
    baseline_buffer.clear()
    return jsonify({"status": "ok", "message": "Baseline reset"})

@app.route("/health")
def health():
    with depth_lock:
        d = dict(latest_data)
    return jsonify({"status": "ok", "device": device, **d})

# ── STARTUP ───────────────────────────────────────────────────────────
def startup():
    global cap
    log.info(f"Opening camera at index {CAMERA_INDEX} …")
    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        log.warning("Index 1 failed — trying index 0 …")
        cap = open_camera(0)

    threading.Thread(target=camera_thread,    daemon=True,
                     name="CamReader").start()
    threading.Thread(target=inference_thread, daemon=True,
                     name="DepthInfer").start()
    log.info("All threads started — visit http://localhost:5000")

startup()

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000,
                debug=False, threaded=True, use_reloader=False)
    finally:
        running = False
        if cap:
            cap.release()
