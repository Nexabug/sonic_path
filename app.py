"""
VisionSound — AI Depth & Spatial Audio Assist
Two-thread architecture:
  Thread 1: reads raw camera frames as fast as possible (30+ FPS)
  Thread 2: runs MiDaS depth inference (~5-15 FPS on CPU)
  MJPEG stream: overlays latest depth result onto latest raw frame → smooth video
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import torch
import numpy as np
import threading
import time
import logging

# ── CONFIG ────────────────────────────────────────────────────────────
CAMERA_INDEX     = 1       # USB camera index
DANGER_THRESHOLD = 150     # depth 0–255, above = obstacle alert
WARN_THRESHOLD   = 100
STREAM_WIDTH     = 640
STREAM_HEIGHT    = 480
INFER_WIDTH      = 256     # resize to this before MiDaS (much faster)
INFER_HEIGHT     = 192
JPEG_QUALITY     = 75

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
               "stream_fps": 0.0}
running     = True
cap         = None
cam_lock    = threading.Lock()

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

        put(f"L:{left:.0f}",   8,  30, lc)
        put(f"C:{center:.0f}", w // 2 - 28, 30, cc)
        put(f"R:{right:.0f}",  2 * w // 3 + 8, 30, rc)
        put(f"DEPTH {ifps:.0f}fps", w - 105, h - 10, (180, 180, 180))

        if is_danger:
            cv2.rectangle(coloured, (0, 0), (w, h), (0, 0, 255), 4)
            put(f"!! OBSTACLE {direction} !!", w // 2 - 100, h // 2, (0, 0, 255))

        with depth_lock:
            depth_frame = coloured
            latest_data.update({
                "left":            round(left,   1),
                "center":          round(center, 1),
                "right":           round(right,  1),
                "alert":           is_danger,
                "alert_direction": direction,
                "fps":             round(ifps, 1),
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

@app.route("/health")
def health():
    with depth_lock:
        d = dict(latest_data)
    return jsonify({"status": "ok", "device": device, **d})

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