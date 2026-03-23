"""
Driver Drowsiness Detection - Flask App
Provides:
  GET  /                  → dashboard UI
  GET  /video_feed        → MJPEG stream (annotated frames)
  POST /start             → start detection  (JSON body: {source, ip_url})
  POST /stop              → stop detection
  GET  /status            → current alert + session stats (JSON)
  GET  /health            → liveness check (JSON)
"""

import cv2
import numpy as np
import datetime
import threading
import time
from collections import deque
from pathlib import Path
from flask import Flask, Response, render_template_string, jsonify, request

# ── Try to import YOLO; graceful message if ultralytics not installed ──────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ─── Constants (same as Streamlit version) ────────────────────────────────────
FPS                   = 30
QUEUE_DURATION        = 2
DROWSY_THRESHOLD_FRAC = 0.8
YAWN_THRESHOLD_FRAC   = 1.0
HEAD_THRESHOLD_FRAC   = 0.8
WARNING_DURATION      = 2
CONFIDENCE_THRESH     = 0.5
MODEL_PATH            = "best.pt"

EYE_CLOSED_LABELS  = {0, 1, 2}
HEAD_MOTION_LABELS = {4, 5}
YAWN_LABEL         = 8

CLASS_NAMES = {
    0: "Closed_Eyes_L",  1: "Closed_Eyes_R", 2: "Closed_Eyes",
    3: "Open_Eyes",      4: "Head_Up",        5: "Head_Down",
    6: "Face",           7: "Open_Mouth",     8: "Yawn",
}

# ─── Global application state ─────────────────────────────────────────────────
app_state = {
    "running":     False,
    "thread":      None,
    "latest_jpg":  None,   # bytes of the latest JPEG frame
    "lock":        threading.Lock(),
    "status": {
        "drowsy": False, "yawn": False, "head": False,
        "drowsy_active": False, "yawn_active": False, "head_active": False,
        "total_drowsy": 0, "total_yawn": 0, "total_head": 0,
        "frame_count": 0,
    },
}

# ─── Detector class ───────────────────────────────────────────────────────────
class DrowsinessDetector:
    def __init__(self, fps=30.0):
        self.fps = fps
        q = int(fps * QUEUE_DURATION)
        self.eye_q  = deque(maxlen=q)
        self.yawn_q = deque(maxlen=q)
        self.head_q = deque(maxlen=q)

        self.drowsy_thr = int(fps * QUEUE_DURATION * DROWSY_THRESHOLD_FRAC)
        self.yawn_thr   = int(fps * QUEUE_DURATION * YAWN_THRESHOLD_FRAC)
        self.head_thr   = int(fps * QUEUE_DURATION * HEAD_THRESHOLD_FRAC)

        self.drowsy_ts = self.yawn_ts = self.head_ts = None
        self.total_drowsy = self.total_yawn = self.total_head = 0
        self.frame_count  = 0

    def process(self, frame, model):
        self.frame_count += 1
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=[rgb], save=False, verbose=False)[0]

        cur_eye = cur_yawn = cur_head = False
        dets = []

        xyxy    = results.boxes.xyxy.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            if confs[i] < CONFIDENCE_THRESH:
                continue
            label = int(classes[i])
            xmin, ymin, xmax, ymax = map(int, xyxy[i])
            dets.append((xmin, ymin, xmax, ymax, label, float(confs[i])))
            if label in EYE_CLOSED_LABELS:  cur_eye  = True
            if label in HEAD_MOTION_LABELS: cur_head = True
            if label == YAWN_LABEL:         cur_yawn = True

        self.eye_q.append(cur_eye)
        self.yawn_q.append(cur_yawn)
        self.head_q.append(cur_head)

        now = datetime.datetime.now()

        is_drowsy = sum(self.eye_q)  >= self.drowsy_thr
        is_yawn   = sum(self.yawn_q) >= self.yawn_thr
        is_head   = sum(self.head_q) >= self.head_thr

        if is_drowsy:
            if self.drowsy_ts is None: self.total_drowsy += 1
            self.drowsy_ts = now
        if is_yawn:
            if self.yawn_ts is None: self.total_yawn += 1
            self.yawn_ts = now
            self.yawn_q.clear()
        if is_head:
            if self.head_ts is None: self.total_head += 1
            self.head_ts = now
            self.head_q.clear()

        def active(ts):
            return ts and (now - ts).total_seconds() < WARNING_DURATION

        # ── Draw bounding boxes ──
        out = frame.copy()
        for (xmin, ymin, xmax, ymax, label, conf) in dets:
            name = CLASS_NAMES.get(label, str(label))
            if label in EYE_CLOSED_LABELS:
                color = (0, 0, 255) if is_drowsy else (0, 220, 0)
            elif label in HEAD_MOTION_LABELS or label == YAWN_LABEL:
                color = (0, 220, 220)
            else:
                color = (0, 200, 0)
            cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(out, f"{name} {conf:.2f}",
                        (xmin, max(ymin - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        # ── On-frame text overlays ──
        if active(self.drowsy_ts):
            cv2.putText(out, "WARNING: DROWSY!",    (20, 140),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 255), 2)
        if active(self.yawn_ts):
            cv2.putText(out, "WARNING: YAWNING!",   (20, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 200, 255), 2)
        if active(self.head_ts):
            cv2.putText(out, "WARNING: HEAD MOVE!", (20, 95),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 200, 255), 2)

        status = {
            "drowsy": is_drowsy, "yawn": is_yawn, "head": is_head,
            "drowsy_active": active(self.drowsy_ts),
            "yawn_active":   active(self.yawn_ts),
            "head_active":   active(self.head_ts),
            "total_drowsy":  self.total_drowsy,
            "total_yawn":    self.total_yawn,
            "total_head":    self.total_head,
            "frame_count":   self.frame_count,
        }
        return out, status


# ─── Background capture thread ────────────────────────────────────────────────
def capture_loop(source, model):
    """Run in a daemon thread. Writes annotated JPEGs into app_state."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[DMS] Cannot open source: {source}")
        app_state["running"] = False
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    detector = DrowsinessDetector(fps=fps)

    while app_state["running"]:
        ret, frame = cap.read()
        if not ret:
            print("[DMS] Stream ended.")
            break

        annotated, status = detector.process(frame, model)

        _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with app_state["lock"]:
            app_state["latest_jpg"] = jpg.tobytes()
            app_state["status"]     = status

    cap.release()
    app_state["running"] = False
    print("[DMS] Capture thread stopped.")


# ─── Flask app ────────────────────────────────────────────────────────────────
flask_app = Flask(__name__)

# ─── Dashboard HTML (single-file, inline JS polling) ─────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SafeDrive DMS</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; color: #eee; font-family: 'Segoe UI', sans-serif; }
  header { background: #111; padding: 1rem 2rem; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.5rem; }
  .badge { padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.78rem;
           font-weight: 700; }
  .badge-on  { background:#00c85133; border:1px solid #00c851; color:#00c851; }
  .badge-off { background:#ff4b4b33; border:1px solid #ff4b4b; color:#ff4b4b; }

  .container { display: flex; gap: 1.5rem; padding: 1.5rem; flex-wrap: wrap; }
  .video-col { flex: 3; min-width: 320px; }
  .side-col  { flex: 1; min-width: 220px; display: flex; flex-direction: column; gap: 1rem; }

  .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px; padding: 1rem; }
  .card h3 { font-size: 0.9rem; color: #aaa; margin-bottom: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }

  img#stream { width: 100%; border-radius: 8px; display: block; }

  .alert { padding: 0.6rem 1rem; border-radius: 8px; font-weight: 600;
           margin-bottom: 0.5rem; font-size: 0.95rem; transition: all 0.3s; }
  .alert-ok      { background:#00c85122; border:1px solid #00c851; color:#00c851; }
  .alert-drowsy  { background:#ff4b4b22; border:1px solid #ff4b4b; color:#ff4b4b; }
  .alert-yawn    { background:#ffa50022; border:1px solid #ffa500; color:#ffa500; }
  .alert-head    { background:#ffd70022; border:1px solid #ffd700; color:#ffd700; }

  .stat { margin-bottom: 0.75rem; }
  .stat-label { font-size: 0.75rem; color: #888; }
  .stat-val   { font-size: 1.4rem; font-weight: 700; }

  .controls { display: flex; gap: 0.75rem; flex-wrap: wrap; }
  input[type=text] { flex: 1; padding: 0.5rem 0.75rem; border-radius: 6px;
                     background: #2a2a2a; border: 1px solid #444; color: #eee;
                     font-size: 0.9rem; min-width: 160px; }
  select { padding: 0.5rem; border-radius: 6px; background: #2a2a2a;
           border: 1px solid #444; color: #eee; font-size: 0.9rem; }
  button { padding: 0.5rem 1.2rem; border-radius: 6px; font-weight: 600;
           cursor: pointer; font-size: 0.9rem; border: none; }
  .btn-start { background: #00c851; color: #000; }
  .btn-stop  { background: #ff4b4b; color: #fff; }
  .btn-start:hover { background: #00a040; }
  .btn-stop:hover  { background: #cc3333; }
</style>
</head>
<body>
<header>
  <span style="font-size:1.8rem">🚗</span>
  <h1>SafeDrive — Driver Monitoring System</h1>
  <span id="status-badge" class="badge badge-off">STOPPED</span>
</header>

<div class="container">
  <!-- Video feed -->
  <div class="video-col">
    <div class="card">
      <h3>Live Feed</h3>
      <img id="stream" src="/video_feed" alt="Waiting for stream…">
    </div>
    <!-- Controls -->
    <div class="card" style="margin-top:1rem">
      <h3>Controls</h3>
      <div class="controls" style="margin-bottom:0.75rem">
        <select id="src-type">
          <option value="webcam">Webcam (local)</option>
          <option value="ip">IP Webcam URL</option>
        </select>
        <input type="text" id="ip-url" placeholder="http://192.168.x.x:8080/video"
               style="display:none">
      </div>
      <div class="controls">
        <button class="btn-start" onclick="startDetection()">▶ Start</button>
        <button class="btn-stop"  onclick="stopDetection()">⏹ Stop</button>
      </div>
    </div>
  </div>

  <!-- Side panel -->
  <div class="side-col">
    <!-- Alerts -->
    <div class="card">
      <h3>Live Alerts</h3>
      <div id="alerts-panel">
        <div class="alert alert-ok">✅ Driver Alert</div>
      </div>
    </div>
    <!-- Stats -->
    <div class="card">
      <h3>Session Stats</h3>
      <div class="stat">
        <div class="stat-label">Frames processed</div>
        <div class="stat-val" id="s-frames">0</div>
      </div>
      <div class="stat">
        <div class="stat-label">Drowsy events</div>
        <div class="stat-val" style="color:#ff4b4b" id="s-drowsy">0</div>
      </div>
      <div class="stat">
        <div class="stat-label">Yawn events</div>
        <div class="stat-val" style="color:#ffa500" id="s-yawn">0</div>
      </div>
      <div class="stat">
        <div class="stat-label">Head-movement events</div>
        <div class="stat-val" style="color:#ffd700" id="s-head">0</div>
      </div>
    </div>
  </div>
</div>

<script>
const srcSelect = document.getElementById('src-type');
const ipInput   = document.getElementById('ip-url');
srcSelect.addEventListener('change', () => {
  ipInput.style.display = srcSelect.value === 'ip' ? 'block' : 'none';
});

async function startDetection() {
  const source = srcSelect.value === 'ip' ? ipInput.value : '0';
  await fetch('/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ source })
  });
  document.getElementById('stream').src = '/video_feed?' + Date.now();
}

async function stopDetection() {
  await fetch('/stop', { method: 'POST' });
}

// Poll /status every 500 ms and update UI
setInterval(async () => {
  try {
    const r = await fetch('/status');
    const d = await r.json();

    // Badge
    const badge = document.getElementById('status-badge');
    badge.textContent   = d.running ? 'RUNNING' : 'STOPPED';
    badge.className     = 'badge ' + (d.running ? 'badge-on' : 'badge-off');

    // Stats
    document.getElementById('s-frames').textContent = d.frame_count  || 0;
    document.getElementById('s-drowsy').textContent = d.total_drowsy || 0;
    document.getElementById('s-yawn').textContent   = d.total_yawn   || 0;
    document.getElementById('s-head').textContent   = d.total_head   || 0;

    // Alerts
    let html = '';
    if (d.drowsy_active) html += '<div class="alert alert-drowsy">😴 DROWSY DETECTED</div>';
    if (d.yawn_active)   html += '<div class="alert alert-yawn">🥱 YAWNING DETECTED</div>';
    if (d.head_active)   html += '<div class="alert alert-head">⬆ HEAD MOVEMENT</div>';
    if (!html)           html  = '<div class="alert alert-ok">✅ Driver Alert</div>';
    document.getElementById('alerts-panel').innerHTML = html;
  } catch(e) {}
}, 500);
</script>
</body>
</html>
"""


@flask_app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@flask_app.route("/health")
def health():
    return jsonify({"status": "ok", "yolo_available": YOLO_AVAILABLE})


@flask_app.route("/start", methods=["POST"])
def start():
    if app_state["running"]:
        return jsonify({"message": "Already running"}), 400
    if not YOLO_AVAILABLE:
        return jsonify({"error": "ultralytics not installed"}), 500

    data   = request.get_json(silent=True) or {}
    source = data.get("source", "0")
    # Convert "0" string → integer 0 for cv2
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # keep as string (IP URL)

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        return jsonify({"error": f"Model load failed: {e}"}), 500

    app_state["running"] = True
    t = threading.Thread(target=capture_loop, args=(source, model), daemon=True)
    t.start()
    app_state["thread"] = t
    return jsonify({"message": "Detection started", "source": str(source)})


@flask_app.route("/stop", methods=["POST"])
def stop():
    app_state["running"] = False
    return jsonify({"message": "Detection stopped"})


@flask_app.route("/status")
def status():
    with app_state["lock"]:
        s = dict(app_state["status"])
    s["running"] = app_state["running"]
    return jsonify(s)


def gen_frames():
    """Generator for MJPEG streaming."""
    placeholder = None  # lazy-create a "waiting" frame
    while True:
        with app_state["lock"]:
            jpg = app_state["latest_jpg"]

        if jpg is None:
            # Emit a dark placeholder frame
            if placeholder is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for stream...", (120, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
                _, enc = cv2.imencode(".jpg", blank)
                placeholder = enc.tobytes()
            jpg = placeholder

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(1 / FPS)


@flask_app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
