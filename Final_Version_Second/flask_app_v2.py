"""
SafeDrive DMS — Flask App (v2)
────────────────────────────────────────────────────────────────────────────────
Alarms for:  Drowsy · Looking Left · Looking Right · Yawning
Real class map from best.pt:
  0: eyes_closed          → drowsy cue
  1: eyes_closed_head_left  → drowsy cue
  2: eyes_closed_head_right → drowsy cue
  3: focused
  4: head_down            → head movement
  5: head_up              → head movement
  6: seeing_left          → gaze-left alarm
  7: seeing_right         → gaze-right alarm
  8: yarning              → yawn alarm  (labelled 'yarning' in dataset)

Endpoints:
  GET  /                → dashboard
  GET  /video_feed      → MJPEG stream
  POST /start           → { "source": "0" | "url" }
  POST /stop
  GET  /status          → JSON alert + stats
  GET  /health
"""

import cv2, numpy as np, datetime, threading, time
from collections import deque
from flask import Flask, Response, render_template_string, jsonify, request

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

# ── Constants ──────────────────────────────────────────────────────────────────
FPS          = 30
QUEUE_SEC    = 2
CONF_THRESH  = 0.5
WARNING_SEC  = 2
DROWSY_FRAC  = 0.80
GAZE_FRAC    = 0.80
YAWN_FRAC    = 1.00
MODEL_PATH   = "best.pt"
ALARM_WAV    = "alarm.wav"

CLASS_NAMES = {
    0:"eyes_closed", 1:"eyes_closed_head_left", 2:"eyes_closed_head_right",
    3:"focused",     4:"head_down",              5:"head_up",
    6:"seeing_left", 7:"seeing_right",           8:"yarning",
}
DROWSY_LABELS = {0, 1, 2}
GAZE_L = 6
GAZE_R = 7
YAWN   = 8
HEAD_LABELS = {4, 5}

COLOR_OK     = (0, 200, 80)
COLOR_DROWSY = (0,  40, 255)
COLOR_GAZE   = (0, 160, 255)
COLOR_YAWN   = (0, 210, 255)
COLOR_HEAD   = (180, 80, 255)

# ── Alarm thread ──────────────────────────────────────────────────────────────
_alarm_lock   = threading.Lock()
_alarm_active = False

def _alarm_worker(duration_ms):
    global _alarm_active
    try:
        import pygame
        pygame.mixer.init()
        snd = pygame.mixer.Sound(ALARM_WAV)
        snd.play(maxtime=duration_ms)
        time.sleep(duration_ms / 1000.0)
    except Exception:
        pass
    finally:
        with _alarm_lock:
            _alarm_active = False

def fire_alarm(duration_ms: int):
    global _alarm_active
    with _alarm_lock:
        if _alarm_active:
            return
        _alarm_active = True
    threading.Thread(target=_alarm_worker, args=(duration_ms,), daemon=True).start()


# ── Detector ──────────────────────────────────────────────────────────────────
class DrowsinessDetector:
    def __init__(self, fps=30.0):
        q = int(fps * QUEUE_SEC)
        self.drowsy_q = deque(maxlen=q)
        self.gaze_l_q = deque(maxlen=q)
        self.gaze_r_q = deque(maxlen=q)
        self.yawn_q   = deque(maxlen=q)
        self.head_q   = deque(maxlen=q)

        self.drowsy_thr = int(fps * QUEUE_SEC * DROWSY_FRAC)
        self.gaze_thr   = int(fps * QUEUE_SEC * GAZE_FRAC)
        self.yawn_thr   = int(fps * QUEUE_SEC * YAWN_FRAC)
        self.head_thr   = int(fps * QUEUE_SEC * DROWSY_FRAC)

        self.ts = {k: None for k in ("drowsy","gaze_l","gaze_r","yawn","head")}
        self.alarm_until = {k: None for k in ("drowsy","gaze_l","gaze_r","yawn")}
        self.cnt = {k: 0 for k in ("drowsy","gaze_l","gaze_r","yawn","head")}
        self.frame_n = 0

    def _active(self, key):
        ts = self.ts[key]
        return ts is not None and (datetime.datetime.now() - ts).total_seconds() < WARNING_SEC

    def _fire(self, key, ms):
        now = datetime.datetime.now()
        until = self.alarm_until[key]
        if until is None or now >= until:
            fire_alarm(ms)
            self.alarm_until[key] = now + datetime.timedelta(milliseconds=ms)

    def process(self, frame, model):
        self.frame_n += 1
        now = datetime.datetime.now()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=[rgb], save=False, verbose=False)[0]

        cur = {k: False for k in ("drowsy","gaze_l","gaze_r","yawn","head")}
        dets = []

        if len(results.boxes):
            xyxy    = results.boxes.xyxy.cpu().numpy()
            confs   = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                if confs[i] < CONF_THRESH: continue
                lbl = int(classes[i])
                x0,y0,x1,y1 = map(int, xyxy[i])
                dets.append((x0,y0,x1,y1, lbl, float(confs[i])))
                if lbl in DROWSY_LABELS: cur["drowsy"] = True
                if lbl == GAZE_L:        cur["gaze_l"] = True
                if lbl == GAZE_R:        cur["gaze_r"] = True
                if lbl == YAWN:          cur["yawn"]   = True
                if lbl in HEAD_LABELS:   cur["head"]   = True

        self.drowsy_q.append(cur["drowsy"])
        self.gaze_l_q.append(cur["gaze_l"])
        self.gaze_r_q.append(cur["gaze_r"])
        self.yawn_q.append(cur["yawn"])
        self.head_q.append(cur["head"])

        flags = {
            "drowsy": sum(self.drowsy_q) >= self.drowsy_thr,
            "gaze_l": sum(self.gaze_l_q) >= self.gaze_thr,
            "gaze_r": sum(self.gaze_r_q) >= self.gaze_thr,
            "yawn":   sum(self.yawn_q)   >= self.yawn_thr,
            "head":   sum(self.head_q)   >= self.head_thr,
        }

        # timestamps + alarms
        if flags["drowsy"]:
            if self.ts["drowsy"] is None: self.cnt["drowsy"] += 1
            self.ts["drowsy"] = now
            self._fire("drowsy", 3000)
        else:
            self.ts["drowsy"] = None
            self.alarm_until["drowsy"] = None

        if flags["gaze_l"]:
            if self.ts["gaze_l"] is None: self.cnt["gaze_l"] += 1
            self.ts["gaze_l"] = now
            self._fire("gaze_l", 1500)
            self.gaze_l_q.clear()

        if flags["gaze_r"]:
            if self.ts["gaze_r"] is None: self.cnt["gaze_r"] += 1
            self.ts["gaze_r"] = now
            self._fire("gaze_r", 1500)
            self.gaze_r_q.clear()

        if flags["yawn"]:
            if self.ts["yawn"] is None: self.cnt["yawn"] += 1
            self.ts["yawn"] = now
            self._fire("yawn", 1000)
            self.yawn_q.clear()

        if flags["head"]:
            if self.ts["head"] is None: self.cnt["head"] += 1
            self.ts["head"] = now
            self.head_q.clear()

        # ── annotate ──────────────────────────────────────────────────────────
        out = frame.copy()
        for (x0,y0,x1,y1, lbl, conf) in dets:
            name = CLASS_NAMES.get(lbl, str(lbl))
            if lbl in DROWSY_LABELS:
                color = COLOR_DROWSY if flags["drowsy"] else COLOR_OK
            elif lbl in (GAZE_L, GAZE_R):
                color = COLOR_GAZE
            elif lbl == YAWN:
                color = COLOR_YAWN
            elif lbl in HEAD_LABELS:
                color = COLOR_HEAD
            else:
                color = COLOR_OK
            cv2.rectangle(out, (x0,y0), (x1,y1), color, 2)
            cv2.putText(out, f"{name} {conf:.2f}", (x0, max(y0-7,14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

        overlays = []
        if self._active("drowsy"): overlays.append(("WARNING: DROWSY!",        (0, 40, 255)))
        if self._active("gaze_l"): overlays.append(("WARNING: LOOKING LEFT!",  (0,140, 255)))
        if self._active("gaze_r"): overlays.append(("WARNING: LOOKING RIGHT!", (0,140, 255)))
        if self._active("yawn"):   overlays.append(("WARNING: YAWNING!",       (0,210, 255)))
        if self._active("head"):   overlays.append(("WARNING: HEAD MOVEMENT!", (160,60,255)))

        for idx, (txt, col) in enumerate(overlays):
            cv2.putText(out, txt, (20, 45 + idx*42),
                        cv2.FONT_HERSHEY_DUPLEX, 0.82, col, 2)

        status = {
            "drowsy_on":  self._active("drowsy"),
            "gaze_l_on":  self._active("gaze_l"),
            "gaze_r_on":  self._active("gaze_r"),
            "yawn_on":    self._active("yawn"),
            "head_on":    self._active("head"),
            "cnt_drowsy": self.cnt["drowsy"],
            "cnt_gaze_l": self.cnt["gaze_l"],
            "cnt_gaze_r": self.cnt["gaze_r"],
            "cnt_yawn":   self.cnt["yawn"],
            "cnt_head":   self.cnt["head"],
            "frame_count":self.frame_n,
        }
        return out, status


# ── Global app state ──────────────────────────────────────────────────────────
state = {
    "running":    False,
    "latest_jpg": None,
    "lock":       threading.Lock(),
    "status": {k: False for k in
               ("drowsy_on","gaze_l_on","gaze_r_on","yawn_on","head_on")},
}

def capture_loop(source, model):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        state["running"] = False
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    det = DrowsinessDetector(fps=fps)

    while state["running"]:
        ret, frame = cap.read()
        if not ret: break
        ann, status = det.process(frame, model)
        _, jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with state["lock"]:
            state["latest_jpg"] = jpg.tobytes()
            state["status"]     = {**status, "running": True}

    cap.release()
    state["running"] = False


# ── Dashboard HTML ────────────────────────────────────────────────────────────
DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SafeDrive DMS</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#080c10;color:#c8d0dc;font-family:'Rajdhani',sans-serif;}

header{
  background:#0d1117;
  border-bottom:1px solid #1a2030;
  padding:0.9rem 1.8rem;
  display:flex;align-items:center;gap:1rem;
}
header h1{font-size:1.4rem;letter-spacing:0.06em;color:#e0e8f0;}
.mono{font-family:'Share Tech Mono',monospace;}
.sub{font-size:0.72rem;color:#445;letter-spacing:0.12em;}

.badge{
  padding:0.2rem 0.75rem;border-radius:20px;
  font-family:'Share Tech Mono',monospace;font-size:0.72rem;
  font-weight:700;letter-spacing:0.08em;margin-left:auto;
}
.badge-on {background:#00ff8812;border:1px solid #00ff8866;color:#00ff88;}
.badge-off{background:#ff003312;border:1px solid #ff003366;color:#ff4455;}

.layout{display:flex;gap:1.4rem;padding:1.4rem;flex-wrap:wrap;}
.col-video{flex:3;min-width:300px;}
.col-side {flex:1;min-width:220px;display:flex;flex-direction:column;gap:1rem;}

.card{
  background:#0d1117;
  border:1px solid #1a2030;
  border-radius:8px;padding:1rem;
}
.card-title{
  font-family:'Share Tech Mono',monospace;
  font-size:0.7rem;color:#334;
  letter-spacing:0.14em;text-transform:uppercase;
  border-bottom:1px solid #1a2030;
  padding-bottom:0.4rem;margin-bottom:0.75rem;
}

#stream{width:100%;border-radius:6px;display:block;min-height:240px;background:#050810;}

/* Alerts */
.alert{
  padding:0.55rem 0.9rem;border-radius:5px;
  font-family:'Share Tech Mono',monospace;
  font-size:0.8rem;letter-spacing:0.04em;
  margin-bottom:0.4rem;
  display:flex;align-items:center;gap:0.5rem;
  transition:all 0.3s;
}
.a-ok    {background:#00ff8808;border:1px solid #00ff8840;color:#00cc66;}
.a-drowsy{background:#ff003320;border:1px solid #ff003388;color:#ff4466;
          animation:pulse 0.9s infinite;}
.a-gaze  {background:#ff880018;border:1px solid #ff880077;color:#ffaa44;}
.a-yawn  {background:#ffcc0018;border:1px solid #ffcc0077;color:#ffdd44;}
.a-head  {background:#9944ff18;border:1px solid #9944ff77;color:#bb77ff;}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}

/* Stats */
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.45rem;}
.stat-tile{
  background:#080c10;border:1px solid #141c28;
  border-radius:5px;padding:0.55rem 0.7rem;text-align:center;
}
.sn{font-family:'Share Tech Mono',monospace;font-size:1.5rem;font-weight:700;}
.sl{font-size:0.68rem;color:#445;letter-spacing:0.07em;text-transform:uppercase;margin-top:1px;}

/* Controls */
.ctrl{display:flex;flex-wrap:wrap;gap:0.6rem;align-items:center;}
select,input[type=text]{
  flex:1;padding:0.45rem 0.7rem;border-radius:5px;
  background:#141c28;border:1px solid #2a3545;color:#c8d0dc;
  font-family:'Rajdhani',sans-serif;font-size:0.9rem;min-width:140px;
}
button{
  padding:0.45rem 1.1rem;border-radius:5px;
  font-family:'Rajdhani',sans-serif;font-weight:700;
  font-size:0.9rem;cursor:pointer;border:none;letter-spacing:0.05em;
}
.btn-s{background:#00c851;color:#000;}
.btn-s:hover{background:#00a040;}
.btn-x{background:#ff3344;color:#fff;}
.btn-x:hover{background:#cc2233;}
</style>
</head>
<body>
<header>
  <span style="font-size:1.8rem">🚗</span>
  <div>
    <h1 class="mono">SafeDrive DMS</h1>
    <div class="sub">DRIVER MONITORING SYSTEM · YOLOv8</div>
  </div>
  <span id="badge" class="badge badge-off">STOPPED</span>
</header>

<div class="layout">
  <!-- Video -->
  <div class="col-video">
    <div class="card">
      <div class="card-title">Live Feed</div>
      <img id="stream" src="/video_feed" alt="stream">
    </div>
    <!-- Controls -->
    <div class="card" style="margin-top:1rem">
      <div class="card-title">Controls</div>
      <div class="ctrl" style="margin-bottom:0.6rem">
        <select id="src">
          <option value="0">Webcam (local)</option>
          <option value="ip">IP Webcam URL</option>
        </select>
        <input type="text" id="ip" placeholder="http://192.168.x.x:8080/video" style="display:none">
      </div>
      <div class="ctrl">
        <button class="btn-s" onclick="startDetection()">▶ Start</button>
        <button class="btn-x" onclick="stopDetection()">⏹ Stop</button>
      </div>
    </div>
  </div>

  <!-- Side panel -->
  <div class="col-side">
    <div class="card">
      <div class="card-title">Live Alerts</div>
      <div id="alerts"><div class="alert a-ok">✔ DRIVER ALERT — OK</div></div>
    </div>
    <div class="card">
      <div class="card-title">Session Stats</div>
      <div class="stat-grid">
        <div class="stat-tile"><div class="sn" style="color:#4af" id="s-f">0</div><div class="sl">Frames</div></div>
        <div class="stat-tile"><div class="sn" style="color:#f46" id="s-d">0</div><div class="sl">Drowsy</div></div>
        <div class="stat-tile"><div class="sn" style="color:#fa6" id="s-l">0</div><div class="sl">Gaze-L</div></div>
        <div class="stat-tile"><div class="sn" style="color:#fa6" id="s-r">0</div><div class="sl">Gaze-R</div></div>
        <div class="stat-tile"><div class="sn" style="color:#fd4" id="s-y">0</div><div class="sl">Yawn</div></div>
        <div class="stat-tile"><div class="sn" style="color:#a6f" id="s-h">0</div><div class="sl">Head</div></div>
      </div>
    </div>
  </div>
</div>

<script>
const srcSel = document.getElementById('src');
const ipIn   = document.getElementById('ip');
srcSel.addEventListener('change', () => {
  ipIn.style.display = srcSel.value === 'ip' ? 'block' : 'none';
});

async function startDetection() {
  const source = srcSel.value === 'ip' ? ipIn.value.trim() : '0';
  await fetch('/start', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({source})
  });
  document.getElementById('stream').src = '/video_feed?' + Date.now();
}

async function stopDetection() {
  await fetch('/stop', {method:'POST'});
}

// Poll /status every 400 ms
setInterval(async () => {
  try {
    const d = await (await fetch('/status')).json();

    // Badge
    const b = document.getElementById('badge');
    b.textContent = d.running ? 'RUNNING' : 'STOPPED';
    b.className   = 'badge ' + (d.running ? 'badge-on' : 'badge-off');

    // Stats
    document.getElementById('s-f').textContent = d.frame_count  || 0;
    document.getElementById('s-d').textContent = d.cnt_drowsy   || 0;
    document.getElementById('s-l').textContent = d.cnt_gaze_l   || 0;
    document.getElementById('s-r').textContent = d.cnt_gaze_r   || 0;
    document.getElementById('s-y').textContent = d.cnt_yawn     || 0;
    document.getElementById('s-h').textContent = d.cnt_head     || 0;

    // Alerts
    let html = '';
    if (d.drowsy_on) html += '<div class="alert a-drowsy">🔴 DROWSY — EYES CLOSED</div>';
    if (d.gaze_l_on) html += '<div class="alert a-gaze">◀ LOOKING LEFT</div>';
    if (d.gaze_r_on) html += '<div class="alert a-gaze">▶ LOOKING RIGHT</div>';
    if (d.yawn_on)   html += '<div class="alert a-yawn">😮 YAWNING DETECTED</div>';
    if (d.head_on)   html += '<div class="alert a-head">↕ HEAD MOVEMENT</div>';
    if (!html)       html  = '<div class="alert a-ok">✔ DRIVER ALERT — OK</div>';
    document.getElementById('alerts').innerHTML = html;
  } catch(e) {}
}, 400);
</script>
</body>
</html>
"""

# ── Flask routes ──────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD)

@app.route("/health")
def health():
    return jsonify({"ok": True, "yolo": YOLO_OK})

@app.route("/start", methods=["POST"])
def start():
    if state["running"]:
        return jsonify({"msg": "Already running"}), 400
    if not YOLO_OK:
        return jsonify({"error": "ultralytics not installed"}), 500

    data   = request.get_json(silent=True) or {}
    source = data.get("source", "0")
    try: source = int(source)
    except (ValueError, TypeError): pass

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    state["running"] = True
    threading.Thread(target=capture_loop, args=(source, model), daemon=True).start()
    return jsonify({"msg": "Started", "source": str(source)})

@app.route("/stop", methods=["POST"])
def stop():
    state["running"] = False
    return jsonify({"msg": "Stopped"})

@app.route("/status")
def status():
    with state["lock"]:
        s = dict(state["status"])
    s["running"] = state["running"]
    return jsonify(s)

def gen_frames():
    blank = None
    while True:
        with state["lock"]:
            jpg = state["latest_jpg"]
        if jpg is None:
            if blank is None:
                b = np.zeros((360, 640, 3), np.uint8)
                cv2.putText(b, "Waiting for stream...", (130, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40,50,60), 2)
                _, enc = cv2.imencode(".jpg", b)
                blank = enc.tobytes()
            jpg = blank
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(1 / FPS)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
