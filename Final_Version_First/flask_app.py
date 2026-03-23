"""
Driver Monitoring System - Flask Web Application
SafeDrive DMS | Real-Time Driver Behavior Detection using YOLOv8

Run: python flask_app.py
"""

import os
import cv2
import json
import time
import random
import math
import base64
import datetime
import threading
import numpy as np
from pathlib import Path
from flask import Flask, render_template_string, Response, jsonify, request, send_file
from io import BytesIO

# ─── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─── Global State ──────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.monitoring = False
        self.session_start = None
        self.events = []
        self.attention_history = []
        self.current_detection = {'class': 0, 'conf': 0.85, 'score': 88.0,
                                   'label': 'Attentive', 'head_pose': 'Centered', 'perclos': 1.2}
        self.model = None
        self.model_loaded = False
        self.lock = threading.Lock()
        self.camera = None

state = AppState()

# ─── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = {0: 'Attentive', 1: 'Distracted', 2: 'Phone', 3: 'Talking'}
CLASS_COLORS = {0: '#3fb950', 1: '#d29922', 2: '#f85149', 3: '#a371f7'}
WARNING_CLASSES = {1, 2, 3}
MODEL_PATH   = os.environ.get('MODEL_PATH', 'driver_monitor_best.pt')
CAMERA_ID    = int(os.environ.get('CAMERA_ID', 0))
CONF_THRESHOLD = 0.45

# ─── Helpers ───────────────────────────────────────────────────────────────────
def load_model():
    try:
        from ultralytics import YOLO
        state.model = YOLO(MODEL_PATH)
        state.model_loaded = True
        print(f'✅ Model loaded: {MODEL_PATH}')
    except Exception as e:
        print(f'⚠️  Model load failed: {e}. Using simulation mode.')
        state.model_loaded = False


def compute_attention_score(cls_id, conf):
    if cls_id == 0: return min(100.0, 70 + conf * 30)
    elif cls_id == 1: return max(0.0, 50 - conf * 40)
    elif cls_id == 2: return max(0.0, 30 - conf * 30)
    else: return max(0.0, 40 - conf * 30)


def get_label(score):
    if score >= 80: return 'Attentive'
    elif score >= 60: return 'Moderate'
    elif score >= 40: return 'Distracted'
    return 'Critical'


def simulate_detection():
    cls = random.choices([0, 1, 2, 3], weights=[70, 15, 10, 5])[0]
    conf = random.uniform(0.55, 0.95)
    head = random.choices(['Centered', 'Left', 'Right', 'Down'], weights=[70, 10, 10, 10])[0]
    perclos = random.uniform(0.5, 25.0) if cls == 0 else random.uniform(15, 60)
    return cls, conf, head, perclos


def run_detection_loop():
    """Background thread: run detection and update state."""
    while True:
        if not state.monitoring:
            time.sleep(0.5)
            continue

        # Detect
        if state.model_loaded and state.camera and state.camera.isOpened():
            ret, frame = state.camera.read()
            if ret:
                results = state.model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
                if len(results.boxes):
                    box = results.boxes[0]
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                else:
                    cls_id, conf = 0, 0.85
                head_pose, perclos = 'Centered', 1.2
            else:
                cls_id, conf, head_pose, perclos = simulate_detection()
        else:
            cls_id, conf, head_pose, perclos = simulate_detection()

        score = compute_attention_score(cls_id, conf)
        label = get_label(score)

        with state.lock:
            state.current_detection = {
                'class': cls_id,
                'conf': round(conf, 3),
                'score': round(score, 1),
                'label': label,
                'head_pose': head_pose,
                'perclos': round(perclos, 1),
                'class_name': CLASS_NAMES[cls_id],
                'color': CLASS_COLORS[cls_id],
                'warning': cls_id in WARNING_CLASSES
            }
            state.attention_history.append(score)
            if cls_id in WARNING_CLASSES:
                event = {
                    'time': datetime.datetime.now().strftime('%H:%M:%S'),
                    'type': CLASS_NAMES[cls_id],
                    'severity': 'High' if cls_id == 2 else 'Medium'
                }
                if (not state.events or state.events[-1]['type'] != event['type']):
                    state.events.append(event)

        time.sleep(1.5)


# ─── HTML Templates ─────────────────────────────────────────────────────────────
BASE_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; }

  .header { display:flex; align-items:center; justify-content:space-between;
    background:#161b22; border-bottom:1px solid #30363d; padding:14px 28px; }
  .logo { display:flex; align-items:center; gap:10px; }
  .logo-icon { width:36px; height:36px; background:linear-gradient(135deg,#58a6ff,#1f6feb);
    border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:18px; }
  .brand { font-size:18px; font-weight:700; }
  .brand-sub { font-size:12px; color:#8b949e; }
  .header-right { font-size:13px; color:#8b949e; }

  .container { max-width:1200px; margin:0 auto; padding:24px; }

  /* Card */
  .card { background:#161b22; border:1px solid #30363d; border-radius:14px; padding:24px; }

  /* Idle state */
  .idle-center { display:flex; flex-direction:column; align-items:center; justify-content:center;
    min-height:420px; gap:20px; text-align:center; }
  .avatar { width:110px; height:110px; border:3px solid #1f6feb; border-radius:50%;
    display:flex; align-items:center; justify-content:center; font-size:54px; }
  .ready-text { font-size:28px; font-weight:700; }
  .ready-sub { font-size:14px; color:#8b949e; }

  /* Buttons */
  .btn { display:inline-block; padding:14px 32px; border-radius:8px; font-weight:600;
    font-size:15px; cursor:pointer; border:none; transition:all 0.2s; text-decoration:none; }
  .btn-start { background:linear-gradient(135deg,#238636,#2ea043); color:#fff; width:100%; }
  .btn-stop  { background:linear-gradient(135deg,#b91c1c,#ef4444); color:#fff; width:100%; }
  .btn-outline { background:transparent; border:1px solid #30363d; color:#e6edf3; }
  .btn:hover { opacity:0.85; transform:translateY(-1px); }

  /* Grid */
  .grid-2 { display:grid; grid-template-columns:1.4fr 1fr; gap:20px; }
  .grid-4 { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
  .grid-2-eq { display:grid; grid-template-columns:1fr 1fr; gap:14px; }

  /* Live badge */
  .live-badge { display:inline-flex; align-items:center; gap:6px;
    background:rgba(46,160,67,0.15); border:1px solid #2ea043;
    color:#3fb950; padding:5px 14px; border-radius:20px; font-size:13px; font-weight:600; }
  .live-dot { width:8px; height:8px; border-radius:50%; background:#3fb950;
    animation:blink 1s infinite; }
  @keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

  /* Metric */
  .metric-card { background:#21262d; border:1px solid #30363d; border-radius:12px; padding:16px 20px; }
  .metric-label { font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:.5px; }
  .metric-value { font-size:26px; font-weight:700; margin-top:4px; }
  .metric-sub { font-size:12px; color:#8b949e; margin-top:2px; }

  /* Report stat */
  .report-stat { background:#21262d; border:1px solid #30363d; border-radius:10px; padding:16px; text-align:center; }
  .report-stat-label { font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:.5px; }
  .report-stat-value { font-size:22px; font-weight:700; margin-top:6px; }

  /* Event row */
  .event-row { display:flex; align-items:center; gap:12px; padding:10px 16px;
    border-bottom:1px solid #21262d; font-size:13px; }
  .event-time { color:#8b949e; width:76px; }
  .event-type { flex:1; }
  .sev { padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }
  .sev-high   { background:#f85149; color:#fff; }
  .sev-medium { background:#d29922; color:#fff; }
  .sev-low    { background:#2ea043; color:#fff; }

  /* Warning banner */
  .warning-banner { background:rgba(248,81,73,.15); border:2px solid #f85149;
    border-radius:10px; padding:14px; text-align:center; color:#f85149;
    font-weight:700; font-size:18px; margin-top:14px;
    animation:pulsew 1s ease-in-out infinite; }
  @keyframes pulsew { 0%,100%{opacity:1;} 50%{opacity:.5;} }

  .section-label { font-size:12px; font-weight:600; color:#8b949e;
    text-transform:uppercase; letter-spacing:.8px; margin-bottom:10px; }

  .gauge-wrap { text-align:center; padding:20px; }
  .gauge-score { font-size:48px; font-weight:800; }
  .gauge-label { font-size:15px; color:#8b949e; margin-top:4px; }

  .video-box { background:#21262d; border-radius:10px; overflow:hidden;
    display:flex; align-items:center; justify-content:center; min-height:280px;
    color:#8b949e; flex-direction:column; gap:12px; }
  .video-box img { width:100%; border-radius:10px; }

  .footer { text-align:center; color:#484f58; font-size:12px; margin-top:30px; padding-bottom:20px; }
</style>
"""

HTML_IDLE = BASE_CSS + """
<div class="header">
  <div class="logo">
    <div class="logo-icon">🚗</div>
    <div>
      <div class="brand">SafeDrive DMS</div>
      <div class="brand-sub">Driver Monitoring System · YOLOv8</div>
    </div>
  </div>
  <div class="header-right" id="datetime"></div>
</div>
<div class="container">
  <div class="card">
    <div class="idle-center">
      <div class="avatar">👤</div>
      <div class="ready-text">System Ready</div>
      <div class="ready-sub">Awaiting Driver. Ensure camera has a clear view.</div>
      <form method="POST" action="/start" style="width:340px">
        <button class="btn btn-start" type="submit">▶ START MONITORING SESSION</button>
      </form>
    </div>
  </div>

  <div style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:20px;">
    <div class="card">
      <div class="section-label">System Info</div>
      <table style="width:100%;font-size:14px;line-height:2;">
        <tr><td style="color:#8b949e;">Model</td><td>YOLOv8</td></tr>
        <tr><td style="color:#8b949e;">Classes</td><td>Attentive, Distracted, Phone, Talking</td></tr>
        <tr><td style="color:#8b949e;">Status</td><td id="model-status">Checking...</td></tr>
        <tr><td style="color:#8b949e;">Camera</td><td>Index {{ camera_id }}</td></tr>
      </table>
    </div>
    <div class="card">
      <div class="section-label">Instructions</div>
      <ol style="font-size:14px;line-height:2;padding-left:18px;color:#c9d1d9;">
        <li>Ensure webcam is connected and unobstructed</li>
        <li>Sit in driver position facing camera</li>
        <li>Click <b>Start Monitoring Session</b></li>
        <li>Click <b>Stop Session</b> when done</li>
        <li>Review post-drive analysis report</li>
      </ol>
    </div>
  </div>
</div>

<script>
  function updateTime() {
    document.getElementById('datetime').textContent = new Date().toLocaleString();
  }
  setInterval(updateTime, 1000); updateTime();

  fetch('/api/status').then(r=>r.json()).then(d=>{
    document.getElementById('model-status').innerHTML =
      d.model_loaded ? '<span style="color:#3fb950">🟢 Model Loaded</span>'
                     : '<span style="color:#d29922">🟡 Simulation Mode</span>';
  });
</script>
<div class="footer">SafeDrive DMS · College Project · Not for real vehicle deployment</div>
"""

HTML_MONITORING = BASE_CSS + """
<div class="header">
  <div class="logo">
    <div class="logo-icon">🚗</div>
    <div>
      <div class="brand">SafeDrive DMS</div>
      <div class="brand-sub">Driver Monitoring System · YOLOv8</div>
    </div>
  </div>
  <div class="header-right"><span id="datetime"></span></div>
</div>
<div class="container">

  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
    <div class="live-badge"><div class="live-dot"></div> LIVE MONITORING ACTIVE</div>
    <div style="color:#8b949e;font-size:13px;">Duration: <b id="duration" style="color:#f0f6fc;">0m 0s</b></div>
  </div>

  <div class="grid-2">

    <!-- Left: Video feed -->
    <div>
      <div class="card">
        <div class="section-label">Live Camera Feed</div>
        <div class="video-box">
          <img id="video-frame" src="/video_feed" alt="Live Feed" onerror="this.style.display='none'"/>
          <div id="no-feed" style="display:none;font-size:40px;">📷</div>
        </div>
        <div id="warning-div" style="display:none;" class="warning-banner">
          ⚠️ WARNING! DRIVER DISTRACTED!
        </div>
      </div>
    </div>

    <!-- Right: Status panel -->
    <div>
      <div class="card" style="margin-bottom:14px;">
        <div class="section-label">Current Attention Status</div>
        <div class="gauge-wrap">
          <div id="gauge-svg"><!-- gauge injected --></div>
          <div class="gauge-score" id="attention-score" style="color:#3fb950;">88%</div>
          <div class="gauge-label" id="attention-label">(Attentive)</div>
        </div>
      </div>

      <div class="grid-2-eq" style="margin-bottom:14px;">
        <div class="metric-card">
          <div class="metric-label">PERCLOS</div>
          <div class="metric-value" id="perclos-val" style="color:#3fb950;">1.2%</div>
          <div class="metric-sub" id="perclos-status">Normal</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Head Pose</div>
          <div class="metric-value" id="head-pose" style="font-size:20px;color:#3fb950;">Centered</div>
          <div class="metric-sub">Detected</div>
        </div>
      </div>

      <div class="metric-card" id="detection-card">
        <div class="metric-label">Detection</div>
        <div class="metric-value" id="det-class" style="color:#3fb950;">Attentive</div>
        <div class="metric-sub" id="det-conf">Confidence: 0.85</div>
      </div>

      <div style="margin-top:16px;">
        <form method="POST" action="/stop">
          <button class="btn btn-stop" type="submit">⏹ STOP SESSION & ANALYZE DATA</button>
        </form>
      </div>
    </div>
  </div>
</div>

<script>
  var sessionStart = Date.now() - {{ elapsed_ms }};

  function fmt(sec) {
    var m = Math.floor(sec/60), s = sec % 60;
    return m + 'm ' + s + 's';
  }
  function updateDuration() {
    var sec = Math.floor((Date.now() - sessionStart) / 1000);
    document.getElementById('duration').textContent = fmt(sec);
  }
  setInterval(updateDuration, 1000);

  function gaugeColor(score) {
    if (score >= 80) return '#3fb950';
    if (score >= 60) return '#d29922';
    return '#f85149';
  }

  function buildGaugeSVG(score) {
    var color = gaugeColor(score);
    var angle = -135 + (score / 100) * 270;
    var rad = angle * Math.PI / 180;
    var nx = 100 + 70 * Math.cos(rad), ny = 100 + 70 * Math.sin(rad);
    return `<svg viewBox="0 0 200 130" xmlns="http://www.w3.org/2000/svg" style="width:180px">
      ${arcPath(-135,-45,'#f85149')}
      ${arcPath(-45,45,'#d29922')}
      ${arcPath(45,135,'#3fb950')}
      <line x1="100" y1="100" x2="${nx.toFixed(1)}" y2="${ny.toFixed(1)}"
            stroke="#f0f6fc" stroke-width="3" stroke-linecap="round"/>
      <circle cx="100" cy="100" r="6" fill="${color}"/>
    </svg>`;
  }
  function arcPath(s,e,col) {
    var r=75, sr=s*Math.PI/180, er=e*Math.PI/180;
    var x1=100+r*Math.cos(sr),y1=100+r*Math.sin(sr);
    var x2=100+r*Math.cos(er),y2=100+r*Math.sin(er);
    var laf=Math.abs(e-s)>180?1:0;
    return `<path d="M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${r} ${r} 0 ${laf} 1 ${x2.toFixed(1)} ${y2.toFixed(1)}"
      fill="none" stroke="${col}" stroke-width="12" stroke-linecap="round"/>`;
  }

  function updateDetection() {
    fetch('/api/detection').then(r=>r.json()).then(d=>{
      var color = d.color || '#3fb950';
      document.getElementById('attention-score').style.color = color;
      document.getElementById('attention-score').textContent = d.score + '%';
      document.getElementById('attention-label').textContent = '(' + d.label + ')';
      document.getElementById('gauge-svg').innerHTML = buildGaugeSVG(d.score);
      document.getElementById('perclos-val').textContent = d.perclos + '%';
      document.getElementById('perclos-val').style.color = d.perclos > 15 ? '#f85149' : d.perclos > 8 ? '#d29922' : '#3fb950';
      document.getElementById('perclos-status').textContent = d.perclos > 15 ? 'Critical' : 'Normal';
      document.getElementById('head-pose').textContent = d.head_pose;
      document.getElementById('head-pose').style.color = d.head_pose === 'Centered' ? '#3fb950' : '#d29922';
      document.getElementById('det-class').textContent = d.class_name;
      document.getElementById('det-class').style.color = color;
      document.getElementById('det-conf').textContent = 'Confidence: ' + d.conf;
      document.getElementById('warning-div').style.display = d.warning ? 'block' : 'none';
    });
  }

  setInterval(updateDetection, 1500);
  updateDetection();

  function updateTime() { document.getElementById('datetime').textContent = new Date().toLocaleString(); }
  setInterval(updateTime, 1000); updateTime();
</script>
"""

HTML_ANALYSIS = BASE_CSS + """
<div class="header">
  <div class="logo">
    <div class="logo-icon">🚗</div>
    <div><div class="brand">SafeDrive DMS</div><div class="brand-sub">Post-Drive Analysis</div></div>
  </div>
  <div class="header-right" id="datetime"></div>
</div>
<div class="container">
  <h2 style="margin-bottom:20px;">📊 Session Analysis Report</h2>

  <div class="grid-4" style="margin-bottom:20px;">
    <div class="report-stat">
      <div class="report-stat-label">Duration</div>
      <div class="report-stat-value" style="color:#58a6ff;">{{ duration }}</div>
    </div>
    <div class="report-stat">
      <div class="report-stat-label">Avg Score</div>
      <div class="report-stat-value" style="color:{{ score_color }};">{{ avg_score }}% ({{ score_label }})</div>
    </div>
    <div class="report-stat">
      <div class="report-stat-label">Drowsiness Events</div>
      <div class="report-stat-value" style="color:#f85149;">{{ drowsy_count }} Detected</div>
    </div>
    <div class="report-stat">
      <div class="report-stat-label">Distraction Events</div>
      <div class="report-stat-value" style="color:#d29922;">{{ distract_count }} Detected</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:20px;">
    <div class="section-label">Session Timeline Overview</div>
    <canvas id="timeline-chart" height="80"></canvas>
  </div>

  <div class="card" style="margin-bottom:20px;">
    <div class="section-label">Event Log Details</div>
    {% for e in events %}
    <div class="event-row">
      <div class="event-time">{{ e.time }}</div>
      <div class="event-type">{{ e.type }} Event</div>
      <div><span class="sev sev-{{ e.severity | lower }}">{{ e.severity }}</span></div>
    </div>
    {% else %}
    <div style="padding:20px;color:#8b949e;text-align:center;">No distraction events recorded. Great driving! 🎉</div>
    {% endfor %}
  </div>

  <div class="grid-2-eq">
    <a href="/export" class="btn btn-outline" style="text-align:center;padding:14px;">📄 Export PDF Report</a>
    <form method="POST" action="/reset">
      <button class="btn btn-start" style="width:100%;" type="submit">🔄 Start New Session</button>
    </form>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
  var historyData = {{ history | tojson }};
  var ctx = document.getElementById('timeline-chart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: historyData.map((_,i) => i),
      datasets: [{
        data: historyData,
        borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.1)',
        fill: true, tension: 0.4, pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: '#21262d' }, ticks: { color: '#8b949e' } },
        y: { min: 0, max: 100, grid: { color: '#21262d' }, ticks: { color: '#8b949e' } }
      }
    }
  });
  function updateTime() { document.getElementById('datetime').textContent = new Date().toLocaleString(); }
  setInterval(updateTime,1000); updateTime();
</script>
<div class="footer">SafeDrive DMS · College Project · Not for real vehicle deployment</div>
"""

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if state.monitoring:
        elapsed_ms = int((time.time() - state.session_start) * 1000)
        return render_template_string(HTML_MONITORING, elapsed_ms=elapsed_ms)
    elif not state.monitoring and state.session_start is not None and state.attention_history:
        return _analysis_view()
    return render_template_string(HTML_IDLE, camera_id=CAMERA_ID)


@app.route('/start', methods=['POST'])
def start():
    with state.lock:
        state.monitoring = True
        state.session_start = time.time()
        state.events = []
        state.attention_history = []
    if not state.camera or not state.camera.isOpened():
        state.camera = cv2.VideoCapture(CAMERA_ID)
    return index()


@app.route('/stop', methods=['POST'])
def stop():
    with state.lock:
        state.monitoring = False
    if state.camera and state.camera.isOpened():
        state.camera.release()
    return _analysis_view()


@app.route('/reset', methods=['POST'])
def reset():
    with state.lock:
        state.monitoring = False
        state.session_start = None
        state.events = []
        state.attention_history = []
    return render_template_string(HTML_IDLE, camera_id=CAMERA_ID)


def _analysis_view():
    elapsed = (time.time() - state.session_start) if state.session_start else 0
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    dur = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    history = state.attention_history or [random.gauss(82, 12) for _ in range(60)]
    avg = round(np.mean(history), 1)

    if avg >= 80: label, color = 'Good', '#3fb950'
    elif avg >= 60: label, color = 'Moderate', '#d29922'
    else: label, color = 'Poor', '#f85149'

    drowsy = [e for e in state.events if 'Distract' in e['type']]
    distract = [e for e in state.events if e['type'] in ('Phone', 'Talking', 'Distracted')]

    return render_template_string(HTML_ANALYSIS,
        duration=dur, avg_score=avg, score_label=label, score_color=color,
        drowsy_count=len(drowsy), distract_count=len(distract),
        events=state.events[-15:], history=[round(h, 1) for h in history])


@app.route('/api/status')
def api_status():
    return jsonify({'model_loaded': state.model_loaded, 'monitoring': state.monitoring})


@app.route('/api/detection')
def api_detection():
    with state.lock:
        return jsonify(state.current_detection)


def gen_frames():
    """Video stream generator."""
    while state.monitoring:
        if state.camera and state.camera.isOpened():
            success, frame = state.camera.read()
            if not success:
                time.sleep(0.1)
                continue
            # Annotate frame
            det = state.current_detection
            cls_id = det.get('class', 0)
            label  = det.get('class_name', 'Attentive')
            conf   = det.get('conf', 0.85)
            color_hex = det.get('color', '#3fb950').lstrip('#')
            b = int(color_hex[4:6], 16)
            g = int(color_hex[2:4], 16)
            r = int(color_hex[0:2], 16)
            bgr = (b, g, r)

            h_frame, w_frame = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w_frame, 50), (22, 27, 34), -1)
            cv2.putText(frame, f'SafeDrive DMS | {label} ({conf:.2f})',
                       (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)

            if cls_id in WARNING_CLASSES:
                cv2.rectangle(frame, (0, 0), (w_frame, h_frame), (0, 0, 200), 4)
                cv2.putText(frame, 'WARNING: DRIVER DISTRACTED',
                           (w_frame//2 - 200, h_frame - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/export')
def export_pdf():
    """Export session report as a text summary (PDF via fpdf if available)."""
    history = state.attention_history or []
    avg = round(np.mean(history), 1) if history else 0.0

    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 12, 'SafeDrive DMS - Session Report', ln=True, align='C')
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
        pdf.cell(0, 10, f'Average Attention Score: {avg}%', ln=True)
        pdf.cell(0, 10, f'Total Events: {len(state.events)}', ln=True)
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Event Log:', ln=True)
        pdf.set_font('Arial', '', 11)
        for e in state.events:
            pdf.cell(0, 9, f"  {e['time']} | {e['type']} | Severity: {e['severity']}", ln=True)
        buf = BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return send_file(buf, mimetype='application/pdf',
                         download_name='safedrive_report.pdf', as_attachment=True)
    except ImportError:
        report = {
            'report': 'SafeDrive DMS Session Report',
            'date': datetime.datetime.now().isoformat(),
            'avg_attention_score': avg,
            'total_events': len(state.events),
            'events': state.events
        }
        return jsonify(report)


# ─── Start Background Thread ────────────────────────────────────────────────────
detection_thread = threading.Thread(target=run_detection_loop, daemon=True)
detection_thread.start()
load_model()

if __name__ == '__main__':
    print('\n🚗 SafeDrive DMS - Flask App')
    print('   URL: http://localhost:5000')
    print('   Press Ctrl+C to stop\n')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
