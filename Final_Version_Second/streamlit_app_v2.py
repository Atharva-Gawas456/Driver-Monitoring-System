"""
SafeDrive DMS — Streamlit App (v2)
────────────────────────────────────────────────────────────────────────────────
Alarms triggered for:
  • Drowsy   — eyes_closed / eyes_closed_head_left / eyes_closed_head_right
               sustained for 0.8 × 2-sec window (≥ 24 frames @ 30fps)
  • Gaze-L   — seeing_left  sustained for 0.8 × 2-sec window
  • Gaze-R   — seeing_right sustained for 0.8 × 2-sec window
  • Yawn     — yarning (yawning) sustained for 1.0 × 2-sec window

Alarm sound: alarm.wav played via pygame.mixer in a background thread.
Place alarm.wav in the same directory as this script.

REAL class labels extracted from best.pt weights:
  0: eyes_closed
  1: eyes_closed_head_left
  2: eyes_closed_head_right
  3: focused
  4: head_down
  5: head_up
  6: seeing_left
  7: seeing_right
  8: yarning  (yawning — labelled as 'yarning' in this dataset)
"""

import streamlit as st
import cv2
import numpy as np
import datetime
import time
import threading
from collections import deque
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeDrive DMS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}
.main { background: #080c10; }

/* Alert cards */
.alert-card {
    padding: 0.7rem 1.1rem;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.88rem;
    margin-bottom: 0.45rem;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    transition: all 0.25s ease;
}
.card-ok      { background:#00ff8808; border:1px solid #00ff8855; color:#00ff88; }
.card-drowsy  { background:#ff003322; border:1px solid #ff0033aa; color:#ff4466;
                box-shadow: 0 0 12px #ff003344; animation: pulse 1s infinite; }
.card-gaze    { background:#ff880022; border:1px solid #ff8800aa; color:#ffaa33;
                box-shadow: 0 0 10px #ff880033; }
.card-yawn    { background:#ffcc0022; border:1px solid #ffcc00aa; color:#ffdd44;
                box-shadow: 0 0 10px #ffcc0033; }
.card-head    { background:#aa44ff22; border:1px solid #aa44ffaa; color:#cc77ff; }

@keyframes pulse {
    0%,100% { opacity:1; }
    50%      { opacity:0.6; }
}

/* Stat tiles */
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.stat-tile {
    background: #0d1117;
    border: 1px solid #1e2530;
    border-radius: 6px;
    padding: 0.65rem 0.8rem;
    text-align: center;
}
.stat-num   { font-family:'Share Tech Mono',monospace; font-size:1.6rem; font-weight:700; }
.stat-label { font-size:0.72rem; color:#556; letter-spacing:0.06em; text-transform:uppercase; margin-top:2px; }

.section-title {
    font-family:'Share Tech Mono',monospace;
    font-size:0.72rem;
    color:#445;
    letter-spacing:0.14em;
    text-transform:uppercase;
    margin-bottom:0.6rem;
    border-bottom:1px solid #1a2030;
    padding-bottom:0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FPS              = 30
QUEUE_SEC        = 2          # rolling window in seconds
CONF_THRESH      = 0.5
WARNING_SEC      = 2          # how long warning stays visible after trigger

# Thresholds: fraction of the rolling window that must be positive
DROWSY_FRAC = 0.80
GAZE_FRAC   = 0.80
YAWN_FRAC   = 1.00

# ── Real class map from best.pt ───────────────────────────────────────────────
CLASS_NAMES = {
    0: "eyes_closed",
    1: "eyes_closed_head_left",
    2: "eyes_closed_head_right",
    3: "focused",
    4: "head_down",
    5: "head_up",
    6: "seeing_left",
    7: "seeing_right",
    8: "yarning",           # yawning (labelled 'yarning' in dataset)
}

# Label groups
DROWSY_LABELS = {0, 1, 2}   # any closed-eye state → drowsy cue
GAZE_L_LABEL  = 6
GAZE_R_LABEL  = 7
YAWN_LABEL    = 8
HEAD_LABELS   = {4, 5}

# BB colors per category (BGR for OpenCV)
COLOR_OK     = (0,  220,  80)
COLOR_DROWSY = (0,   40, 255)
COLOR_GAZE   = (0,  160, 255)
COLOR_YAWN   = (0,  210, 255)
COLOR_HEAD   = (180, 80, 255)

# ── Alarm player ──────────────────────────────────────────────────────────────
_alarm_lock   = threading.Lock()
_alarm_active = False

def _play_alarm_thread(wav_path: str, duration_ms: int):
    global _alarm_active
    try:
        import pygame
        pygame.mixer.init()
        sound = pygame.mixer.Sound(wav_path)
        sound.play(maxtime=duration_ms)
        time.sleep(duration_ms / 1000.0)
    except Exception as e:
        pass   # silently skip if pygame/wav not available
    finally:
        with _alarm_lock:
            _alarm_active = False

def trigger_alarm(wav_path: str, duration_ms: int):
    global _alarm_active
    with _alarm_lock:
        if _alarm_active:
            return
        _alarm_active = True
    t = threading.Thread(target=_play_alarm_thread,
                         args=(wav_path, duration_ms), daemon=True)
    t.start()


# ── Core detector ─────────────────────────────────────────────────────────────
class DrowsinessDetector:
    def __init__(self, fps: float = 30.0, wav_path: str = "alarm.wav"):
        self.fps      = fps
        self.wav_path = wav_path
        q = int(fps * QUEUE_SEC)

        self.drowsy_q = deque(maxlen=q)
        self.gaze_l_q = deque(maxlen=q)
        self.gaze_r_q = deque(maxlen=q)
        self.yawn_q   = deque(maxlen=q)
        self.head_q   = deque(maxlen=q)

        # thresholds (absolute frame counts)
        self.drowsy_thr = int(fps * QUEUE_SEC * DROWSY_FRAC)
        self.gaze_thr   = int(fps * QUEUE_SEC * GAZE_FRAC)
        self.yawn_thr   = int(fps * QUEUE_SEC * YAWN_FRAC)
        self.head_thr   = int(fps * QUEUE_SEC * DROWSY_FRAC)

        # timestamps of last event (for on-screen warning duration)
        self.ts_drowsy = self.ts_gaze_l = self.ts_gaze_r = None
        self.ts_yawn   = self.ts_head   = None

        # alarm cooldown timestamps
        self.alarm_drowsy_until = None
        self.alarm_gaze_l_until = None
        self.alarm_gaze_r_until = None
        self.alarm_yawn_until   = None

        # session counters
        self.cnt_drowsy = self.cnt_gaze_l = self.cnt_gaze_r = 0
        self.cnt_yawn   = self.cnt_head   = self.frame_n   = 0

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _active(ts, sec=WARNING_SEC):
        return ts is not None and (datetime.datetime.now() - ts).total_seconds() < sec

    def _fire(self, cooldown_attr: str, duration_ms: int):
        """Fire alarm only if cooldown has expired."""
        now = datetime.datetime.now()
        until = getattr(self, cooldown_attr)
        if until is None or now >= until:
            trigger_alarm(self.wav_path, duration_ms)
            setattr(self, cooldown_attr,
                    now + datetime.timedelta(milliseconds=duration_ms))

    # ── main method ───────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray, model) -> tuple:
        self.frame_n += 1
        now = datetime.datetime.now()

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=[rgb], save=False, verbose=False)[0]

        # ── parse detections ──────────────────────────────────────────────────
        cur = {k: False for k in ("drowsy","gaze_l","gaze_r","yawn","head")}
        dets = []

        if len(results.boxes):
            xyxy    = results.boxes.xyxy.cpu().numpy()
            confs   = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                if confs[i] < CONF_THRESH:
                    continue
                label = int(classes[i])
                xmin, ymin, xmax, ymax = map(int, xyxy[i])
                dets.append((xmin, ymin, xmax, ymax, label, float(confs[i])))

                if label in DROWSY_LABELS: cur["drowsy"] = True
                if label == GAZE_L_LABEL:  cur["gaze_l"] = True
                if label == GAZE_R_LABEL:  cur["gaze_r"] = True
                if label == YAWN_LABEL:    cur["yawn"]   = True
                if label in HEAD_LABELS:   cur["head"]   = True

        # ── update queues ─────────────────────────────────────────────────────
        self.drowsy_q.append(cur["drowsy"])
        self.gaze_l_q.append(cur["gaze_l"])
        self.gaze_r_q.append(cur["gaze_r"])
        self.yawn_q.append(cur["yawn"])
        self.head_q.append(cur["head"])

        # ── evaluate thresholds ───────────────────────────────────────────────
        is_drowsy = sum(self.drowsy_q) >= self.drowsy_thr
        is_gaze_l = sum(self.gaze_l_q) >= self.gaze_thr
        is_gaze_r = sum(self.gaze_r_q) >= self.gaze_thr
        is_yawn   = sum(self.yawn_q)   >= self.yawn_thr
        is_head   = sum(self.head_q)   >= self.head_thr

        # ── update timestamps & fire alarms ───────────────────────────────────
        if is_drowsy:
            if self.ts_drowsy is None:
                self.cnt_drowsy += 1
            self.ts_drowsy = now
            self._fire("alarm_drowsy_until", 3000)   # 3 s repeating

        if is_gaze_l:
            if self.ts_gaze_l is None:
                self.cnt_gaze_l += 1
            self.ts_gaze_l = now
            self._fire("alarm_gaze_l_until", 1500)   # 1.5 s single shot
            self.gaze_l_q.clear()

        if is_gaze_r:
            if self.ts_gaze_r is None:
                self.cnt_gaze_r += 1
            self.ts_gaze_r = now
            self._fire("alarm_gaze_r_until", 1500)
            self.gaze_r_q.clear()

        if is_yawn:
            if self.ts_yawn is None:
                self.cnt_yawn += 1
            self.ts_yawn = now
            self._fire("alarm_yawn_until", 1000)     # 1 s single shot
            self.yawn_q.clear()

        if is_head:
            if self.ts_head is None:
                self.cnt_head += 1
            self.ts_head = now
            self.head_q.clear()

        # reset drowsy alarm cooldown when eyes open
        if not is_drowsy:
            self.alarm_drowsy_until = None
            self.ts_drowsy = None

        # ── annotate frame ────────────────────────────────────────────────────
        out = frame.copy()
        for (xmin, ymin, xmax, ymax, label, conf) in dets:
            name = CLASS_NAMES.get(label, str(label))
            # color by severity
            if label in DROWSY_LABELS:
                color = COLOR_DROWSY if is_drowsy else COLOR_OK
            elif label in (GAZE_L_LABEL, GAZE_R_LABEL):
                color = COLOR_GAZE
            elif label == YAWN_LABEL:
                color = COLOR_YAWN
            elif label in HEAD_LABELS:
                color = COLOR_HEAD
            else:
                color = COLOR_OK

            cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(out, f"{name} {conf:.2f}",
                        (xmin, max(ymin - 7, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

        # ── on-frame overlay warnings ─────────────────────────────────────────
        overlay_items = []
        if self._active(self.ts_drowsy): overlay_items.append(("DROWSY!", (0, 40, 255)))
        if self._active(self.ts_gaze_l): overlay_items.append(("LOOKING LEFT!", (0, 140, 255)))
        if self._active(self.ts_gaze_r): overlay_items.append(("LOOKING RIGHT!", (0, 140, 255)))
        if self._active(self.ts_yawn):   overlay_items.append(("YAWNING!", (0, 210, 255)))
        if self._active(self.ts_head):   overlay_items.append(("HEAD MOVEMENT!", (160, 60, 255)))

        for idx, (text, color) in enumerate(overlay_items):
            y = 45 + idx * 42
            cv2.putText(out, f"WARNING: {text}", (20, y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.82, color, 2)

        status = {
            "drowsy_on":  self._active(self.ts_drowsy),
            "gaze_l_on":  self._active(self.ts_gaze_l),
            "gaze_r_on":  self._active(self.ts_gaze_r),
            "yawn_on":    self._active(self.ts_yawn),
            "head_on":    self._active(self.ts_head),
            "cnt_drowsy": self.cnt_drowsy,
            "cnt_gaze_l": self.cnt_gaze_l,
            "cnt_gaze_r": self.cnt_gaze_r,
            "cnt_yawn":   self.cnt_yawn,
            "cnt_head":   self.cnt_head,
            "frames":     self.frame_n,
        }
        return out, status


# ── Model loader (cached across reruns) ───────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙ Configuration")
    model_path = st.text_input("Model weights", value="best.pt")
    wav_path   = st.text_input("Alarm .wav file", value="alarm.wav")

    st.markdown("---")
    source_type = st.radio("Video source", [
        "Webcam (local)",
        "IP Webcam (phone)",
        "Upload video file",
    ])
    ip_url       = ""
    uploaded_vid = None
    if source_type == "IP Webcam (phone)":
        ip_url = st.text_input("Stream URL", placeholder="http://192.168.x.x:8080/video")
    elif source_type == "Upload video file":
        uploaded_vid = st.file_uploader("Video file", type=["mp4","avi","mov"])

    st.markdown("---")
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)
    CONF_THRESH = conf_thresh

    st.markdown("---")
    st.markdown("""
**Alert thresholds** *(fraction of 2-sec window)*
- Drowsy / Gaze: 80 %
- Yawn: 100 %
""")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:1rem;margin-bottom:1rem'>
  <span style='font-size:2.2rem'>🚗</span>
  <div>
    <div style='font-family:Share Tech Mono,monospace;font-size:1.55rem;
                color:#e8eaf0;letter-spacing:0.04em'>SafeDrive DMS</div>
    <div style='font-size:0.82rem;color:#445;letter-spacing:0.08em'>
      DRIVER MONITORING SYSTEM · YOLOv8</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_vid, col_panel = st.columns([3, 1])

with col_vid:
    frame_ph = st.empty()

with col_panel:
    st.markdown("<div class='section-title'>Live Alerts</div>", unsafe_allow_html=True)
    alert_ph = st.empty()
    st.markdown("<div class='section-title' style='margin-top:1rem'>Session Stats</div>",
                unsafe_allow_html=True)
    stats_ph = st.empty()

st.markdown("---")
c1, c2, _ = st.columns([1, 1, 5])
start_btn = c1.button("▶  Start", type="primary", use_container_width=True)
stop_btn  = c2.button("⏹  Stop",  use_container_width=True)

if "running" not in st.session_state:
    st.session_state.running = False
if start_btn: st.session_state.running = True
if stop_btn:  st.session_state.running = False


# ── Detection loop ────────────────────────────────────────────────────────────
def render_alerts(s):
    html = ""
    if s["drowsy_on"]:
        html += '<div class="alert-card card-drowsy">🔴 DROWSY — EYES CLOSED</div>'
    if s["gaze_l_on"]:
        html += '<div class="alert-card card-gaze">◀ LOOKING LEFT</div>'
    if s["gaze_r_on"]:
        html += '<div class="alert-card card-gaze">▶ LOOKING RIGHT</div>'
    if s["yawn_on"]:
        html += '<div class="alert-card card-yawn">😮 YAWNING DETECTED</div>'
    if s["head_on"]:
        html += '<div class="alert-card card-head">↕ HEAD MOVEMENT</div>'
    if not html:
        html = '<div class="alert-card card-ok">✔ DRIVER ALERT — OK</div>'
    return html

def render_stats(s):
    tiles = [
        (s["frames"],     "#4af", "Frames"),
        (s["cnt_drowsy"], "#f46", "Drowsy"),
        (s["cnt_gaze_l"], "#fa6", "Gaze-L"),
        (s["cnt_gaze_r"], "#fa6", "Gaze-R"),
        (s["cnt_yawn"],   "#fd4", "Yawn"),
        (s["cnt_head"],   "#a6f", "Head"),
    ]
    inner = ""
    for val, color, label in tiles:
        inner += f"""
        <div class='stat-tile'>
          <div class='stat-num' style='color:{color}'>{val}</div>
          <div class='stat-label'>{label}</div>
        </div>"""
    return f"<div class='stat-grid'>{inner}</div>"


if st.session_state.running:
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    if source_type == "Webcam (local)":
        cap = cv2.VideoCapture(0)
    elif source_type == "IP Webcam (phone)":
        clean = ip_url.strip()
        if not clean:
            st.warning("Enter the IP Webcam URL first.")
            st.stop()
        cap = cv2.VideoCapture(clean)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        if uploaded_vid is None:
            st.warning("Upload a video file first.")
            st.stop()
        tmp = f"/tmp/dms_{int(time.time())}.mp4"
        with open(tmp, "wb") as f:
            f.write(uploaded_vid.read())
        cap = cv2.VideoCapture(tmp)

    if not cap.isOpened():
        st.error("Cannot open video source. Check camera index / URL / file.")
        st.stop()

    fps      = cap.get(cv2.CAP_PROP_FPS) or FPS
    detector = DrowsinessDetector(fps=fps, wav_path=wav_path)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.info("Stream ended.")
            break

        annotated, status = detector.process(frame, model)
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        frame_ph.image(rgb_frame, channels="RGB", use_container_width=True)
        alert_ph.markdown(render_alerts(status), unsafe_allow_html=True)
        stats_ph.markdown(render_stats(status),  unsafe_allow_html=True)

    cap.release()
    st.session_state.running = False
    st.success("Detection stopped.")
else:
    frame_ph.markdown("""
    <div style='background:#0d1117;border:1px dashed #1e2530;border-radius:8px;
                height:380px;display:flex;align-items:center;justify-content:center;
                color:#2a3545;font-family:Share Tech Mono,monospace;font-size:1rem;
                letter-spacing:0.1em'>
      PRESS ▶ START TO BEGIN MONITORING
    </div>
    """, unsafe_allow_html=True)
