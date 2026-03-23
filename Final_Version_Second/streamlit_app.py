"""
Driver Drowsiness Detection - Streamlit App
Supports: webcam (local), IP Webcam (phone), and uploaded video file
"""

import streamlit as st
import cv2
import numpy as np
import datetime
import time
import threading
import base64
from collections import deque
from pathlib import Path
from ultralytics import YOLO

# ─── Constants ────────────────────────────────────────────────────────────────
FPS                   = 30
QUEUE_DURATION        = 2          # seconds of history kept in each deque
DROWSY_THRESHOLD_FRAC = 0.8        # fraction of queue frames that must be positive
YAWN_THRESHOLD_FRAC   = 1.0
HEAD_THRESHOLD_FRAC   = 0.8
WARNING_DURATION      = 2          # seconds to keep on-screen warning alive

# YOLOv8 class-ID mapping from this model
# Labels  0,1,2 → eyes-closed states  (drowsy cue)
# Labels  4,5   → head up / head down (head-movement cue)
# Label   8     → yawn               (yawn cue)
EYE_CLOSED_LABELS  = {0, 1, 2}
HEAD_MOTION_LABELS = {4, 5}
YAWN_LABEL         = 8
CONFIDENCE_THRESH  = 0.5

CLASS_NAMES = {
    0: "Closed_Eyes_Left",
    1: "Closed_Eyes_Right",
    2: "Closed_Eyes_Both",
    3: "Open_Eyes",
    4: "Head_Up",
    5: "Head_Down",
    6: "Face",
    7: "Open_Mouth",
    8: "Yawn",
}

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeDrive DMS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Minimal custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .alert-box {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .alert-drowsy  { background: #ff4b4b33; border: 2px solid #ff4b4b; color: #ff4b4b; }
    .alert-yawn    { background: #ffa50033; border: 2px solid #ffa500; color: #ffa500; }
    .alert-head    { background: #ffd70033; border: 2px solid #ffd700; color: #ffd700; }
    .alert-ok      { background: #00c85133; border: 2px solid #00c851; color: #00c851; }
    .stat-label { font-size: 0.78rem; color: #aaa; margin-bottom: 0; }
    .stat-val   { font-size: 1.6rem; font-weight: 700; margin-top: 0; }
</style>
""", unsafe_allow_html=True)


# ─── Model loader (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    return YOLO(path)


# ─── Core frame-processing logic ─────────────────────────────────────────────
class DrowsinessDetector:
    """Stateful detector: call process_frame() per frame."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        q_len = int(fps * QUEUE_DURATION)
        self.eye_q  = deque(maxlen=q_len)
        self.yawn_q = deque(maxlen=q_len)
        self.head_q = deque(maxlen=q_len)

        self.drowsy_thr = int(fps * QUEUE_DURATION * DROWSY_THRESHOLD_FRAC)
        self.yawn_thr   = int(fps * QUEUE_DURATION * YAWN_THRESHOLD_FRAC)
        self.head_thr   = int(fps * QUEUE_DURATION * HEAD_THRESHOLD_FRAC)

        self.drowsy_ts = None
        self.yawn_ts   = None
        self.head_ts   = None

        self.total_drowsy_events = 0
        self.total_yawn_events   = 0
        self.total_head_events   = 0
        self.frame_count         = 0

    def process_frame(self, frame: np.ndarray, model) -> tuple[np.ndarray, dict]:
        """
        Run YOLOv8 on one BGR frame, annotate it, update queues.
        Returns (annotated_frame, status_dict).
        """
        self.frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=[rgb], save=False, verbose=False)[0]

        cur_eye   = False
        cur_yawn  = False
        cur_head  = False
        detections = []  # (xmin,ymin,xmax,ymax, label_id, conf)

        boxes   = results.boxes
        xyxy    = boxes.xyxy.cpu().numpy()
        confs   = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            conf  = confs[i]
            if conf < CONFIDENCE_THRESH:
                continue
            label = int(classes[i])
            xmin, ymin, xmax, ymax = map(int, xyxy[i])
            detections.append((xmin, ymin, xmax, ymax, label, conf))

            if label in EYE_CLOSED_LABELS:  cur_eye  = True
            if label in HEAD_MOTION_LABELS: cur_head = True
            if label == YAWN_LABEL:         cur_yawn = True

        # Update rolling queues
        self.eye_q.append(cur_eye)
        self.yawn_q.append(cur_yawn)
        self.head_q.append(cur_head)

        now = datetime.datetime.now()

        # ── Drowsiness check ──
        is_drowsy = sum(self.eye_q) >= self.drowsy_thr
        if is_drowsy:
            if self.drowsy_ts is None:
                self.total_drowsy_events += 1
            self.drowsy_ts = now
        
        # ── Yawn check ──
        is_yawn = sum(self.yawn_q) >= self.yawn_thr
        if is_yawn:
            if self.yawn_ts is None:
                self.total_yawn_events += 1
            self.yawn_ts = now
            self.yawn_q.clear()

        # ── Head-movement check ──
        is_head = sum(self.head_q) >= self.head_thr
        if is_head:
            if self.head_ts is None:
                self.total_head_events += 1
            self.head_ts = now
            self.head_q.clear()

        # ── Annotate frame ──
        annotated = frame.copy()
        for (xmin, ymin, xmax, ymax, label, conf) in detections:
            name = CLASS_NAMES.get(label, str(label))
            # Color logic: red for drowsy-trigger labels, yellow for head/yawn, else green
            if label in EYE_CLOSED_LABELS:
                color = (0, 0, 255) if is_drowsy else (0, 255, 0)
            elif label in HEAD_MOTION_LABELS or label == YAWN_LABEL:
                color = (0, 255, 255)
            else:
                color = (0, 200, 0)

            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(annotated, f"{name} {conf:.2f}",
                        (xmin, max(ymin - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ── On-frame warning overlays ──
        def still_active(ts):
            return ts and (now - ts).total_seconds() < WARNING_DURATION

        if still_active(self.drowsy_ts):
            cv2.putText(annotated, "⚠ DROWSY DETECTED!",  (30, 140),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
        if still_active(self.yawn_ts):
            cv2.putText(annotated, "⚠ YAWNING DETECTED!", (30, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 255), 2)
        if still_active(self.head_ts):
            cv2.putText(annotated, "⚠ HEAD MOVEMENT!",    (30, 95),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 255), 2)

        status = {
            "drowsy": is_drowsy,
            "yawn":   is_yawn,
            "head":   is_head,
            "drowsy_active": still_active(self.drowsy_ts),
            "yawn_active":   still_active(self.yawn_ts),
            "head_active":   still_active(self.head_ts),
            "total_drowsy":  self.total_drowsy_events,
            "total_yawn":    self.total_yawn_events,
            "total_head":    self.total_head_events,
            "frame_count":   self.frame_count,
        }
        return annotated, status


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    model_path = st.text_input("Model path (best.pt)", value="best.pt")

    source_type = st.radio(
        "Video source",
        ["Webcam (local)", "IP Webcam (phone)", "Upload video file"],
    )

    ip_url = ""
    uploaded_file = None

    if source_type == "IP Webcam (phone)":
        ip_url = st.text_input(
            "Stream URL",
            value="http://192.168.x.x:8080/video",
            help="Use IP Webcam Android app. Find the URL on the app's main screen.",
        )
    elif source_type == "Upload video file":
        uploaded_file = st.file_uploader("Upload .mp4 / .avi", type=["mp4", "avi", "mov"])

    st.markdown("---")
    st.markdown("**Detection thresholds**")
    conf_thresh  = st.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)
    CONFIDENCE_THRESH = conf_thresh

    st.markdown("---")
    st.caption("SafeDrive DMS · YOLOv8")

# ─── Main layout ──────────────────────────────────────────────────────────────
st.title("🚗 SafeDrive — Driver Monitoring System")
st.markdown("Real-time drowsiness, yawn, and head-movement detection using **YOLOv8**.")

col_video, col_status = st.columns([3, 1])

with col_video:
    frame_placeholder = st.empty()

with col_status:
    st.markdown("### 🔴 Live Alerts")
    alert_placeholder = st.empty()
    st.markdown("### 📊 Session Stats")
    stats_placeholder = st.empty()

st.markdown("---")
start_col, stop_col, _ = st.columns([1, 1, 4])
start_btn = start_col.button("▶ Start Detection", type="primary", use_container_width=True)
stop_btn  = stop_col.button("⏹ Stop",             use_container_width=True)

if "running" not in st.session_state:
    st.session_state.running = False
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# ─── Detection loop ───────────────────────────────────────────────────────────
if st.session_state.running:
    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Open video source
    if source_type == "Webcam (local)":
        cap = cv2.VideoCapture(0)
    elif source_type == "IP Webcam (phone)":
        cap = cv2.VideoCapture(ip_url)
    else:  # uploaded file
        if uploaded_file is None:
            st.warning("Please upload a video file first.")
            st.stop()
        tmp_path = f"/tmp/dms_input_{int(time.time())}.mp4"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(tmp_path)

    if not cap.isOpened():
        st.error("Cannot open video source. Check camera index / URL / file.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    detector = DrowsinessDetector(fps=fps)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.info("Stream ended or no more frames.")
            break

        annotated, status = detector.process_frame(frame, model)

        # Convert BGR → RGB for Streamlit
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        # Alert panel
        alerts_html = ""
        if status["drowsy_active"]:
            alerts_html += '<div class="alert-box alert-drowsy">😴 DROWSY DETECTED</div>'
        if status["yawn_active"]:
            alerts_html += '<div class="alert-box alert-yawn">🥱 YAWNING DETECTED</div>'
        if status["head_active"]:
            alerts_html += '<div class="alert-box alert-head">⬆ HEAD MOVEMENT</div>'
        if not (status["drowsy_active"] or status["yawn_active"] or status["head_active"]):
            alerts_html = '<div class="alert-box alert-ok">✅ Driver Alert</div>'
        alert_placeholder.markdown(alerts_html, unsafe_allow_html=True)

        # Stats panel
        stats_md = f"""
<p class='stat-label'>Frames processed</p>
<p class='stat-val'>{status['frame_count']}</p>
<p class='stat-label'>Drowsy events</p>
<p class='stat-val' style='color:#ff4b4b'>{status['total_drowsy']}</p>
<p class='stat-label'>Yawn events</p>
<p class='stat-val' style='color:#ffa500'>{status['total_yawn']}</p>
<p class='stat-label'>Head-movement events</p>
<p class='stat-val' style='color:#ffd700'>{status['total_head']}</p>
"""
        stats_placeholder.markdown(stats_md, unsafe_allow_html=True)

    cap.release()
    st.session_state.running = False
    st.success("Detection stopped.")
else:
    frame_placeholder.info("Press **▶ Start Detection** to begin.")
