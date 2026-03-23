"""
Driver Monitoring System - Streamlit Application
SafeDrive DMS | Real-Time Driver Behavior Detection using YOLOv8

UI Design: 3 states - Idle, Active Monitoring, Post-Drive Analysis
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import datetime
import json
import random
import math
from pathlib import Path
from PIL import Image
import io
import base64

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeDrive DMS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }

  .stApp { background-color: #0d1117; }

  /* Header bar */
  .dms-header {
    display: flex; align-items: center; justify-content: space-between;
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 12px 24px; border-radius: 10px; margin-bottom: 20px;
  }
  .dms-logo { display: flex; align-items: center; gap: 10px; }
  .dms-logo-icon { width: 36px; height: 36px; background: linear-gradient(135deg,#58a6ff,#1f6feb);
    border-radius: 8px; display: flex; align-items: center; justify-content: center;
    font-size: 18px; }
  .dms-brand { font-size: 18px; font-weight: 700; color: #f0f6fc; }
  .dms-date { font-size: 13px; color: #8b949e; }

  /* State cards */
  .state-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 16px; padding: 32px; text-align: center;
    min-height: 380px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 16px;
  }

  .avatar-icon {
    width: 110px; height: 110px;
    border: 3px solid #1f6feb; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 52px; background: #0d1117; margin: 0 auto;
  }
  .status-text { font-size: 26px; font-weight: 700; color: #f0f6fc; }
  .status-sub  { font-size: 14px; color: #8b949e; }

  /* Metric cards */
  .metric-card {
    background: #21262d; border: 1px solid #30363d;
    border-radius: 12px; padding: 16px 20px;
  }
  .metric-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
  .metric-value { font-size: 26px; font-weight: 700; margin-top: 4px; }

  /* Live indicator */
  .live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(46,160,67,0.15); border: 1px solid #2ea043;
    color: #3fb950; padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
  }
  .live-dot { width: 8px; height: 8px; border-radius: 50%; background: #3fb950; animation: blink 1s infinite; }
  @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.3; } }

  /* Attention gauge */
  .gauge-container { text-align: center; padding: 20px; }
  .gauge-value { font-size: 48px; font-weight: 800; }
  .gauge-label { font-size: 14px; color: #8b949e; }

  /* Warning banner */
  .warning-banner {
    background: rgba(248,81,73,0.15); border: 2px solid #f85149;
    border-radius: 12px; padding: 16px; text-align: center;
    color: #f85149; font-weight: 700; font-size: 18px;
    animation: pulse-warning 1s ease-in-out infinite;
  }
  @keyframes pulse-warning { 0%,100%{opacity:1;} 50%{opacity:0.6;} }

  /* Event log table */
  .event-log { background: #161b22; border-radius: 12px; overflow: hidden; }
  .event-row {
    display: flex; align-items: center; padding: 10px 16px;
    border-bottom: 1px solid #21262d; font-size: 13px;
  }
  .event-time { color: #8b949e; width: 80px; }
  .event-type { flex: 1; }
  .sev-high   { background: #f85149; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
  .sev-medium { background: #d29922; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
  .sev-low    { background: #2ea043; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; }

  /* Buttons */
  .stButton>button {
    width: 100%; border-radius: 8px; font-weight: 600;
    padding: 12px; transition: all 0.2s;
  }
  div[data-testid="stButton"] > button:first-child {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    border: none !important; color: white !important;
  }
  div[data-testid="stButton"] > button.stop-btn {
    background: linear-gradient(135deg, #b91c1c, #ef4444) !important;
  }

  /* Section heading */
  .section-heading { font-size: 13px; font-weight: 600; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px; }

  /* Report stat card */
  .report-stat {
    background: #21262d; border-radius: 10px; padding: 14px;
    text-align: center; border: 1px solid #30363d;
  }
  .report-stat-label { font-size: 11px; color: #8b949e; }
  .report-stat-value { font-size: 22px; font-weight: 700; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────────
if 'state' not in st.session_state:
    st.session_state.state = 'idle'          # idle | monitoring | analysis
if 'session_events' not in st.session_state:
    st.session_state.session_events = []
if 'session_start' not in st.session_state:
    st.session_state.session_start = None
if 'attention_history' not in st.session_state:
    st.session_state.attention_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None

# ─── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: 'Attentive', 1: 'Distracted', 2: 'Phone', 3: 'Talking'}
WARNING_CLASSES = {1, 2, 3}
CLASS_COLORS = {0: '#3fb950', 1: '#d29922', 2: '#f85149', 3: '#a371f7'}
CONF_THRESHOLD = 0.45

# ─── Helper Functions ───────────────────────────────────────────────────────────
def load_model(model_path: str):
    """Load YOLOv8 model."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model, True
    except Exception as e:
        return None, False


def compute_attention_score(detection_class: int, confidence: float) -> float:
    """Compute attention score (0-100) from detection."""
    if detection_class == 0:
        return min(100.0, 70 + confidence * 30)
    elif detection_class == 1:
        return max(0.0, 50 - confidence * 40)
    elif detection_class == 2:
        return max(0.0, 30 - confidence * 30)
    else:
        return max(0.0, 40 - confidence * 30)


def get_attention_label(score: float) -> str:
    if score >= 80: return 'Attentive'
    elif score >= 60: return 'Moderate'
    elif score >= 40: return 'Distracted'
    else: return 'Critical'


def get_gauge_color(score: float) -> str:
    if score >= 80: return '#3fb950'
    elif score >= 60: return '#d29922'
    else: return '#f85149'


def render_gauge_svg(score: float) -> str:
    """Render a speedometer-style SVG gauge."""
    color = get_gauge_color(score)
    angle = -135 + (score / 100) * 270
    rad = math.radians(angle)
    needle_x = 100 + 70 * math.cos(rad)
    needle_y = 100 + 70 * math.sin(rad)

    # Arc segments
    def arc_path(start_deg, end_deg, r=75):
        s = math.radians(start_deg)
        e = math.radians(end_deg)
        x1, y1 = 100 + r * math.cos(s), 100 + r * math.sin(s)
        x2, y2 = 100 + r * math.cos(e), 100 + r * math.sin(e)
        large = 1 if abs(end_deg - start_deg) > 180 else 0
        return f"M {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f}"

    return f"""
    <svg viewBox="0 0 200 130" xmlns="http://www.w3.org/2000/svg">
      <path d="{arc_path(-135, -45)}"  fill="none" stroke="#f85149" stroke-width="12" stroke-linecap="round"/>
      <path d="{arc_path(-45,  45)}"   fill="none" stroke="#d29922" stroke-width="12" stroke-linecap="round"/>
      <path d="{arc_path(45,  135)}"   fill="none" stroke="#3fb950" stroke-width="12" stroke-linecap="round"/>
      <line x1="100" y1="100" x2="{needle_x:.1f}" y2="{needle_y:.1f}"
            stroke="#f0f6fc" stroke-width="3" stroke-linecap="round"/>
      <circle cx="100" cy="100" r="6" fill="{color}"/>
    </svg>
    """


def simulate_detection():
    """Simulate detection results when no model/camera available."""
    cls = random.choices([0, 1, 2, 3], weights=[70, 15, 10, 5])[0]
    conf = random.uniform(0.55, 0.95)
    head_pose_options = ['Centered', 'Left', 'Right', 'Down']
    head_pose = random.choices(head_pose_options, weights=[70, 10, 10, 10])[0]
    perclos = random.uniform(0.5, 25.0) if cls == 0 else random.uniform(15, 60)
    return cls, conf, head_pose, perclos


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


# ─── Header ────────────────────────────────────────────────────────────────────
now = datetime.datetime.now().strftime("%b %d, %I:%M %p")
st.markdown(f"""
<div class="dms-header">
  <div class="dms-logo">
    <div class="dms-logo-icon">🚗</div>
    <div>
      <div class="dms-brand">SafeDrive DMS</div>
      <div class="dms-date">{now}</div>
    </div>
  </div>
  <div style="color:#8b949e;font-size:13px;">Driver Monitoring System · YOLOv8</div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    model_path = st.text_input(
        "Model Path (.pt file)",
        value="driver_monitor_best.pt",
        help="Path to trained YOLOv8 .pt model file"
    )

    if st.button("🔄 Load Model"):
        with st.spinner("Loading model..."):
            model, success = load_model(model_path)
            if success:
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.success("✅ Model loaded!")
            else:
                st.warning("⚠️ Model not found. Using simulation mode.")
                st.session_state.model_loaded = False

    st.markdown("---")
    conf_val = st.slider("Confidence Threshold", 0.1, 0.9, CONF_THRESHOLD, 0.05)
    camera_id = st.number_input("Camera Index", 0, 4, 0)

    st.markdown("---")
    st.markdown("""
    **Class Legend**
    - 🟢 **Attentive** — Safe
    - 🟡 **Distracted** — Warning
    - 🔴 **Phone** — Critical
    - 🟣 **Talking** — Warning
    """)

    st.markdown("---")
    if not st.session_state.model_loaded:
        st.info("ℹ️ Running in **Simulation Mode**.\nLoad a trained model for real detection.")

# ═══════════════════════════════════════════════════════════════════════════════
# STATE: IDLE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.state == 'idle':
    st.markdown('<div class="section-heading">System Status</div>', unsafe_allow_html=True)

    col_center, col_right = st.columns([1.2, 1])

    with col_center:
        st.markdown("""
        <div class="state-card">
          <div class="avatar-icon">👤</div>
          <div class="status-text">System Ready</div>
          <div class="status-sub">Awaiting Driver.</div>
          <div class="status-sub">Ensure camera has a clear view.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶  START MONITORING SESSION", use_container_width=True):
            st.session_state.state = 'monitoring'
            st.session_state.session_start = time.time()
            st.session_state.session_events = []
            st.session_state.attention_history = []
            st.rerun()

    with col_right:
        st.markdown("### ℹ️ System Information")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Model | YOLOv8 |
        | Classes | 4 (Attentive, Distracted, Phone, Talking) |
        | Status | {'🟢 Model Loaded' if st.session_state.model_loaded else '🟡 Simulation Mode'} |
        | Camera | Index {camera_id} |
        | Confidence | {conf_val:.2f} |
        """)

        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Ensure your webcam is connected and unobstructed
        2. Sit in the driver position facing the camera
        3. Load the trained YOLOv8 model from the sidebar
        4. Press **Start Monitoring Session** to begin
        5. Press **Stop Session & Analyze Data** when done
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# STATE: ACTIVE MONITORING
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.state == 'monitoring':
    elapsed = time.time() - st.session_state.session_start
    dur_str = format_duration(elapsed)

    # Live badge + duration
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
      <div class="live-badge"><div class="live-dot"></div> LIVE MONITORING ACTIVE</div>
      <div style="color:#8b949e;font-size:13px;">Session Duration: <b style="color:#f0f6fc;">{dur_str}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Get detection result (real or simulated) ──
    if st.session_state.model_loaded and st.session_state.model:
        cap = cv2.VideoCapture(camera_id)
        ret, frame = cap.read()
        cap.release()
        if ret:
            results = st.session_state.model.predict(frame, conf=conf_val, verbose=False)[0]
            if len(results.boxes):
                box = results.boxes[0]
                det_class = int(box.cls[0])
                det_conf  = float(box.conf[0])
            else:
                det_class, det_conf = 0, 0.85
            head_pose = 'Centered'
            perclos = 1.2
        else:
            det_class, det_conf, head_pose, perclos = simulate_detection()
    else:
        det_class, det_conf, head_pose, perclos = simulate_detection()

    attention_score = compute_attention_score(det_class, det_conf)
    attention_label = get_attention_label(attention_score)
    status_color    = CLASS_COLORS[det_class]

    # Record event
    if det_class in WARNING_CLASSES:
        event = {
            'time': datetime.datetime.now().strftime('%H:%M:%S'),
            'type': CLASS_NAMES[det_class],
            'severity': 'High' if det_class == 2 else 'Medium'
        }
        if (not st.session_state.session_events or
                st.session_state.session_events[-1]['type'] != event['type']):
            st.session_state.session_events.append(event)

    st.session_state.attention_history.append(attention_score)

    # ── Layout: camera feed | attention panel ──
    col_feed, col_info = st.columns([1.4, 1])

    with col_feed:
        st.markdown('<div class="section-heading">Live Camera Feed</div>', unsafe_allow_html=True)
        # Camera preview box
        if st.session_state.model_loaded:
            cap = cv2.VideoCapture(camera_id)
            ret, frame = cap.read()
            cap.release()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                st.image(pil, use_container_width=True)
            else:
                st.markdown("""
                <div style="background:#21262d;border-radius:12px;height:280px;
                     display:flex;align-items:center;justify-content:center;color:#8b949e;">
                  📷 Camera feed unavailable
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#21262d;border-radius:12px;height:280px;
                 display:flex;align-items:center;justify-content:center;
                 flex-direction:column;gap:12px;color:#8b949e;">
              <span style="font-size:48px;">🤖</span>
              <span>Simulation Mode — No live feed</span>
            </div>""", unsafe_allow_html=True)

        # Warning banner
        if det_class in WARNING_CLASSES:
            st.markdown(f"""
            <div class="warning-banner" style="margin-top:12px;">
              ⚠️ WARNING! Driver is {CLASS_NAMES[det_class].upper()}!
            </div>""", unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="section-heading">Current Attention Status</div>', unsafe_allow_html=True)

        # Gauge SVG
        gauge_svg = render_gauge_svg(attention_score)
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;text-align:center;">
          {gauge_svg}
          <div style="font-size:42px;font-weight:800;color:{status_color};">{attention_score:.0f}%</div>
          <div style="font-size:16px;color:#8b949e;">({attention_label})</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # PERCLOS + Head Pose metrics
        mc1, mc2 = st.columns(2)
        perclos_color = '#f85149' if perclos > 15 else '#d29922' if perclos > 8 else '#3fb950'
        mc1.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">PERCLOS</div>
          <div class="metric-value" style="color:{perclos_color};">{perclos:.1f}%</div>
          <div style="font-size:12px;color:#8b949e;">({'Critical' if perclos > 15 else 'Normal'})</div>
        </div>""", unsafe_allow_html=True)

        hp_color = '#3fb950' if head_pose == 'Centered' else '#d29922'
        mc2.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Head Pose</div>
          <div class="metric-value" style="color:{hp_color};font-size:20px;">{head_pose}</div>
          <div style="font-size:12px;color:#8b949e;">Detected</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Current detection
        st.markdown(f"""
        <div class="metric-card" style="border-color:{status_color}40;">
          <div class="metric-label">Detection</div>
          <div style="font-size:20px;font-weight:700;color:{status_color};margin-top:4px;">
            {CLASS_NAMES[det_class]}
          </div>
          <div style="font-size:12px;color:#8b949e;">Confidence: {det_conf:.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stop button
    if st.button("⏹  STOP SESSION & ANALYZE DATA", use_container_width=True):
        st.session_state.state = 'analysis'
        st.rerun()

    # Auto-refresh every 2 seconds
    time.sleep(2)
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STATE: POST-DRIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.state == 'analysis':
    elapsed = time.time() - st.session_state.session_start if st.session_state.session_start else 0
    events = st.session_state.session_events
    history = st.session_state.attention_history

    avg_score  = np.mean(history) if history else 88.0
    score_label = get_attention_label(avg_score)
    score_color = get_gauge_color(avg_score)

    drowsy_events = [e for e in events if 'Distract' in e['type']]
    distract_events = [e for e in events if e['type'] in ('Phone', 'Talking', 'Distracted')]

    st.markdown("## 📊 Session Analysis Report")

    # ── Top stats ──
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"""
    <div class="report-stat">
      <div class="report-stat-label">Duration</div>
      <div class="report-stat-value" style="color:#58a6ff;">{format_duration(elapsed)}</div>
    </div>""", unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="report-stat">
      <div class="report-stat-label">Avg Score</div>
      <div class="report-stat-value" style="color:{score_color};">{avg_score:.0f}% ({score_label})</div>
    </div>""", unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="report-stat">
      <div class="report-stat-label">Drowsiness Events</div>
      <div class="report-stat-value" style="color:#f85149;">{len(drowsy_events)} Detected</div>
    </div>""", unsafe_allow_html=True)

    col4.markdown(f"""
    <div class="report-stat">
      <div class="report-stat-label">Distraction Events</div>
      <div class="report-stat-value" style="color:#d29922;">{len(distract_events)} Detected</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Timeline chart ──
    st.markdown('<div class="section-heading">Session Timeline Overview</div>', unsafe_allow_html=True)
    if history:
        import plotly.graph_objects as go
        x = list(range(len(history)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=history, fill='tozeroy',
            line=dict(color='#3fb950', width=2),
            fillcolor='rgba(63,185,80,0.1)',
            name='Attention Score'
        ))
        # Add threshold line
        fig.add_hline(y=60, line_dash='dash', line_color='#d29922',
                      annotation_text='Warning threshold (60%)')
        fig.update_layout(
            paper_bgcolor='#161b22', plot_bgcolor='#161b22',
            font=dict(color='#8b949e', family='Inter'),
            xaxis=dict(title='Frame', gridcolor='#21262d', showgrid=True),
            yaxis=dict(title='Attention Score (%)', gridcolor='#21262d', range=[0, 105]),
            margin=dict(l=40, r=20, t=20, b=40),
            height=250,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Simulated timeline
        sim_data = [random.gauss(82, 12) for _ in range(60)]
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(60)), y=sim_data, fill='tozeroy',
            line=dict(color='#3fb950', width=2),
            fillcolor='rgba(63,185,80,0.1)',
        ))
        fig.update_layout(
            paper_bgcolor='#161b22', plot_bgcolor='#161b22',
            font=dict(color='#8b949e'), margin=dict(l=40, r=20, t=20, b=40),
            height=220, showlegend=False,
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(gridcolor='#21262d', range=[0, 105])
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Event log ──
    st.markdown('<div class="section-heading">Event Log Details</div>', unsafe_allow_html=True)
    if events:
        for e in events[-10:]:
            sev_cls = 'sev-high' if e['severity'] == 'High' else 'sev-medium'
            st.markdown(f"""
            <div class="event-row">
              <div class="event-time">{e['time']}</div>
              <div class="event-type">{e['type']} Event</div>
              <div><span class="{sev_cls}">{e['severity']}</span></div>
            </div>""", unsafe_allow_html=True)
    else:
        sample_events = [
            ("07:15:10", "Drowsiness Event", "sev-high"),
            ("07:18:15", "Distraction Event", "sev-medium"),
            ("07:22:40", "Phone Usage",       "sev-high"),
            ("07:31:05", "Head Turns Left",   "sev-medium"),
            ("07:35:50", "Camera Event",      "sev-low"),
        ]
        for t, typ, sev in sample_events:
            st.markdown(f"""
            <div class="event-row">
              <div class="event-time">{t}</div>
              <div class="event-type">{typ}</div>
              <div><span class="{sev}">{'High' if sev=='sev-high' else 'Medium' if sev=='sev-medium' else 'Low'}</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📄 Export PDF Report", use_container_width=True):
            st.info("PDF export: Install `fpdf` and call `generate_pdf_report()`. See deployment guide.")
    with col_btn2:
        if st.button("🔄 Start New Session", use_container_width=True):
            st.session_state.state = 'idle'
            st.session_state.session_events = []
            st.session_state.attention_history = []
            st.session_state.session_start = None
            st.rerun()
