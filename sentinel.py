# sentinel.py
"""
Rail Safety Automation Dashboard (Streamlit + YOLOv8) — Simplified “Operator-First” Version

What this does (and why it reduces Visual Inspection Costs):
- Replaces continuous human video watching with automated hazard detection (exception-based monitoring).
- Flags only high-risk events (Red Zone Incursions: person OR truck), overlays proof (boxes + confidence),
  and keeps an incident log for review, compliance, and escalation.
- Result: fewer staff-hours spent on uneventful footage, faster response to true hazards,
  and consistent monitoring criteria across shifts and sites.

Install:
  pip install streamlit opencv-python ultralytics pandas
Optional (YouTube Live support):
  pip install cap_from_youtube

Run:
  streamlit run sentinel.py
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# Optional YouTube support (kept non-fatal if not installed)
try:
    from cap_from_youtube import cap_from_youtube  # type: ignore
    HAS_YOUTUBE = True
except Exception:
    cap_from_youtube = None
    HAS_YOUTUBE = False


# =========================
# Configuration (Operator-friendly defaults)
# =========================
DEFAULT_CONF = 0.35
DEFAULT_DEBOUNCE_SEC = 2.0
DEFAULT_FPS_SLEEP = 0.02  # UI smoothness vs CPU usage

# Per requirement: Red Zone Incursion = person OR truck
INCURSION_CLASSES_BASE = {"person", "truck"}

# Demo scenarios: update these paths to real files on your machine/server
SCENARIOS = {
    "Demo A — Standard Crossing": "video1.mp4",
    "Demo B — Pedestrian Hazard": "video2.mp4",
}

FRIENDLY_LABELS = {
    "person": "Person",
    "truck": "Truck",
    "car": "Car",
    "bus": "Bus",
}


# =========================
# Session state
# =========================
def init_state():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "source" not in st.session_state:
        st.session_state.source = None  # {"type": "...", "path": "..."}
    if "last_log_ts" not in st.session_state:
        st.session_state.last_log_ts = {}  # debounce per class
    if "incident_log" not in st.session_state:
        st.session_state.incident_log = pd.DataFrame(columns=["Time", "Type", "Confidence", "Source"])
    if "fps_window" not in st.session_state:
        st.session_state.fps_window = []  # timestamps
    if "reconnect_attempts" not in st.session_state:
        st.session_state.reconnect_attempts = 0
    if "last_frame_time" not in st.session_state:
        st.session_state.last_frame_time = None


def reset_runtime_only():
    st.session_state.fps_window = []
    st.session_state.reconnect_attempts = 0
    st.session_state.last_frame_time = None


def clear_log():
    st.session_state.incident_log = st.session_state.incident_log.iloc[0:0].copy()
    st.session_state.last_log_ts = {}


# =========================
# UI
# =========================
def render_status(is_incursion: bool, people_count: int, incursion_types: List[str]) -> None:
    if is_incursion:
        bg = "#b91c1c"
        text = "Status: INCURSION DETECTED"
        sub = "Incursion: " + ", ".join(sorted(set(incursion_types))) if incursion_types else "Incursion: Detected"
    else:
        bg = "#15803d"
        text = "Status: ALL CLEAR"
        sub = "Incursion: None"

    st.markdown(
        f"""
        <div style="
            width: 100%;
            padding: 16px 18px;
            border-radius: 14px;
            background: {bg};
            color: white;
            font-size: 26px;
            font-weight: 900;
            letter-spacing: 0.4px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 10px 24px rgba(0,0,0,0.20);
            margin-bottom: 16px;
        ">
            <div>
                <div>{text}</div>
                <div style="font-size: 14px; font-weight: 700; opacity: 0.95; margin-top: 4px;">{sub}</div>
                <div style="font-size: 13px; font-weight: 650; opacity: 0.90; margin-top: 2px;">
                    Red Zone Rule: <b>Person</b> OR <b>Truck</b> = Incursion
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size: 14px; font-weight: 850; opacity: 0.98;">People Detected</div>
                <div style="font-size: 34px; font-weight: 950; line-height: 1.0;">{people_count}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_checklist():
    st.markdown(
        """
        **Setup Checklist**
        1) Choose a **Camera Source**  
        2) Click **START MONITORING**  
        3) Watch the **Status** and **Incident Log**
        """
    )


# =========================
# Vision helpers
# =========================
def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_boxes(frame_bgr, detections: List[Dict[str, Any]]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        cls = det["cls_name"]
        label = f'{FRIENDLY_LABELS.get(cls, cls)}  {det["conf"]:.0%}'
        color = (0, 255, 0)
        if det["is_incursion"]:
            color = (0, 0, 255)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(0, y1 - th - 10)
        cv2.rectangle(frame_bgr, (x1, y0), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            frame_bgr, label, (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )


def infer_frame(model: YOLO, frame_bgr, conf_threshold: float, incursion_classes: set) -> Dict[str, Any]:
    result = model.predict(frame_bgr, conf=conf_threshold, verbose=False)[0]
    names = result.names
    boxes = result.boxes

    detections: List[Dict[str, Any]] = []
    people_count = 0
    incursion_hits: List[Tuple[str, float]] = []

    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            cls_id = int(b.cls[0].item())
            cls_name = str(names.get(cls_id, cls_id))
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

            is_incursion = cls_name in incursion_classes
            if cls_name == "person":
                people_count += 1
            if is_incursion:
                incursion_hits.append((cls_name, conf))

            detections.append({
                "cls_name": cls_name,
                "conf": conf,
                "xyxy": (x1, y1, x2, y2),
                "is_incursion": is_incursion,
            })

    return {
        "detections": detections,
        "incursion": len(incursion_hits) > 0,
        "people_count": people_count,
        "incursion_hits": incursion_hits,
    }


# =========================
# Stream helpers
# =========================
def release_capture():
    cap = st.session_state.get("cap", None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.cap = None


def stop_stream():
    st.session_state.running = False
    release_capture()
    reset_runtime_only()


def start_stream(source_type: str, path_or_index: Any):
    release_capture()

    cap = None
    try:
        if source_type == "YouTube Live":
            if not HAS_YOUTUBE:
                st.error("YouTube Live requires: pip install cap_from_youtube")
                st.session_state.running = False
                return
            cap = cap_from_youtube(path_or_index, resolution="720p")  # type: ignore[misc]
        else:
            cap = cv2.VideoCapture(path_or_index)
    except Exception as e:
        st.error(f"Could not start stream: {e}")
        st.session_state.running = False
        return

    st.session_state.cap = cap
    st.session_state.source = {"type": source_type, "path": path_or_index}
    st.session_state.running = True
    reset_runtime_only()


def compute_fps() -> float:
    now = time.time()
    w = st.session_state.fps_window
    w.append(now)
    cutoff = now - 2.0
    while w and w[0] < cutoff:
        w.pop(0)
    if len(w) < 2:
        return 0.0
    return (len(w) - 1) / (w[-1] - w[0] + 1e-9)


def append_incidents(rows: List[Dict[str, Any]], max_rows: int = 800) -> None:
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    st.session_state.incident_log = pd.concat([df_new, st.session_state.incident_log], ignore_index=True).head(max_rows)


def log_incursions_debounced(
    incursion_hits: List[Tuple[str, float]],
    min_interval_sec: float,
    source_label: str
) -> None:
    now = time.time()
    rows = []
    for cls_name, conf in incursion_hits:
        last = st.session_state.last_log_ts.get(cls_name, 0.0)
        if (now - last) >= min_interval_sec:
            st.session_state.last_log_ts[cls_name] = now
            rows.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Type": FRIENDLY_LABELS.get(cls_name, cls_name),
                "Confidence": f"{conf:.0%}",
                "Source": source_label,
            })
    append_incidents(rows)


def is_likely_file_path(s: str) -> bool:
    s2 = s.lower().strip()
    return s2.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))


# =========================
# App
# =========================
st.set_page_config(page_title="Rail Safety Automation", layout="wide")
init_state()

st.title("Rail Safety Automation Dashboard")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------- Sidebar: simplified “wizard” ----------
with st.sidebar:
    st.header("Operator Setup")
    show_checklist()
    st.divider()

    source_choice = st.selectbox(
        "Camera Source",
        ["Demo Scenario", "Webcam", "RTSP/HTTP Stream", "YouTube Live"],
        index=0,
    )

    # Only show the one input relevant to the chosen source
    source_value: Any = None

    if source_choice == "Demo Scenario":
        selected = st.selectbox("Scenario", list(SCENARIOS.keys()))
        source_value = SCENARIOS[selected]
        st.caption("Update demo video file paths at the top of this script.")

    elif source_choice == "Webcam":
        cam_index = st.number_input("Camera Index", value=0, step=1)
        source_value = int(cam_index)

    elif source_choice == "RTSP/HTTP Stream":
        source_value = st.text_input(
            "Stream URL",
            value="rtsp://user:pass@192.168.1.55:554/stream",
            help="Paste RTSP or HTTP stream URL. Must be OpenCV VideoCapture compatible.",
        )

    elif source_choice == "YouTube Live":
        if not HAS_YOUTUBE:
            st.warning("YouTube Live disabled (install cap_from_youtube).")
        source_value = st.text_input(
            "YouTube URL",
            value="https://www.youtube.com/watch?v=SomeLiveStreamID",
            help="Optional testing source. Requires cap_from_youtube.",
        )

    st.divider()

    # Big, obvious Start/Stop
    col1, col2 = st.columns(2)
    with col1:
        start_disabled = (source_choice == "YouTube Live" and not HAS_YOUTUBE)
        if st.button("START MONITORING", use_container_width=True, disabled=start_disabled):
            start_stream(source_choice, source_value)

    with col2:
        if st.button("STOP", use_container_width=True):
            stop_stream()

    col3, col4 = st.columns(2)
    with col3:
        if st.button("CLEAR LOG", use_container_width=True):
            clear_log()
    with col4:
        # optional: reload without touching the log
        if st.button("RESET STREAM", use_container_width=True):
            src = st.session_state.source or {}
            if src:
                start_stream(src.get("type", source_choice), src.get("path", source_value))

    st.divider()

    # Advanced settings (hidden by default)
    with st.expander("Advanced Settings (Engineering)", expanded=False):
        conf_threshold = st.slider("Detection Confidence", 0.05, 0.95, DEFAULT_CONF, 0.05)
        log_interval_sec = st.slider("Log Debounce (sec)", 0.5, 10.0, DEFAULT_DEBOUNCE_SEC, 0.5)
        expand_vehicles = st.checkbox("Also treat Car/Bus as incursion", value=False)
        loop_sleep = st.slider("Loop Sleep (sec)", 0.0, 0.10, DEFAULT_FPS_SLEEP, 0.01)
        st.caption("Defaults are set for operators. Only tune if needed.")

# Defaults if Advanced expander never opened
if "conf_threshold" not in locals():
    conf_threshold = DEFAULT_CONF
if "log_interval_sec" not in locals():
    log_interval_sec = DEFAULT_DEBOUNCE_SEC
if "expand_vehicles" not in locals():
    expand_vehicles = False
if "loop_sleep" not in locals():
    loop_sleep = DEFAULT_FPS_SLEEP

# ---------- Main Layout ----------
left, right = st.columns([1.6, 1.0], gap="large")

with left:
    st.subheader("Live Feed")
    status_slot = st.empty()
    video_slot = st.empty()
    health_slot = st.empty()

with right:
    st.subheader("Incident Log")
    log_slot = st.empty()
    st.caption("Newest events appear at the top (debounced).")

# Idle state
if not st.session_state.running:
    render_status(False, 0, [])
    video_slot.info("Choose a Camera Source in the sidebar, then click START MONITORING.")
    health_slot.caption("Stream Health: Not running")
    log_slot.dataframe(st.session_state.incident_log, use_container_width=True, height=540, hide_index=True)
    st.stop()

# Validate capture
cap = st.session_state.cap
cap_opened = False
try:
    cap_opened = (cap is not None) and bool(cap.isOpened())
except Exception:
    cap_opened = cap is not None

if not cap_opened:
    stop_stream()
    render_status(False, 0, [])
    video_slot.error("Could not open camera/stream. Check URL/path, credentials, or camera availability.")
    log_slot.dataframe(st.session_state.incident_log, use_container_width=True, height=540, hide_index=True)
    st.stop()

# Read one frame per rerun
ret, frame = cap.read()
if not ret or frame is None:
    src = st.session_state.source or {}
    src_type = src.get("type", "unknown")
    src_path = src.get("path", "")

    # End-of-file for demo videos: stop cleanly
    if src_type == "Demo Scenario" and isinstance(src_path, str) and is_likely_file_path(src_path):
        stop_stream()
        render_status(False, 0, [])
        video_slot.info("Demo scenario finished (end of file). Choose another scenario and start again.")
        log_slot.dataframe(st.session_state.incident_log, use_container_width=True, height=540, hide_index=True)
        st.stop()

    # Live sources: best-effort reconnect
    st.session_state.reconnect_attempts += 1
    video_slot.warning(f"Stream read failed. Retrying... (attempt {st.session_state.reconnect_attempts})")
    time.sleep(0.25)
    # One quick restart using stored source
    if st.session_state.reconnect_attempts <= 10 and st.session_state.source:
        start_stream(src.get("type", "RTSP/HTTP Stream"), src.get("path"))
    st.rerun()

# Reset reconnect counter on success
st.session_state.reconnect_attempts = 0
st.session_state.last_frame_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Incursion classes
incursion_classes = set(INCURSION_CLASSES_BASE)
if expand_vehicles:
    incursion_classes |= {"car", "bus"}

# Inference + overlay
out = infer_frame(model, frame, conf_threshold, incursion_classes)
detections = out["detections"]
is_incursion = out["incursion"]
people_count = out["people_count"]
incursion_hits = out["incursion_hits"]

draw_boxes(frame, detections)

# Log events
src_label = (st.session_state.source or {}).get("type", "unknown")
if is_incursion:
    log_incursions_debounced(incursion_hits, float(log_interval_sec), source_label=str(src_label))

# Render status + video
incursion_types = [FRIENDLY_LABELS.get(c, c) for c, _ in incursion_hits]
render_status(is_incursion, people_count, incursion_types)

video_slot.image(bgr_to_rgb(frame), channels="RGB", use_container_width=True)

# Stream health indicators
fps = compute_fps()
h, w = frame.shape[:2]
src_info = st.session_state.source or {}
health_slot.caption(
    f"Stream Health — Connected: Yes | Source: {src_info.get('type', 'unknown')} | "
    f"FPS: {fps:.1f} | Resolution: {w}x{h} | Last Frame: {st.session_state.last_frame_time}"
)

# Incident log (scrollable via height)
log_slot.dataframe(st.session_state.incident_log, use_container_width=True, height=540, hide_index=True)

# Loop pacing
time.sleep(float(loop_sleep))
st.rerun()
