import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
import tempfile
import time
from collections import deque
from datetime import datetime

# ------------- FIREBASE IMPORTS -------------
import firebase_admin
from firebase_admin import credentials, firestore

# ------------- BASIC CONFIG -------------
st.set_page_config(
    page_title="AI Crowd Analysis + DeepSORT",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------- THEME TOGGLE LOGIC -------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

with st.sidebar:
    st.subheader("Theme Settings")
    theme_choice = st.radio("Select Mode", ["Dark", "Light"], index=0 if st.session_state.theme == "Dark" else 1)
    st.session_state.theme = theme_choice

# Define colors based on theme for high contrast
if st.session_state.theme == "Dark":
    bg_color = "#050814"
    card_bg = "#0f172a"
    card_border = "#1f2937"
    text_color = "#e2e8f0"  # Light grey for dark mode
    sub_text_color = "#94a3b8"
    accent_color = "#18dbff"
else:
    bg_color = "#FFFFFF"
    card_bg = "#f1f5f9"
    card_border = "#cbd5e1"
    text_color = "#0f172a"  # Dark blue/black for light mode
    sub_text_color = "#475569"
    accent_color = "#0284c7"

# ------------- FIREBASE SETUP -------------
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"🔥 Firebase Error: {e}")
        st.stop()

db = firestore.client()

# ------------- DB HELPERS -------------
def check_credentials(email, password, role_expected):
    try:
        users_ref = db.collection("users")
        query = users_ref.where("email", "==", email).where("password", "==", password).where("role", "==", role_expected).stream()
        for doc in query:
            return doc.to_dict()
        return None
    except Exception as e:
        st.error(f"Database Error: {e}")
        return None

def log_analysis(max_count, total_unique, alert_triggered):
    if "user_email" in st.session_state and st.session_state.user_email:
        db.collection("logs").add({
            "user_email": st.session_state.user_email,
            "peak_count": max_count,
            "total_unique_people": total_unique,
            "alert_triggered": alert_triggered,
            "timestamp": firestore.SERVER_TIMESTAMP
        })

def update_live_firestore(current_count, total_unique):
    try:
        doc_ref = db.collection("people_counter").document("live")
        doc_ref.set({
            "current_inside": current_count,
            "total_detected": total_unique,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, merge=True)
    except:
        pass

# ------------- CUSTOM CSS (Enhanced for Visibility) -------------
st.markdown(f"""
    <style>
    /* Main app background */
    .stApp {{ background-color: {bg_color}; }}
    
    /* Global text color override for visibility */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, .stTextInput label {{
        color: {text_color} !important;
    }}

    /* Custom Metric Cards */
    .metric-card {{
        background-color: {card_bg};
        padding: 15px;
        border-radius: 10px;
        border: 1px solid {card_border};
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }}
    .metric-value {{ font-size: 2.5rem; font-weight: 800; color: {accent_color}; }}
    .metric-label {{ font-size: 1rem; color: {sub_text_color}; font-weight: 600; }}

    /* Overcrowding Alert */
    .overcrowded-alert {{
        background-color: #7f1d1d;
        color: #fecaca !important;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ef4444;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 20px;
        animation: blinker 1s linear infinite;
    }}
    @keyframes blinker {{ 50% {{ opacity: 0.3; }} }}

    /* Fix for input fields in light mode */
    .stTextInput input {{
        background-color: {card_bg};
        color: {text_color};
        border: 1px solid {card_border};
    }}
    </style>
""", unsafe_allow_html=True)

# ------------- MODEL & TRACKER LOADER -------------
@st.cache_resource
def load_assets():
    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)
    return model, tracker

model, tracker = load_assets()

# ------------- HEATMAP HELPER -------------
def generate_heatmap(centroids, frame_shape):
    h, w, _ = frame_shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    for cx, cy in centroids:
        if 0 <= cx < w and 0 <= cy < h:
            heatmap[int(cy), int(cx)] += 1
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=30, sigmaY=30)
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

# ------------- CORE LOGIC -------------
def process_video(video_source, crowd_limit, is_webcam=False):
    st.markdown("### 🔴 Live Tracking Dashboard")
    alert_placeholder = st.empty()

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        live_count_ph = st.empty()
    with m_col2:
        total_unique_ph = st.empty()
    with m_col3:
        peak_count_ph = st.empty()

    video_placeholder = st.empty()
    st.markdown("---")
    line_col, heat_col = st.columns(2)

    with line_col:
        st.markdown("#### 📈 Occupancy Trend")
        line_chart_ph = st.empty()
    with heat_col:
        st.markdown("#### 🔥 Density Heatmap")
        heatmap_ph = st.empty()

    stop_button = st.sidebar.button("Stop Analysis", type="primary")

    cap = cv2.VideoCapture(0 if is_webcam else video_source)
    unique_ids = set()
    peak_count = 0
    recent_centroids = deque(maxlen=500)
    count_history = deque(maxlen=200)
    last_firebase_update = time.time()
    alert_was_triggered = False

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, conf=0.4, classes=[0], verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])
            detections.append(([x1, y1, w, h], conf, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)
        current_live_count = 0

        for track in tracks:
            if not track.is_confirmed(): continue
            current_live_count += 1
            unique_ids.add(track.track_id)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track.track_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            recent_centroids.append((int((x1+x2)/2), int((y1+y2)/2)))

        if current_live_count > crowd_limit:
            alert_was_triggered = True
            alert_placeholder.markdown(
                f"<div class='overcrowded-alert'>⚠️ OVERCROWDED DETECTED: {current_live_count} / {crowd_limit}</div>", 
                unsafe_allow_html=True
            )
            cv2.putText(frame, "!!! OVERCROWDED !!!", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
        else:
            alert_placeholder.empty()

        if current_live_count > peak_count: peak_count = current_live_count
        count_history.append(current_live_count)

        live_count_ph.markdown(f"<div class='metric-card'><div class='metric-label'>Live Count</div><div class='metric-value'>{current_live_count}</div></div>", unsafe_allow_html=True)
        total_unique_ph.markdown(f"<div class='metric-card'><div class='metric-label'>Total People</div><div class='metric-value'>{len(unique_ids)}</div></div>", unsafe_allow_html=True)
        peak_count_ph.markdown(f"<div class='metric-card'><div class='metric-label'>Peak Detected</div><div class='metric-value'>{peak_count}</div></div>", unsafe_allow_html=True)

        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        if len(count_history) > 1:
            line_chart_ph.line_chart(pd.DataFrame({"People": list(count_history)}))
        if len(recent_centroids) > 0:
            heatmap_ph.image(generate_heatmap(recent_centroids, frame.shape), use_container_width=True)

        if time.time() - last_firebase_update > 2.0:
            update_live_firestore(current_live_count, len(unique_ids))
            last_firebase_update = time.time()

    cap.release()
    if st.session_state.user_role == 'user':
        log_analysis(peak_count, len(unique_ids), alert_was_triggered)
    st.success("Session Ended.")

# ------------- LOGIN & NAVIGATION -------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    st.sidebar.title(f"Welcome, {st.session_state.user_name}")
    if st.sidebar.button("Log Out"):
        st.session_state.clear()
        st.rerun()

    if st.session_state.user_role == "user":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Safety Parameters")
        manual_limit = st.sidebar.number_input("Set Person Limit", min_value=1, max_value=1000, value=15)
        mode = st.sidebar.radio("Select Input Source", ["Webcam", "Video File"])
        if mode == "Video File":
            f = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
            if f and st.button("Run AI Analysis"):
                t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                t.write(f.read())
                process_video(t.name, manual_limit)
        else:
            if st.button("Launch Webcam Feed"):
                process_video(0, manual_limit, True)
    else:
        st.title("Admin Activity Logs")
        logs = db.collection("logs").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(25).stream()
        df_logs = pd.DataFrame([l.to_dict() for l in logs])
        if not df_logs.empty:
            st.dataframe(df_logs, use_container_width=True)
else:
    st.title("AI Crowd Management System")
    c1, c2 = st.columns(2)
    with c1:
        with st.form("u_login"):
            st.subheader("User Portal")
            u_e = st.text_input("Email")
            u_p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                row = check_credentials(u_e, u_p, "user")
                if row:
                    st.session_state.update({"logged_in":True, "user_role":"user", "user_name":row.get("name"), "user_email":u_e})
                    st.rerun()
    with c2:
        with st.form("a_login"):
            st.subheader("Admin Portal")
            a_e = st.text_input("Admin Email")
            a_p = st.text_input("Admin Password", type="password")
            if st.form_submit_button("Admin Login"):
                row = check_credentials(a_e, a_p, "admin")
                if row:
                    st.session_state.update({"logged_in":True, "user_role":"admin", "user_name":row.get("name"), "user_email":a_e})
                    st.rerun()

            
