import streamlit as st
import firebase_admin
from firebase_admin import credentials, db, storage
import json
import datetime
import pandas as pd
from email_utils import send_session_summary

# --- 1. STREAMLIT GLOBAL WORKSPACE INITIALIZATION ---
st.set_page_config(
    page_title="CajuNet Core Matrix",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enforce crisp custom styling properties via direct CSS injection
st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        color: #00FFCC !important;
        font-family: monospace;
    }
    .stExpander {
        border: 1px solid #1C212D !important;
        background-color: #131722 !important;
    }
    button[data-testid="baseButton-secondary"] {
        background-color: #00FFCC !important;
        color: #0E1117 !important;
        font-weight: bold !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. BACKEND CLOUD SERVICE CREDENTIAL BRIDGING ---
if not firebase_admin._apps:
    try:
        cred_dict = json.loads(st.secrets["FIREBASE_CREDENTIALS_JSON"])
        firebase_admin.initialize_app(credentials.Certificate(cred_dict), {
            'databaseURL': st.secrets["FIREBASE_DB_URL"],
            'storageBucket': st.secrets["FIREBASE_STORAGE_BUCKET"] # Now pulling securely from secrets!
        })
    except Exception as e:
        st.error(f"Cloud Architecture Authentication Crash: {e}")

db_ref = db.reference("tracking_system")
bucket = storage.bucket()

# --- 3. UI SIDEBAR LAYOUT SELECTION NAVIGATION ---
st.sidebar.title("🐾 CajuNet System Menu")
page_selection = st.sidebar.radio(
    "Navigate Workspace:", 
    ["Live Tracking Monitor", "📜 Historical Play Archive", "📊 Playtime Analytics"]
)

# Fetch current snapshot once for historical pages
system_data = db_ref.get() or {}
sessions_history = system_data.get("sessions", {})

# =========================================================================
# RUNTIME MODULE 1: LIVE TRACKING MONITOR
# =========================================================================
if page_selection == "Live Tracking Monitor":
    st.title("🎯 Live Tracking Monitor")
    st.subheader("Real-Time Autonomous Tracking State Dashboard")
    st.markdown("---")
    
    @st.fragment(run_every=2.0)
    def render_live_telemetry_block():
        live_data = db_ref.get() or {}
        active_session = live_data.get("active_session", {"is_playing": False, "current_duration_sec": 0})
        
        col_status, col_timer = st.columns(2)
        with col_status:
            if active_session["is_playing"]:
                st.metric(label="System Target Connection Status", value="🟢 CAT TRACKING ACTIVE")
                st.success("Caju is currently playing! The ONNX model is driving the servos.")
            else:
                st.metric(label="System Target Connection Status", value="💤 EDGE SENSOR IDLE")
                st.info("The edge camera is armed and scanning for targets.")
                
        with col_timer:
            st.metric(label="Active Interaction Clock", value=f"{active_session['current_duration_sec']} Seconds")

    render_live_telemetry_block()

# =========================================================================
# RUNTIME MODULE 2: HISTORICAL PLAY ARCHIVE
# =========================================================================
elif page_selection == "📜 Historical Play Archive":
    st.title("📜 Historical Play Archive")
    st.subheader("Stored Logs, Media Frames, and Diagnostics Data")
    st.markdown("---")
    
    if not sessions_history:
        st.warning("No autonomous play sessions recorded in the cloud system directory yet.")
    else:
        sorted_sessions = [{"id": k, **v} for k, v in sessions_history.items()]
        sorted_sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        for session in sorted_sessions:
            with st.expander(f"📅 Session: {session['timestamp']} | ⏱️ Run Duration: {session['duration_sec']}s"):
                col_data, col_photo = st.columns([3, 2])
                
                with col_data:
                    st.markdown("#### 📊 Session Telemetry Metrics")
                    st.write(f"**Unique System Key Node:** `{session['id']}`")
                    st.write(f"**Total Track Processing Window:** {session['duration_sec']} seconds")
                    
                    if st.button("Dispatch Session Summary Email", key=f"action_{session['id']}", use_container_width=True):
                        manual_report_placeholder = "Autonomous edge play session successfully tracked and logged by the CajuNet hardware stack matrix."
                        with st.spinner("Transmitting encrypted summary notification report..."):
                            try:
                                send_session_summary(
                                    duration_sec=session['duration_sec'],
                                    date_str=session['timestamp'],
                                    diagnostic_text=manual_report_placeholder
                                )
                                st.toast("Email dispatched successfully to the engineering center!", icon="📨")
                            except Exception as email_err:
                                st.error(f"Failed to transmit email notification: {email_err}")
                                
                with col_photo:
                    st.markdown("#### 📷 Session Camera Ingress Frame")
                    dynamic_storage_path = f"sessions/{session['id']}.jpg"
                    try:
                        blob = bucket.blob(dynamic_storage_path)
                        if blob.exists():
                            signed_url = blob.generate_signed_url(version="v4", expiration=datetime.timedelta(minutes=10), method="GET")
                            st.image(signed_url, caption=f"Capture Target ID: {session['id']}", use_container_width=True)
                        else:
                            st.info("📷 No edge camera frame uploaded for this session yet.")
                            st.caption(f"Expected storage path: `{dynamic_storage_path}`")
                    except Exception as storage_err:
                        st.error(f"Cloud storage connection timeout: {storage_err}")

# =========================================================================
# NEW RUNTIME MODULE 3: PLAYTIME ANALYTICS PAGE
# =========================================================================
else:
    st.title("📊 Playtime Analytics Dashboard")
    st.subheader("Historical Tracking Insight & Interaction Trends")
    st.markdown("---")
    
    if not sessions_history:
        st.warning("Insufficent historical parameters found. Let your edge device log data blocks first.")
    else:
        # 1. Gather sessions data map straight into a Pandas DataFrame
        raw_list = [{"id": k, **v} for k, v in sessions_history.items()]
        df = pd.DataFrame(raw_list)
        
        # 2. Clean and convert text data strings into true Date/Time types
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['Date'] = df['datetime'].dt.date
        df['Playtime (Minutes)'] = df['duration_sec'] / 60.0  
        
        # Sort values properly matching chronological dates
        df = df.sort_values(by='datetime')

        # 3. Aggregate high-level calculated metrics row panels
        total_sessions = len(df)
        total_minutes = df['duration_sec'].sum() / 60.0
        avg_seconds = df['duration_sec'].mean()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Play Events Logged", f"{total_sessions} Sessions")
        with c2:
            st.metric("Cumulative Exercise Duration", f"{total_minutes:.1f} Mins")
        with c3:
            st.metric("Average Session Window Length", f"{avg_seconds:.0f} Seconds")
            
        st.markdown("---")
        
        # 4. Render Chart 1: Time Series Trend Line (Total playtime grouped per day)
        st.markdown("### 📈 Daily Interaction Volume (Total Minutes per Day)")
        daily_summary = df.groupby('Date')['Playtime (Minutes)'].sum().reset_index()
        daily_summary = daily_summary.set_index('Date')
        st.line_chart(daily_summary, color="#00FFCC", use_container_width=True)
        
        # 5. Render Chart 2: Individual Session Burst Comparison
        st.markdown("### 📊 Session Burst Comparisons (Length of each consecutive play event)")
        session_chart_data = df[['timestamp', 'Playtime (Minutes)']].set_index('timestamp')
        st.bar_chart(session_chart_data, color="#1C212D", use_container_width=True)