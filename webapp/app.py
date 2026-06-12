import streamlit as st
import firebase_admin
from firebase_admin import credentials, db, storage
import json
import datetime
import pandas as pd
import requests
import altair as alt

# --- 1. STREAMLIT GLOBAL WORKSPACE INITIALIZATION ---
st.set_page_config(
    page_title="CatPlayground",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enforce crisp custom styling properties via direct CSS injection
st.markdown("""
    <style>
    /* 1. Make the login text input boxes smaller and sleeker */
    div[data-baseweb="input"] {
        height: 38px !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="input"] input {
        padding: 8px 12px !important;
        font-size: 14px !important;
    }
    
    /* 2. Custom Dashboard Styling */
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

# --- 2. FIREBASE REST AUTHENTICATION FUNCTIONS ---
def sign_in_with_email_and_password(email, password, api_key):
    rest_api_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    try:
        response = requests.post(rest_api_url, json=payload)
        response.raise_for_status() 
        return response.json()      
    except requests.exceptions.HTTPError as e:
        error_msg = e.response.json().get("error", {}).get("message", "Authentication Failed")
        return {"error": error_msg}

def sign_up_with_email_and_password(email, password, api_key):
    rest_api_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={api_key}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    try:
        response = requests.post(rest_api_url, json=payload)
        response.raise_for_status() 
        return response.json()      
    except requests.exceptions.HTTPError as e:
        error_msg = e.response.json().get("error", {}).get("message", "Sign Up Failed")
        return {"error": error_msg}

# Initialize session state for auth
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# =========================================================================
# --- LOGIN / SIGN UP GATEWAY ---
# =========================================================================
if not st.session_state['authenticated']:
    st.title("🐾 CajuNet Gateway")
    st.info("Log in or create a new account to view the edge tracking telemetry.")
    
    # Create neat tabs for Login and Sign Up
    tab_login, tab_signup = st.tabs(["🔐 Log In", "📝 Create Account"])
    
    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted_login = st.form_submit_button("Log In")
            
            if submitted_login:
                api_key = st.secrets["FIREBASE_WEB_API_KEY"]
                with st.spinner("Verifying..."):
                    auth_result = sign_in_with_email_and_password(email, password, api_key)
                    
                    if "error" in auth_result:
                        st.error(f"Access Denied: {auth_result['error']}")
                    else:
                        st.session_state['authenticated'] = True
                        st.session_state['user_email'] = email
                        st.success("Welcome back!")
                        st.rerun() 

    with tab_signup:
        with st.form("signup_form"):
            new_email = st.text_input("New Email")
            new_password = st.text_input("New Password (min 6 characters)", type="password")
            submitted_signup = st.form_submit_button("Create Account")
            
            if submitted_signup:
                if len(new_password) < 6:
                    st.error("Firebase requires passwords to be at least 6 characters long.")
                else:
                    api_key = st.secrets["FIREBASE_WEB_API_KEY"]
                    with st.spinner("Creating account..."):
                        signup_result = sign_up_with_email_and_password(new_email, new_password, api_key)
                        
                        if "error" in signup_result:
                            st.error(f"Error: {signup_result['error']}")
                        else:
                            st.success("Account created successfully! You can now log in using the tab above.")

# =========================================================================
# --- SUCCESSFUL LOGIN: SHOW THE DASHBOARD ---
# =========================================================================
else:
    # 3. BACKEND CLOUD SERVICE CREDENTIAL BRIDGING
    if not firebase_admin._apps:
        try:
            cred_dict = json.loads(st.secrets["FIREBASE_CREDENTIALS_JSON"])
            firebase_admin.initialize_app(credentials.Certificate(cred_dict), {
                'databaseURL': st.secrets["FIREBASE_DB_URL"],
                'storageBucket': st.secrets["FIREBASE_STORAGE_BUCKET"] 
            })
        except Exception as e:
            st.error(f"Cloud Architecture Authentication Crash: {e}")

    db_ref = db.reference("/")
    bucket = storage.bucket()

    # 4. UI SIDEBAR LAYOUT SELECTION NAVIGATION
    st.sidebar.title("🐾 CatPlayground Menu")
    st.sidebar.markdown("---")
    
    page_selection = st.sidebar.radio(
        "Navigate Workspace:", 
        ["Live Tracking Monitor", "📜 Historical Play Archive", "📊 Playtime Analytics"]
    )
    
    # Render the logout button at the bottom of the sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("Log Out of Session"):
        st.session_state['authenticated'] = False
        st.rerun()

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
        st.subheader("Past Play Sessions and Camera Highlights")
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
                        st.write(f"**Session Number:** `{session['id']}`")
                        st.write(f"**Total Session Playing Time:** {session['duration_sec']} seconds")
                                    
                    with col_photo:
                        st.markdown("#### 📷 Session Camera Frame")
                        dynamic_storage_path = f"sessions/{session['id']}.png"
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
    # RUNTIME MODULE 3: PLAYTIME ANALYTICS PAGE
    # =========================================================================
    else:
        st.title("📊 Playtime Analytics Dashboard")
        st.subheader("Historical Tracking Insight & Interaction Trends")
        st.markdown("---")
        
        if not sessions_history:
            st.warning("Insufficient historical parameters found. Let your edge device log data blocks first.")
        else:
            raw_list = [{"id": k, **v} for k, v in sessions_history.items()]
            df = pd.DataFrame(raw_list)
            
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df['Date'] = df['datetime'].dt.date
            df['Playtime (Minutes)'] = df['duration_sec'] / 60.0  
            
            import datetime as dt
            today = dt.date.today()
            df = df[df['Date'] <= today]
            
            df = df.sort_values(by='datetime')

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
            
            st.markdown("### 📈 Daily Interaction Volume (Total Minutes per Day)")
            daily_summary = df.groupby('Date')['Playtime (Minutes)'].sum().reset_index()
            daily_summary = daily_summary.set_index('Date')
            st.line_chart(daily_summary, color="#00FFCC", use_container_width=True)
            
            st.markdown("### 📊 Session Burst Comparisons")
            session_chart_data = df[['timestamp', 'Playtime (Minutes)']].set_index('timestamp')
            st.bar_chart(session_chart_data, color="#1C212D", use_container_width=True)

            st.markdown("### ⏱️ Attention Span Distribution")
            
            histogram = alt.Chart(df).mark_bar(color="#00FFCC").encode(
                alt.X("duration_sec:Q", bin=alt.Bin(maxbins=20), title="Session Length (Seconds)"),
                alt.Y("count()", title="Number of Sessions"),
                tooltip=['count()']
            ).properties(height=300)
            
            st.altair_chart(histogram, use_container_width=True)

            st.markdown("### 🕒 Active Interaction Time of Day")
            
            df['Hour'] = df['datetime'].dt.hour
            bins = [0, 6, 12, 18, 24]
            labels = ['Night (12AM-6AM)', 'Morning (6AM-12PM)', 'Afternoon (12PM-6PM)', 'Evening (6PM-12AM)']
            
            df['Time of Day'] = pd.cut(df['Hour'], bins=bins, labels=labels, right=False)
            
            pie_data = df['Time of Day'].value_counts().reset_index()
            pie_data.columns = ['Time of Day', 'Session Count']

            donut_chart = alt.Chart(pie_data).mark_arc(innerRadius=60).encode(
                theta=alt.Theta(field="Session Count", type="quantitative"),
                color=alt.Color(
                    field="Time of Day", 
                    type="nominal", 
                    scale=alt.Scale(range=["#00FFCC", "#00B38F", "#006652", "#E6FFFF"])
                ),
                tooltip=['Time of Day', 'Session Count']
            ).properties(height=350)
            
            st.altair_chart(donut_chart, use_container_width=True)