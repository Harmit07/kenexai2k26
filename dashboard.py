import base64
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pyrebase
import joblib
from tensorflow.keras.models import load_model
import google.generativeai as genai

# ==========================================
# 1. PAGE CONFIG & API CONFIGURATIONS
# ==========================================
st.set_page_config(
    page_title="Academic & Career Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ⚠️ PASTE YOUR GEMINI API KEY HERE
genai.configure(api_key="AIzaSyCc8yr3bRQGPf5B_OTXwP89VXDLOQRHzxM")

# ⚠️ PASTE YOUR FIREBASE KEYS HERE
firebaseConfig = {
  "apiKey": "AIzaSyALR4un3E2MF9K_A4DqqR34y87r1w7cnxA",
  "authDomain": "practical18-a2f95.firebaseapp.com",
  "databaseURL": "https://practical18-a2f95-default-rtdb.firebaseio.com",
  "projectId": "practical18-a2f95",
  "storageBucket": "practical18-a2f95.firebasestorage.app",
  "messagingSenderId": "631969441897",
  "appId": "1:631969441897:web:d8c7b6493bdacc64f6daf8",
  "measurementId": "G-FB3BHGFFB5"
}

try:
    firebase = pyrebase.initialize_app(firebaseConfig)
    auth = firebase.auth()
    db = firebase.database()
except Exception as e:
    st.error(f"Firebase configuration error. Check your credentials. {e}")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.access_val = None

# ==========================================
# 2. UTILITY, DATA & ML FUNCTIONS
# ==========================================
@st.cache_data
def get_base64_of_bin_file(bin_file: str) -> str:
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

@st.cache_resource
def load_ml_assets():
    """Loads Keras models and their Scikit-Learn scalers into memory once."""
    sgpa_model, sgpa_scaler = None, None
    risk_model, risk_scaler = None, None
    stress_model, stress_scaler = None, None
    
    try:
        if os.path.exists("sgpa5_prediction_model.h5"):
            sgpa_model = load_model("sgpa5_prediction_model.h5", compile=False)
        if os.path.exists("scaler.pkl"):
            sgpa_scaler = joblib.load("scaler.pkl")
            
        if os.path.exists("student_risk_detector_model.h5"):
            risk_model = load_model("student_risk_detector_model.h5", compile=False)
        if os.path.exists("risk_scaler.pkl"):
            risk_scaler = joblib.load("risk_scaler.pkl")
            
        if os.path.exists("student_stress_level_model.h5"):
            stress_model = load_model("student_stress_level_model.h5", compile=False)
        if os.path.exists("stress_scaler.pkl"):
            stress_scaler = joblib.load("stress_scaler.pkl")
            
    except Exception as e:
        print(f"Error loading ML assets: {e}")
        
    return sgpa_model, sgpa_scaler, risk_model, risk_scaler, stress_model, stress_scaler

# Load the AI assets on startup
ml_sgpa_model, ml_sgpa_scaler, ml_risk_model, ml_risk_scaler, ml_stress_model, ml_stress_scaler = load_ml_assets()

def get_db_connection():
    db_url = "postgresql://postgres:0000@localhost:5433/student_analytics_db"
    return st.connection("postgresql", type="sql", url=db_url)

def load_data(role: str, access_val) -> pd.DataFrame:
    conn = get_db_connection()
    
    if role == "student":
        safe_id = int(access_val)
        query = f"""
        SELECT f.*, c.career_preference, c.part_timejob, a.attendance_category
        FROM fact_student_performance f
        JOIN dim_career c ON f.career_key = c.career_key
        JOIN dim_attendance a ON f.attendance_key = a.attendance_key
        WHERE f.id = {safe_id};
        """
    else:
        query = """
        SELECT f.*, c.career_preference, c.part_timejob, a.attendance_category
        FROM fact_student_performance f
        JOIN dim_career c ON f.career_key = c.career_key
        JOIN dim_attendance a ON f.attendance_key = a.attendance_key;
        """
        
    df = conn.query(query, ttl=60)
    df = df.loc[:,~df.columns.duplicated()].copy()
    return df

def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def kpi_card(title: str, value: str, caption: str = "", tone: str = "teal") -> None:
    st.markdown(
        f"""
        <div class="kpi-card {tone}">
            <div class="kpi-header">
                <span class="kpi-title">{title}</span>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def apply_ai_risk_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    backlog_component = (out["active_backlogs"] * 20 + out["total_backlogs"] * 6).clip(0, 40)
    attendance_component = ((75 - out["attendance"]).clip(lower=0) * 0.9).clip(0, 20)
    cgpa_component = ((6.5 - out["cgpa"]).clip(lower=0) * 7).clip(0, 20)
    study_component = np.where(out["studytime"] < 2, 10, np.where(out["studytime"] < 4, 5, 0))
    out["risk_score"] = (backlog_component + attendance_component + cgpa_component + study_component).clip(0, 100)
    out["risk_band"] = pd.cut(out["risk_score"], bins=[-1, 29, 54, 100], labels=["Low Risk", "Medium Risk", "High Risk"]).astype(str)
    
    if ml_risk_model and ml_risk_scaler:
        try:
            features = pd.DataFrame()
            for col in ['sgpa1', 'sgpa2', 'sgpa3', 'sgpa4', 'cgpa', 'active_backlogs', 'total_backlogs', '10th_mark', '12th_mark', 'studytime', 'math', 'dsa', 'networks', 'os', 'attendance']:
                features[col] = out[col]
            features['career_preference_higher study'] = (out['career_preference'].str.lower() == 'higher study').astype(int)
            features['career_preference_job'] = (out['career_preference'].str.lower() == 'job').astype(int)
            features['career_preference_none'] = (out['career_preference'].str.lower() == 'none').astype(int)
            features['part_timejob_no'] = (out['part_timejob'].str.lower() == 'no').astype(int)
            features['part_timejob_yes'] = (out['part_timejob'].str.lower() == 'yes').astype(int)
            
            scaled_features = ml_risk_scaler.transform(features)
            predictions = ml_risk_model.predict(scaled_features, verbose=0)
            out['risk_prediction_raw'] = predictions.flatten()
            conditions = [(predictions > 0.7), (predictions >= 0.4) & (predictions <= 0.7), (predictions < 0.4)]
            choices = ["High Risk", "Medium Risk", "Low Risk"]
            out['risk_band'] = np.select(conditions, choices, default="Unknown")
        except Exception as e:
            st.warning(f"AI Risk Model execution failed (using fallback rules): {e}")
    return out

def apply_ai_stress_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stress_score_static"] = ((10 - out["cgpa"].fillna(0)) * 2 + out["active_backlogs"].fillna(0) * 3 + out["total_backlogs"].fillna(0) + (5 - out["studytime"].fillna(0)))
    conditions = [(out["stress_score_static"] < 5), (out["stress_score_static"] >= 5) & (out["stress_score_static"] < 10), (out["stress_score_static"] >= 10)]
    choices = ["Low Stress", "Medium Stress", "High Stress"]
    out['stress_band'] = np.select(conditions, choices, default="High Stress")
    
    if ml_stress_model and ml_stress_scaler:
        try:
            features_for_stress = pd.DataFrame()
            features_for_stress['id'] = out['id']
            features_for_stress['sgpa1'] = out['sgpa1']
            features_for_stress['sgpa2'] = out['sgpa2']
            features_for_stress['sgpa3'] = out['sgpa3']
            features_for_stress['sgpa4'] = out['sgpa4']
            features_for_stress['cgpa'] = out['cgpa']
            features_for_stress['active backlogs'] = out['active_backlogs']
            features_for_stress['total backlogs'] = out['total_backlogs']
            features_for_stress['10th mark'] = out['10th_mark']
            features_for_stress['12th mark'] = out['12th_mark']
            features_for_stress['studytime'] = out['studytime']
            features_for_stress['math'] = out['math']
            features_for_stress['dsa'] = out['dsa']
            features_for_stress['networks'] = out['networks']
            features_for_stress['os'] = out['os']
            features_for_stress['attendance'] = out['attendance']
            features_for_stress['stress_score'] = out['stress_score_static']
            features_for_stress['part timejob_yes'] = (out['part_timejob'].str.lower() == 'yes').astype(int)
            features_for_stress['career preference_job'] = (out['career_preference'].str.lower() == 'job').astype(int)
            features_for_stress['career preference_none'] = (out['career_preference'].str.lower() == 'none').astype(int)
            
            scaled_features = ml_stress_scaler.transform(features_for_stress)
            predictions = ml_stress_model.predict(scaled_features, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            
            ai_conditions = [(pred_classes == 0), (pred_classes == 1), (pred_classes == 2)]
            ai_choices = ["High Stress", "Low Stress", "Medium Stress"]
            out['stress_band'] = np.select(ai_conditions, ai_choices, default="Unknown")
        except Exception as e:
            st.warning(f"AI Stress Model execution failed (using fallback rules): {e}")
    return out

# ==========================================
# 3. GENERATIVE AI AGENT FUNCTIONS
# ==========================================
def generate_system_context(df: pd.DataFrame, role: str) -> str:
    """Dynamically generates the prompt context based on the logged-in user's data."""
    if role == "student":
        student = df.iloc[0]
        context = f"""
        You are an expert academic and career counseling AI agent. 
        You are speaking directly to a student. Here is their live data:
        - CGPA: {student.get('cgpa', 'N/A')}
        - Active Backlogs: {student.get('active_backlogs', 0)}
        - Attendance: {student.get('attendance', 'N/A')}%
        - Daily Study Time: {student.get('studytime', 0)} hours
        - Career Preference: {student.get('career_preference', 'N/A')}
        - AI Risk Level: {student.get('risk_band', 'N/A')}
        - AI Stress Level: {student.get('stress_band', 'N/A')}
        - Readiness Band: {student.get('career_readiness_band', 'N/A')}
        - Subject Marks -> Math: {student.get('math', 0)}, DSA: {student.get('dsa', 0)}, Networks: {student.get('networks', 0)}, OS: {student.get('os', 0)}
        
        Your Goal: Answer the student's questions based ONLY on this context. Be encouraging but realistic. 
        If they ask for job suggestions, look at their marks (e.g., high DSA -> SDE roles, high Networks -> DevOps/Cloud).
        If they are High Risk or High Stress, suggest actionable, gentle steps to recover. Format with markdown.
        """
        return context
        
    elif role == "teacher":
        context = f"""
        You are an expert teaching assistant AI. You are helping a faculty member analyze their current class cohort.
        Here is the aggregate data for the filtered students currently visible in their dashboard:
        - Total Students in scope: {len(df)}
        - Average CGPA: {df['cgpa'].mean() if not df.empty else 0:.2f}
        - High Risk Students: {len(df[df['risk_band'] == 'High Risk'])}
        - High Stress Students: {len(df[df['stress_band'] == 'High Stress'])}
        - Average Attendance: {df['attendance'].mean() if not df.empty else 0:.2f}%
        
        Your Goal: Help the teacher identify trends, suggest intervention strategies for at-risk students, and summarize the class performance based on the data provided.
        """
        return context
    return "You are a helpful AI assistant."

def ai_assistant_ui(df: pd.DataFrame, role: str):
    """Renders the Streamlit Chat UI and handles LLM communication."""
    st.markdown("<div class='section-title'>🤖 Personalized AI Assistant</div>", unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        greeting = "Hi! I'm your personalized AI guide. Ask me about your career paths, how to improve your SGPA, or study strategies based on your actual data!" if role == "student" else "Hello Professor. Ask me to analyze the current cohort, suggest interventions, or summarize performance gaps."
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask your AI assistant..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate Context and Call AI
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    system_context = generate_system_context(df, role)
                    full_prompt = f"{system_context}\n\nUser Question: {prompt}"
                    
                    response = model.generate_content(full_prompt)
                    reply = response.text
                    
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"AI Error (Ensure your Gemini API Key is valid): {e}")

# ==========================================
# 4. SECURE AUTHENTICATION UI
# ==========================================
def auth_screen():
    st.markdown("""
    <style>
    .auth-container { max-width: 450px; margin: 80px auto; padding: 40px; border-radius: 12px; border: 1px solid var(--secondary-background-color); box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>🔐 Academic Portal</h2>", unsafe_allow_html=True)
    
    tab_login, tab_signup, tab_admin = st.tabs(["Log In", "Create Account", "Admin"])
    
    with tab_login:
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        if st.button("Log In", use_container_width=True, type="primary"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                profile = db.child("users").child(user['localId']).get().val()
                if profile:
                    st.session_state.logged_in = True
                    st.session_state.user_role = profile['role']
                    st.session_state.access_val = profile.get('access_id') if profile['role'] == 'student' else profile.get('subject')
                    st.rerun()
                else:
                    st.error("Profile not found in Realtime Database.")
            except:
                st.error("Invalid Email or Password.")
                
    with tab_signup:
        reg_email = st.text_input("New Email", key="reg_email")
        reg_password = st.text_input("New Password", type="password", key="reg_pw")
        reg_role = st.selectbox("I am a...", ["Teacher", "Student"])
        
        specific_id = None
        if reg_role == "Student":
            specific_id = st.number_input("Enter your Student ID (e.g., 1)", min_value=1, step=1)
        else:
            specific_id = st.selectbox("Assign your Subject", ["math", "dsa", "networks", "os"])
        if st.button("Register Account", use_container_width=True):
            try:
                user = auth.create_user_with_email_and_password(reg_email, reg_password)
                data = {"role": reg_role.lower(), "email": reg_email}
                if reg_role == "Student": data["access_id"] = int(specific_id)
                else: data["subject"] = specific_id
                
                db.child("users").child(user['localId']).set(data)
                st.success("Account Created! You can now log in above.")
            except Exception as e:
                st.error(f"Registration failed. Email might already exist.")
                
    with tab_admin:
        st.info("System Administration Login")
        admin_email = st.text_input("Admin Email", key="admin_email")
        admin_password = st.text_input("Admin Password", type="password", key="admin_pw")
        if st.button("Admin Log In", use_container_width=True, type="primary"):
            # HARDCODED ADMIN CREDENTIALS
            if admin_email == "admin@portal.com" and admin_password == "admin123":
                st.session_state.logged_in = True
                st.session_state.user_role = "admin"
                st.session_state.access_val = "ALL"
                st.rerun()
            else:
                st.error("Invalid Admin Credentials.")
                
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 5. MAIN DASHBOARD LOGIC
# ==========================================
def main() -> None:
    font_regular_path = "FK-Grotesk-Font-Family 2/FKGroteskTrial-Regular.otf"
    font_bold_path = "FK-Grotesk-Font-Family 2/FKGroteskTrial-Bold.otf"
    font_face_css = ""
    if Path(font_regular_path).exists() and Path(font_bold_path).exists():
        font_regular_b64 = get_base64_of_bin_file(font_regular_path)
        font_bold_b64 = get_base64_of_bin_file(font_bold_path)
        font_face_css = f"""
        @font-face {{ font-family: 'FK Grotesk'; src: url("data:font/otf;base64,{font_regular_b64}") format("opentype"); font-weight: normal; font-style: normal; }}
        @font-face {{ font-family: 'FK Grotesk'; src: url("data:font/otf;base64,{font_bold_b64}") format("opentype"); font-weight: bold; font-style: normal; }}
        html, body, [class*="css"] {{ font-family: 'FK Grotesk', sans-serif !important; }}
        """
    else:
        font_face_css = "html, body, [class*=\"css\"] { font-family: sans-serif; }"
    st.markdown(f"""
        <style>
        {font_face_css}
        .hero {{ background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%); color: white; border-radius: 12px; padding: 24px 30px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .hero h2 {{margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: bold; letter-spacing: -0.5px;}}
        .hero p {{margin: 0; opacity: 0.95; font-size: 1.05rem; font-weight: normal;}}
        .kpi-card {{ border-radius: 12px; padding: 20px; background-color: var(--background-color); border: 1px solid var(--secondary-background-color); min-height: 120px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: all 0.2s ease; }}
        .kpi-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1); }}
        .kpi-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
        .kpi-title {{ font-size: 0.85rem; color: gray; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; }}
        .kpi-value {{ font-size: 2.2rem; font-weight: bold; line-height: 1.1; margin-bottom: 8px; }}
        .kpi-caption {{ color: gray; font-size: 0.8rem; font-weight: normal; }}
        .kpi-card.teal {{ border-left: 4px solid #14b8a6; }} .kpi-card.blue {{ border-left: 4px solid #3b82f6; }}
        .kpi-card.amber {{ border-left: 4px solid #f59e0b; }} .kpi-card.rose {{ border-left: 4px solid #f43f5e; }}
        .kpi-card.emerald {{ border-left: 4px solid #10b981; }} .kpi-card.purple {{ border-left: 4px solid #8b5cf6; }}
        .section-title {{ margin: 1.5rem 0 1rem 0; font-weight: bold; font-size: 1.4rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--secondary-background-color); }}
        .chart-title {{ font-size: 1.15rem; font-weight: 600; margin-bottom: 10px; }}
        </style>
        """, unsafe_allow_html=True)
    
    # GATEKEEPER CHECK
    if not st.session_state.logged_in:
        auth_screen()
        return
        
    # SIDEBAR PROFILE & LOGOUT
    with st.sidebar:
        st.markdown(f"### 👤 Profile")
        st.write(f"Role: **{st.session_state.user_role.title()}**")
        if st.session_state.user_role == "student":
            st.info(f"Student ID: **{st.session_state.access_val}**")
        elif st.session_state.user_role == "teacher":
            st.info(f"Subject Assigned: **{st.session_state.access_val.upper()}**")
        elif st.session_state.user_role == "admin":
            st.info("System Administrator")
            
        if st.button("Log Out", use_container_width=True):
            st.session_state.messages = [] # Clear Chat History on Logout
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.session_state.access_val = None
            st.rerun()
        st.divider()

    # ==========================================
    # ADMIN VIEW
    # ==========================================
    if st.session_state.user_role == "admin":
        st.markdown("<div class='hero'><h2>System Administration Platform</h2><p>Manage student records directly into the PostgreSQL Database.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Add New Student Record</div>", unsafe_allow_html=True)
        
        with st.form("add_student_form"):
            st.write("Enter the academic and personal details for the new student.")
            c1, c2, c3 = st.columns(3)
            with c1:
                new_id = st.number_input("Student ID (Unique)", min_value=1, step=1)
                sgpa1 = st.number_input("SGPA Sem 1", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
                sgpa2 = st.number_input("SGPA Sem 2", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
                sgpa3 = st.number_input("SGPA Sem 3", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
                sgpa4 = st.number_input("SGPA Sem 4", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
                cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            with c2:
                mark_10 = st.number_input("10th Marks (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
                mark_12 = st.number_input("12th Marks (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
                active_back = st.number_input("Active Backlogs", min_value=0, step=1)
                total_back = st.number_input("Total Backlogs (History)", min_value=0, step=1)
                attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
                study_time = st.number_input("Study Time (hrs/day)", min_value=0.0, max_value=24.0, value=0.0, step=0.5)
            with c3:
                math_mark = st.number_input("Math Marks", min_value=0, max_value=100, value=0)
                dsa_mark = st.number_input("DSA Marks", min_value=0, max_value=100, value=0)
                net_mark = st.number_input("Networks Marks", min_value=0, max_value=100, value=0)
                os_mark = st.number_input("OS Marks", min_value=0, max_value=100, value=0)
                
                st.info("Career & Attendance mapped to default keys (1) for this insert.")
                
            submitted = st.form_submit_button("Insert Student to PostgreSQL")
            if submitted:
                conn = get_db_connection()
                check_query = f"SELECT id FROM fact_student_performance WHERE id = {new_id};"
                try:
                    existing = conn.query(check_query)
                    if not existing.empty:
                        st.error(f"Error: Student ID {new_id} already exists in the database.")
                    else:
                        from sqlalchemy import text
                        engine = conn.engine
                        with engine.begin() as db_conn:
                            insert_sql = text("""
                                INSERT INTO fact_student_performance 
                                (id, sgpa1, sgpa2, sgpa3, sgpa4, cgpa, active_backlogs, total_backlogs, 
                                "10th_mark", "12th_mark", studytime, math, dsa, networks, os, attendance, 
                                career_key, attendance_key) 
                                VALUES (:id, :s1, :s2, :s3, :s4, :cg, :ab, :tb, :m10, :m12, :st, :math, :dsa, :net, :os, :att, 1, 1)
                            """)
                            db_conn.execute(insert_sql, {
                                "id": new_id, "s1": sgpa1, "s2": sgpa2, "s3": sgpa3, "s4": sgpa4, "cg": cgpa,
                                "ab": active_back, "tb": total_back, "m10": mark_10, "m12": mark_12, "st": study_time,
                                "math": math_mark, "dsa": dsa_mark, "net": net_mark, "os": os_mark, "att": attendance
                            })
                        st.success(f"Student ID {new_id} successfully added to PostgreSQL database.")
                except Exception as e:
                    st.error(f"Database insertion failed: {e}")
        return # Admin doesn't need to see the rest of the dashboards

    # ==========================================
    # DATA LOADING (For Student and Teacher)
    # ==========================================
    try:
        df = load_data(st.session_state.user_role, st.session_state.access_val)
    except Exception as exc:
        st.error(f"Could not connect to PostgreSQL or fetch data: {exc}")
        st.stop()
    
    numeric_cols = ["sgpa1", "sgpa2", "sgpa3", "sgpa4", "cgpa", "active_backlogs", "total_backlogs", "10th_mark", "12th_mark", "studytime", "math", "dsa", "networks", "os", "attendance"]
    df = safe_numeric(df, numeric_cols)
    for cat_col in ["part_timejob", "career_preference"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["id", "cgpa", "attendance"]).copy()
    
    # APPLY AI PREDICTION MODELS (Risk + Stress)
    df = apply_ai_risk_model(df)
    df = apply_ai_stress_model(df)
    
    # Additional baseline computations
    subj_avg = df[["math", "dsa", "networks", "os"]].mean(axis=1)
    readiness = (df["cgpa"] * 8 + (df["attendance"] / 100) * 10 + (subj_avg / 100) * 10 + (np.minimum(df["studytime"], 12) / 12) * 5 - (df["active_backlogs"] * 4 + df["total_backlogs"] * 2))
    df["career_readiness_score"] = readiness.clip(0, 100)
    df["career_readiness_band"] = pd.cut(df["career_readiness_score"], bins=[-1, 44, 69, 100], labels=["Needs Support", "Developing", "Ready"]).astype(str)
    df["subject_avg"] = subj_avg
    df["sgpa_avg"] = df[["sgpa1", "sgpa2", "sgpa3", "sgpa4"]].mean(axis=1)
    df["sgpa_consistency"] = df[["sgpa1", "sgpa2", "sgpa3", "sgpa4"]].std(axis=1)
    
    st.markdown("<div class='hero'><h2>Student Success & Readiness Platform</h2><p>Integrated analytics for learning gaps, risk detection, stress prediction, and career readiness insights.</p></div>", unsafe_allow_html=True)
    
    # ==========================================
    # STUDENT VIEW WITH AI PREDICTIONS
    # ==========================================
    if st.session_state.user_role == "student":
        if df.empty:
            st.error("No PostgreSQL data found for your Student ID.")
            st.stop()
            
        student_row = df.iloc[0]
        
        # --- SGPA AI PREDICTION ---
        predicted_sgpa = None
        if ml_sgpa_model and ml_sgpa_scaler:
            try:
                avg_sgpa_4 = (student_row['sgpa1'] + student_row['sgpa2'] + student_row['sgpa3'] + student_row['sgpa4']) / 4
                sgpa_trend = student_row['sgpa4'] - student_row['sgpa1']
                recent_trend = student_row['sgpa4'] - student_row['sgpa3']
                academic_score = (student_row['math'] + student_row['dsa'] + student_row['networks'] + student_row['os']) / 4
                job_map = {"yes": 1, "no": 0}
                career_map = {"higher study": 0, "job": 1, "none": 2}
                
                features_df = pd.DataFrame([{
                    'sgpa1': student_row['sgpa1'], 'sgpa2': student_row['sgpa2'],
                    'sgpa3': student_row['sgpa3'], 'sgpa4': student_row['sgpa4'],
                    'cgpa': student_row['cgpa'], 'active_backlogs': student_row['active_backlogs'],
                    'total_backlogs': student_row['total_backlogs'], '10th_mark': student_row['10th_mark'],
                    '12th_mark': student_row['12th_mark'], 'studytime': student_row['studytime'],
                    'part_timejob': job_map.get(str(student_row['part_timejob']).strip().lower(), 0),
                    'math': student_row['math'], 'dsa': student_row['dsa'],
                    'networks': student_row['networks'], 'os': student_row['os'],
                    'attendance': student_row['attendance'],
                    'career_preference': career_map.get(str(student_row['career_preference']).strip().lower(), 0),
                    'avg_sgpa_4': avg_sgpa_4, 'sgpa_trend': sgpa_trend,
                    'recent_trend': recent_trend, 'academic_score': academic_score
                }])
                
                scaled_f = ml_sgpa_scaler.transform(features_df)
                pred_raw = ml_sgpa_model.predict(scaled_f, verbose=0)[0][0]
                predicted_sgpa = min(max(pred_raw, 0.0), 10.0)
            except Exception as e:
                pass 
                
        st.markdown("<div class='section-title'>Personal Analytics View</div>", unsafe_allow_html=True)
        
        # Row 1 - Performance & Predictions
        s1, s2, s3, s4 = st.columns(4)
        with s1: kpi_card("CGPA", f"{round(float(student_row['cgpa']), 2)}", "cumulative", "teal")
        with s2: kpi_card("Attendance", f"{round(float(student_row['attendance']), 2)}%", "overall", "blue")
        
        risk_map = {"High Risk": "rose", "Medium Risk": "amber", "Low Risk": "emerald"}
        risk_color = risk_map.get(str(student_row["risk_band"]), "purple")
        with s3: kpi_card("Risk Level", str(student_row["risk_band"]), "AI Prediction", risk_color)
        
        stress_map = {"High Stress": "rose", "Medium Stress": "amber", "Low Stress": "emerald"}
        stress_color = stress_map.get(str(student_row["stress_band"]), "purple")
        with s4: kpi_card("Stress Level", str(student_row["stress_band"]), "AI Prediction", stress_color)
        
        st.write("")
        
        # Row 2 - Secondary Info
        s5, s6, s7 = st.columns(3)
        with s5: kpi_card("Readiness", str(student_row["career_readiness_band"]), "preparedness", "emerald")
        with s6: kpi_card("Backlogs", str(int(student_row["total_backlogs"])), "pending", "amber")
        with s7:
            if predicted_sgpa is not None:
                kpi_card("Sem 5 Forecast", f"{predicted_sgpa:.2f}", "AI Prediction", "purple")
            else:
                kpi_card("Sem 5 Forecast", "N/A", "Model missing", "purple")
                
        st.write("")
        c1, c2 = st.columns(2)
        
        with c1:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>SGPA Trend (Historical + Forecast)</div>", unsafe_allow_html=True)
                semesters = ["Sem 1", "Sem 2", "Sem 3", "Sem 4"]
                sgpas = [student_row["sgpa1"], student_row["sgpa2"], student_row["sgpa3"], student_row["sgpa4"]]
                if predicted_sgpa is not None:
                    semesters.append("Sem 5 (Est)")
                    sgpas.append(predicted_sgpa)
                sgpa_student = pd.DataFrame({"Semester": semesters, "SGPA": sgpas})
                fig_sgpa = px.line(sgpa_student, x="Semester", y="SGPA", markers=True)
                fig_sgpa.update_traces(line_color='#3b82f6', marker=dict(size=8))
                
                if predicted_sgpa is not None:
                    fig_sgpa.add_scatter(x=["Sem 5 (Est)"], y=[predicted_sgpa], mode='markers', marker=dict(color='#8b5cf6', size=12, symbol='star'), name="AI Prediction")
                    
                fig_sgpa.update_layout(height=320, margin=dict(l=10, r=20, t=10, b=10), xaxis=dict(showgrid=False), yaxis=dict(range=[0, 10], title=None))
                st.plotly_chart(fig_sgpa, use_container_width=True)
        with c2:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>Subject Performance</div>", unsafe_allow_html=True)
                subject_student = pd.DataFrame({"Subject": ["Math", "DSA", "Networks", "OS"], "Marks": [student_row["math"], student_row["dsa"], student_row["networks"], student_row["os"]]})
                avg_score = round(subject_student["Marks"].mean(), 1)
                fig_donut = px.pie(subject_student, names="Subject", values="Marks", hole=0.7, color_discrete_sequence=["#14b8a6", "#3b82f6", "#8b5cf6", "#f43f5e"])
                fig_donut.update_traces(textinfo='percent', textposition='inside', insidetextorientation='horizontal')
                fig_donut.update_layout(height=320, margin=dict(t=10, b=20, l=10, r=10), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5), annotations=[dict(text=f"{avg_score}%<br>Avg", x=0.5, y=0.5, font_size=18, showarrow=False)])
                st.plotly_chart(fig_donut, use_container_width=True)
                
        with st.expander("View Personal Summary Table"):
            personal_summary = pd.DataFrame({
                "Metric": ["Study Time (hrs/day)", "Career Preference", "Part-time Job", "Readiness Score", "AI Risk Level", "AI Stress Level", "10th Mark", "12th Mark"],
                "Value": [round(float(student_row["studytime"]), 2), student_row["career_preference"].title(), student_row["part_timejob"].title(), round(float(student_row["career_readiness_score"]), 2), student_row["risk_band"], student_row["stress_band"], round(float(student_row["10th_mark"]), 2), round(float(student_row["12th_mark"]), 2)]
            })
            personal_summary["Value"] = personal_summary["Value"].astype(str)
            st.dataframe(personal_summary, width="stretch", hide_index=True)
            
        st.divider()
        # AI ASSISTANT INJECTION (STUDENT)
        ai_assistant_ui(df, st.session_state.user_role)
    
    # ==========================================
    # FACULTY VIEW
    # ==========================================
    elif st.session_state.user_role == "teacher":
        subject_assigned = st.session_state.access_val.lower()
        
        # Teacher Sidebar Filters
        st.sidebar.title("Data Filters")
        career_options = sorted(df["career_preference"].dropna().unique().tolist())
        job_options = sorted(df["part_timejob"].dropna().unique().tolist())
        
        available_risk_bands = sorted(df["risk_band"].dropna().unique().tolist())
        available_stress_bands = sorted(df["stress_band"].dropna().unique().tolist())
        
        selected_career = st.sidebar.multiselect("Career Preference", career_options, default=career_options)
        selected_job = st.sidebar.multiselect("Part-time Job", job_options, default=job_options)
        selected_risk = st.sidebar.multiselect("Risk Level", available_risk_bands, default=available_risk_bands)
        selected_stress = st.sidebar.multiselect("Stress Level", available_stress_bands, default=available_stress_bands)
        
        cgpa_min, cgpa_max = float(df["cgpa"].min()), float(df["cgpa"].max())
        att_min, att_max = float(df["attendance"].min()), float(df["attendance"].max())
        cgpa_range = st.sidebar.slider("CGPA Range", cgpa_min, cgpa_max, (cgpa_min, cgpa_max), step=0.1)
        attendance_range = st.sidebar.slider("Attendance (%)", att_min, att_max, (att_min, att_max), step=0.5)
        
        filtered = df[
            df["career_preference"].isin(selected_career) & df["part_timejob"].isin(selected_job) &
            df["risk_band"].isin(selected_risk) & df["stress_band"].isin(selected_stress) &
            df["cgpa"].between(cgpa_range[0], cgpa_range[1]) &
            df["attendance"].between(attendance_range[0], attendance_range[1])
        ].copy()
        
        if filtered.empty:
            st.warning("No rows match current filters.")
            st.stop()
            
        st.markdown(f"<div class='section-title'>Faculty & Mentor Analytics View ({subject_assigned.upper()})</div>", unsafe_allow_html=True)
        
        main_k1, main_k2, main_k3, main_k4 = st.columns(4)
        with main_k1: kpi_card("Students in Scope", str(int(filtered["id"].nunique())), "filtered cohort", "blue")
        with main_k2: kpi_card("Average CGPA", f"{round(float(filtered['cgpa'].mean()), 2)}", "overall academic level", "teal")
        high_risk_count = int((filtered["risk_band"] == "High Risk").sum())
        with main_k3: kpi_card("High Risk Students", str(high_risk_count), "requires intervention", "rose")
        with main_k4: kpi_card(f"Avg {subject_assigned.upper()}", f"{round(float(filtered[subject_assigned].mean()), 1)}", "your subject avg", "emerald")
        
        st.write("")
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1: kpi_card("Avg Attendance", f"{round(float(filtered['attendance'].mean()), 2)}%", "cohort attendance", "blue")
        with f2: kpi_card("Backlog Incidence", f"{round(float((filtered['active_backlogs'] > 0).mean() * 100), 2)}%", "active backlog %", "amber")
        
        high_stress_count = int((filtered["stress_band"] == "High Stress").sum())
        with f3: kpi_card("High Stress", str(high_stress_count), "well-being concern", "rose")
        
        with f4: kpi_card("Needs Support", str(int((filtered["career_readiness_band"] == "Needs Support").sum())), "readiness band", "rose")
        with f5: kpi_card("Global Subj Avg", f"{round(float(filtered[['math', 'dsa', 'networks', 'os']].mean(axis=1).mean()), 2)}", "across core subjects", "teal")
        
        st.write("")
        g1, g2, g3 = st.columns(3)
        
        with g1:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>Cohort SGPA Trend</div>", unsafe_allow_html=True)
                sgpa_trend = pd.DataFrame({"Semester": ["Sem 1", "Sem 2", "Sem 3", "Sem 4"], "Avg SGPA": [filtered["sgpa1"].mean(), filtered["sgpa2"].mean(), filtered["sgpa3"].mean(), filtered["sgpa4"].mean()]})
                fig_cohort_sgpa = px.line(sgpa_trend, x="Semester", y="Avg SGPA", markers=True)
                fig_cohort_sgpa.update_traces(line_color='#8b5cf6', marker=dict(size=8))
                fig_cohort_sgpa.update_layout(height=320, margin=dict(l=10, r=20, t=10, b=10), xaxis=dict(showgrid=False), yaxis=dict(range=[0, 10], title=None))
                st.plotly_chart(fig_cohort_sgpa, use_container_width=True)
        with g2:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>AI Risk Level Distribution</div>", unsafe_allow_html=True)
                risk_dist = filtered["risk_band"].value_counts().rename_axis("Risk Level").reset_index(name="Students")
                color_mapping = {"High Risk": "#f43f5e", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"}
                fig_risk = px.bar(risk_dist, x="Risk Level", y="Students", color="Risk Level", color_discrete_map=color_mapping)
                fig_risk.update_layout(height=320, margin=dict(l=10, r=20, t=10, b=10), showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(title=None))
                st.plotly_chart(fig_risk, use_container_width=True)
        with g3:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>AI Stress Level Distribution</div>", unsafe_allow_html=True)
                stress_dist = filtered["stress_band"].value_counts().rename_axis("Stress Level").reset_index(name="Students")
                color_mapping_stress = {"High Stress": "#f43f5e", "Medium Stress": "#f59e0b", "Low Stress": "#10b981"}
                fig_stress = px.bar(stress_dist, x="Stress Level", y="Students", color="Stress Level", color_discrete_map=color_mapping_stress)
                fig_stress.update_layout(height=320, margin=dict(l=10, r=20, t=10, b=10), showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(title=None))
                st.plotly_chart(fig_stress, use_container_width=True)
                
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Subject Difficulty Index")
            difficulty = []
            for sub in ["math", "dsa", "networks", "os"]:
                avg_mark = float(filtered[sub].mean())
                fail_rate = float((filtered[sub] < 40).mean() * 100)
                score = (100 - avg_mark) * 0.7 + fail_rate * 0.3
                difficulty.append({"Subject": sub.upper(), "Avg Mark": round(avg_mark, 2), "Fail Rate (%)": round(fail_rate, 2), "Difficulty Score": round(score, 2)})
            df_diff = pd.DataFrame(difficulty).sort_values("Difficulty Score", ascending=False)
            st.dataframe(df_diff, width="stretch", hide_index=True)
        with c4:
            st.subheader("Career Readiness by Preference")
            pref_readiness = filtered.groupby("career_preference", as_index=False).agg(Avg_Readiness=("career_readiness_score", "mean"), Avg_CGPA=("cgpa", "mean"), Students=("id", "nunique")).sort_values("Avg_Readiness", ascending=False)
            pref_readiness["career_preference"] = pref_readiness["career_preference"].str.title()
            pref_readiness.rename(columns={"career_preference": "Career"}, inplace=True)
            st.dataframe(pref_readiness.round(2), width="stretch", hide_index=True)
        
        st.subheader("Priority Intervention Lists")
        
        common_cols = ["id", "cgpa", "attendance", "active_backlogs", "total_backlogs", "risk_band", "stress_band", "career_preference"]
        if "risk_prediction_raw" in filtered.columns:
            common_cols.append("risk_prediction_raw")
        
        high_risk = filtered[filtered["risk_band"] == "High Risk"].sort_values(["cgpa", "attendance"], ascending=[True, True]) \
            .loc[:, common_cols].head(15)
        high_risk = high_risk.round(2)
        with st.expander(f"🔴 High Risk Students ({len(high_risk)} students - Priority Interventions Needed)"):
            if high_risk.empty:
                st.info("No high-risk students in current filters.")
            else:
                st.dataframe(high_risk, width="stretch", hide_index=True)
                
        high_stress = filtered[filtered["stress_band"] == "High Stress"].sort_values(["cgpa", "attendance"], ascending=[True, True]) \
            .loc[:, common_cols].head(15)
        high_stress = high_stress.round(2)
        with st.expander(f"😫 High Stress Students ({len(high_stress)} students - Well-being Check Needed)"):
            if high_stress.empty:
                st.info("No high-stress students in current filters.")
            else:
                st.dataframe(high_stress, width="stretch", hide_index=True)
        
        medium_risk = filtered[filtered["risk_band"] == "Medium Risk"].sort_values(["cgpa", "attendance"], ascending=[True, True]) \
            .loc[:, common_cols].head(15)
        medium_risk = medium_risk.round(2)
        with st.expander(f"🟡 Medium Risk Students ({len(medium_risk)} students - Monitor Closely)"):
            if medium_risk.empty:
                st.info("No medium-risk students in current filters.")
            else:
                st.dataframe(medium_risk, width="stretch", hide_index=True)
        
        low_risk = filtered[filtered["risk_band"] == "Low Risk"].sort_values(["cgpa"], ascending=[False]) \
            .loc[:, common_cols].head(15)
        low_risk = low_risk.round(2)
        with st.expander(f"🟢 Low Risk Students ({len(low_risk)} students - Strong Performers)"):
            if low_risk.empty:
                st.info("No low-risk students in current filters.")
            else:
                st.dataframe(low_risk, width="stretch", hide_index=True)
        
        st.write("")
        st.divider()
        st.markdown("<div class='section-title'>Deep Data Inspection</div>", unsafe_allow_html=True)
        
        with st.expander("View Correlation Matrix"):
            corr_cols = ["cgpa", "sgpa1", "sgpa2", "sgpa3", "sgpa4", "studytime", "attendance", "active_backlogs", "total_backlogs", "math", "dsa", "networks", "os", "career_readiness_score"]
            corr = filtered[corr_cols].corr(numeric_only=True)
            st.dataframe(corr.round(2), width="stretch")
        with st.expander("View Raw Filtered Student Records"):
            view_cols = ["id", "cgpa", "sgpa_avg", "active_backlogs", "total_backlogs", "studytime", "attendance", "career_preference", "risk_band", "stress_band", "career_readiness_score", "career_readiness_band"]
            st.dataframe(filtered[view_cols].round(2), width="stretch", hide_index=True)
        st.download_button(
            label="Download Filtered Context CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="student_analytics_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()
        # AI ASSISTANT INJECTION (TEACHER)
        ai_assistant_ui(filtered, st.session_state.user_role)

if __name__ == "__main__":
    main()