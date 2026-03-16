import base64
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pyrebase
import joblib
from tensorflow.keras.models import load_model
import google.generativeai as genai
from fpdf import FPDF
import io

# ==========================================
# 1. PAGE CONFIG & API CONFIGURATIONS
# ==========================================
st.set_page_config(
    page_title="Academic & Career Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ⚠️ PASTE YOUR GEMINI API KEY HERE
genai.configure(api_key="AIzaSyBj4DCOc1rUSSUE6owsnSSNe4CqSPd82mc")

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
        
        # Dynamically build all available fields from the student row
        data_lines = []
        for col in df.columns:
            val = student.get(col, 'N/A')
            # Format column name nicely
            label = col.replace('_', ' ').title()
            data_lines.append(f"- {label}: {val}")
        
        all_fields = "\n        ".join(data_lines)
        
        context = f"""
        You are an expert academic and career counseling AI agent. 
        You are speaking directly to a student. Here is their COMPLETE live data profile.
        You have access to EVERY field — answer any question they ask using exact values from the data below.

        --- COMPLETE STUDENT DATA ---
        {all_fields}
        
        RULES:
        - Answer ANY question the student asks about their own data. Provide exact values.
        - You can ONLY see this one student's data. You do NOT have access to other students.
        - If they ask about marks, SGPAs, attendance, backlogs, career readiness, risk, stress, or ANY other field — look it up from the data above and give the exact value.
        - If they ask for career/job suggestions, analyze their marks and strengths.
        - If they ask about improvement, be encouraging but realistic with specific actionable steps.
        - Format responses with markdown.
        """
        return context
        
    elif role == "teacher":
        # Get teacher's assigned subject dynamically
        teacher_subject = str(st.session_state.get('access_val', '')).lower()
        teacher_subject_upper = teacher_subject.upper()
        
        # All available subjects
        all_subjects = ['math', 'dsa', 'networks', 'os']
        # Other subjects the teacher does NOT have access to
        other_subjects = [s for s in all_subjects if s != teacher_subject]
        
        # Build aggregate stats dynamically
        agg_lines = [
            f"- Total Students: {len(df)}",
            f"- Average CGPA: {df['cgpa'].mean():.2f}" if not df.empty else "- Average CGPA: N/A",
            f"- Average Attendance: {df['attendance'].mean():.1f}%" if not df.empty else "- Average Attendance: N/A",
            f"- Students with Active Backlogs: {(df['active_backlogs'] > 0).sum()} ({(df['active_backlogs'] > 0).mean() * 100:.1f}%)" if not df.empty else "",
        ]
        
        # Add teacher's subject average
        if teacher_subject in df.columns and not df.empty:
            agg_lines.append(f"- Average {teacher_subject_upper} Marks: {df[teacher_subject].mean():.1f}")
            agg_lines.append(f"- Highest {teacher_subject_upper}: {df[teacher_subject].max():.1f}")
            agg_lines.append(f"- Lowest {teacher_subject_upper}: {df[teacher_subject].min():.1f}")
            fail_count = int((df[teacher_subject] < 40).sum())
            agg_lines.append(f"- Students Failing {teacher_subject_upper} (below 40): {fail_count}")
        
        # Risk/Stress breakdown
        for band_col, band_name in [('risk_band', 'Risk'), ('stress_band', 'Stress')]:
            if band_col in df.columns:
                counts = df[band_col].value_counts().to_dict()
                breakdown = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                agg_lines.append(f"- {band_name} Breakdown: {breakdown}")
        
        aggregate_stats = "\n        ".join(agg_lines)
        
        # Build row-level data — include ONLY the teacher's subject marks, NOT other subjects
        # Core columns every teacher can see
        core_cols = ['id', 'sgpa1', 'sgpa2', 'sgpa3', 'sgpa4', 'cgpa', 
                     'active_backlogs', 'total_backlogs', '10th_mark', '12th_mark',
                     'studytime', 'attendance', 'career_preference', 'part_timejob',
                     'risk_band', 'stress_band', 'career_readiness_band', 'career_readiness_score']
        
        # Add ONLY the teacher's assigned subject column
        if teacher_subject in df.columns:
            core_cols.append(teacher_subject)
        
        # Filter to available columns only
        available_cols = [c for c in core_cols if c in df.columns]
        
        # Send ALL students as CSV (no row cap) — CSV is more token-efficient and easier for AI to parse
        student_table = df[available_cols].round(2)
        student_data_csv = student_table.to_csv(index=False)
        
        context = f"""
        You are an expert teaching assistant AI for a faculty member who teaches {teacher_subject_upper}.
        
        IMPORTANT ACCESS RULES:
        - You are a {teacher_subject_upper} teacher. You can ONLY see {teacher_subject_upper} marks.
        - You do NOT have access to marks of other subjects ({', '.join(s.upper() for s in other_subjects)}). If asked about those subjects, politely say you only have access to {teacher_subject_upper} data.
        - You CAN see general student info: ID, CGPA, SGPAs, attendance, backlogs, study time, career preference, risk/stress levels, and career readiness.
        
        --- AGGREGATE CLASS STATISTICS ---
        {aggregate_stats}
        
        --- COMPLETE STUDENT DATA (CSV FORMAT) ---
        Below is the full data for ALL {len(df)} students. Each row is one student. Use this to answer ANY question.
        
{student_data_csv}
        
        HOW TO ANSWER QUESTIONS:
        - When asked about a SPECIFIC student by ID (e.g., "marks of 1358"), find the row where the 'id' column matches and return the requested values.
        - When asked to LIST or FILTER (e.g., "students with {teacher_subject} below 40"), scan the {teacher_subject} column and list matching student IDs and values.
        - When asked to COMPARE students, find both rows and present a side-by-side comparison.
        - When asked about RANKINGS (e.g., "top 5 in {teacher_subject}"), sort by the {teacher_subject} column and return results.
        - When the student ID is not found in the data, clearly say "Student ID [X] was not found in the current data."
        - Always provide EXACT values from the data, never guess or approximate.
        - Format responses with markdown tables when showing multiple students.
        """
        return context
    return "You are a helpful AI assistant."

def ai_assistant_ui(df: pd.DataFrame, role: str):
    """Renders the Streamlit Chat UI and handles LLM communication."""
    st.markdown("<div class='section-title'>Personalized AI Assistant</div>", unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        if role == "student":
            greeting = ("Hi! I'm your personal AI assistant with full access to your academic data. "
                        "You can ask me **anything** — your marks, SGPAs, attendance, backlogs, "
                        "career suggestions, study tips, risk analysis, or any other detail from your profile. "
                        "Just ask!")
        else:
            greeting = ("Hello Professor! I have access to the **complete data** of every student in your current view. "
                        "Ask me anything — a specific student's marks by ID, class comparisons, "
                        "who has the highest/lowest scores, attendance details, risk levels, or any analysis you need.")
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask anything about the data..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate Context and Call AI with full conversation history
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    system_context = generate_system_context(df, role)
                    model = genai.GenerativeModel(
                        'gemini-2.5-flash',
                        system_instruction=system_context
                    )
                    
                    # Build conversation history for multi-turn context
                    chat_history = []
                    for msg in st.session_state.messages[1:]:  # Skip the initial greeting
                        gemini_role = "user" if msg["role"] == "user" else "model"
                        chat_history.append({"role": gemini_role, "parts": [msg["content"]]})
                    
                    # Start chat with history and send current message
                    chat = model.start_chat(history=chat_history[:-1])  # All except last user msg
                    response = chat.send_message(prompt)
                    reply = response.text
                    
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"AI Error (Ensure your Gemini API Key is valid): {e}")

# ==========================================
# 3B. PDF REPORT & AI RECOMMENDATIONS
# ==========================================
def generate_student_pdf_report(student_row, predicted_sgpa=None) -> bytes:
    """Generates a styled PDF report card for a student."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Header
    pdf.set_fill_color(99, 102, 241)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_y(10)
    pdf.cell(0, 12, "Student Report Card", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "Academic & Career Analytics Platform", ln=True, align="C")
    
    pdf.ln(10)
    pdf.set_text_color(30, 30, 30)
    
    # Student Info Section
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Student Overview", ln=True)
    pdf.set_draw_color(99, 102, 241)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "", 11)
    info_items = [
        ("Student ID", str(int(student_row.get('id', 0)))),
        ("CGPA", f"{float(student_row.get('cgpa', 0)):.2f}"),
        ("Attendance", f"{float(student_row.get('attendance', 0)):.1f}%"),
        ("Career Preference", str(student_row.get('career_preference', 'N/A')).title()),
        ("Daily Study Time", f"{float(student_row.get('studytime', 0)):.1f} hours"),
        ("Part-time Job", str(student_row.get('part_timejob', 'N/A')).title()),
    ]
    for label, value in info_items:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 7, f"{label}:", align="L")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, value, ln=True)
    
    # AI Predictions Section
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "AI-Powered Analysis", ln=True)
    pdf.set_draw_color(99, 102, 241)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    risk = str(student_row.get('risk_band', 'N/A'))
    stress = str(student_row.get('stress_band', 'N/A'))
    readiness = str(student_row.get('career_readiness_band', 'N/A'))
    
    # Colored risk/stress boxes
    for label, value in [("Risk Level", risk), ("Stress Level", stress), ("Career Readiness", readiness)]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 8, f"{label}:", align="L")
        if "High" in value:
            pdf.set_text_color(220, 38, 38)
        elif "Medium" in value or "Developing" in value:
            pdf.set_text_color(217, 119, 6)
        else:
            pdf.set_text_color(22, 163, 74)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, value, ln=True)
        pdf.set_text_color(30, 30, 30)
    
    if predicted_sgpa is not None:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 8, "Predicted Sem 5 SGPA:", align="L")
        pdf.set_text_color(99, 102, 241)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, f"{predicted_sgpa:.2f}", ln=True)
        pdf.set_text_color(30, 30, 30)
    
    # SGPA History Table
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Semester-wise SGPA", ln=True)
    pdf.set_draw_color(99, 102, 241)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    semesters = ["Sem 1", "Sem 2", "Sem 3", "Sem 4"]
    sgpas = [student_row.get('sgpa1', 0), student_row.get('sgpa2', 0), student_row.get('sgpa3', 0), student_row.get('sgpa4', 0)]
    
    # Table header
    pdf.set_fill_color(99, 102, 241)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    for sem in semesters:
        pdf.cell(45, 8, sem, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", 10)
    for sgpa in sgpas:
        pdf.cell(45, 8, f"{float(sgpa):.2f}", border=1, align="C")
    pdf.ln()
    
    # Subject Marks Table
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Subject Marks", ln=True)
    pdf.set_draw_color(99, 102, 241)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    subjects = ["Math", "DSA", "Networks", "OS"]
    marks = [student_row.get('math', 0), student_row.get('dsa', 0), student_row.get('networks', 0), student_row.get('os', 0)]
    
    pdf.set_fill_color(99, 102, 241)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    for subj in subjects:
        pdf.cell(45, 8, subj, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", 10)
    for mark in marks:
        pdf.cell(45, 8, f"{float(mark):.1f}", border=1, align="C")
    pdf.ln()
    
    # Footer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "This report was auto-generated by the Academic & Career Analytics Platform.", align="C")
    
    return pdf.output()

def generate_ai_recommendations(student_row) -> list:
    """Uses Gemini to generate 4 actionable recommendations based on student data."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""Based on this student's data, generate exactly 4 short actionable recommendations. 
Each recommendation should be 1-2 sentences max. Be specific with numbers from the data.

Student Data:
- CGPA: {student_row.get('cgpa', 0):.2f}
- Attendance: {student_row.get('attendance', 0):.1f}%
- Study Time: {student_row.get('studytime', 0)} hrs/day
- Math: {student_row.get('math', 0)}, DSA: {student_row.get('dsa', 0)}, Networks: {student_row.get('networks', 0)}, OS: {student_row.get('os', 0)}
- Risk Level: {student_row.get('risk_band', 'N/A')}
- Stress Level: {student_row.get('stress_band', 'N/A')}
- Career Preference: {student_row.get('career_preference', 'N/A')}
- Active Backlogs: {student_row.get('active_backlogs', 0)}

Format your response as exactly 4 lines, each starting with a title followed by a pipe symbol, then the recommendation.
Example format:
Focus Area | Your Networks score of 35 is below passing. Dedicate 1 extra hour daily to this subject.
Career Tip | With strong DSA at 82, explore software engineering roles or competitive programming.
Study Plan | Increase study time from 2 to 4 hours daily to improve your 6.2 CGPA.
Well-being | Your stress level is high. Consider joining a study group to share the academic load.
"""
        response = model.generate_content(prompt)
        lines = [l.strip() for l in response.text.strip().split('\n') if '|' in l][:4]
        recommendations = []
        for line in lines:
            parts = line.split('|', 1)
            if len(parts) == 2:
                recommendations.append({"title": parts[0].strip(), "text": parts[1].strip()})
        return recommendations
    except Exception as e:
        return [
            {"title": "Focus Area", "text": f"Review your weakest subject to improve overall performance."},
            {"title": "Study Plan", "text": f"Aim for at least 4 hours of daily study to boost your CGPA."},
            {"title": "Attendance", "text": f"Maintain above 75% attendance for best academic outcomes."},
            {"title": "Career Prep", "text": f"Explore internships related to your career preference."}
        ]

def generate_quiz_questions(weak_subject: str, marks: float) -> list:
    """Uses Gemini to generate 5 MCQs for a given subject."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""Generate exactly 5 multiple-choice questions for a college-level {weak_subject} course.
The student scored {marks:.0f}/100, so adjust difficulty accordingly (easier if score is low, moderate if mid-range).

Rules:
- Each question must have exactly 4 options (A, B, C, D)
- Exactly one correct answer per question
- Questions should test fundamental concepts
- Keep questions concise (1-2 lines max)

Format EXACTLY like this (use | as separator, one question per line):
What is the time complexity of binary search? | O(n) | O(log n) | O(n^2) | O(1) | B
Which layer handles routing in OSI model? | Transport | Network | Data Link | Application | B
"""
        response = model.generate_content(prompt)
        questions = []
        for line in response.text.strip().split('\n'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 6:
                questions.append({
                    "question": parts[0],
                    "options": [parts[1], parts[2], parts[3], parts[4]],
                    "answer": parts[5].upper()
                })
        return questions[:5] if questions else _fallback_quiz(weak_subject)
    except:
        return _fallback_quiz(weak_subject)

def _fallback_quiz(subject: str) -> list:
    """Fallback quiz if Gemini is unavailable."""
    fallbacks = {
        "math": [
            {"question": "What is the derivative of x^2?", "options": ["x", "2x", "x^2", "2"], "answer": "B"},
            {"question": "What is the integral of 1/x?", "options": ["x", "ln(x)", "1/x^2", "e^x"], "answer": "B"},
            {"question": "What is the value of e^0?", "options": ["0", "1", "e", "infinity"], "answer": "B"},
            {"question": "Eigenvalues are roots of which equation?", "options": ["Quadratic", "Characteristic", "Differential", "Linear"], "answer": "B"},
            {"question": "What is the rank of a 3x3 identity matrix?", "options": ["1", "2", "3", "0"], "answer": "C"},
        ],
        "dsa": [
            {"question": "Time complexity of binary search?", "options": ["O(n)", "O(log n)", "O(n^2)", "O(1)"], "answer": "B"},
            {"question": "Which data structure uses FIFO?", "options": ["Stack", "Queue", "Tree", "Graph"], "answer": "B"},
            {"question": "Worst case of quicksort?", "options": ["O(n log n)", "O(n)", "O(n^2)", "O(log n)"], "answer": "C"},
            {"question": "Which traversal gives sorted order in BST?", "options": ["Preorder", "Postorder", "Inorder", "Level order"], "answer": "C"},
            {"question": "Hash table average lookup time?", "options": ["O(n)", "O(log n)", "O(1)", "O(n^2)"], "answer": "C"},
        ],
        "networks": [
            {"question": "Which layer handles routing?", "options": ["Transport", "Network", "Data Link", "Application"], "answer": "B"},
            {"question": "TCP is which type of protocol?", "options": ["Connectionless", "Connection-oriented", "Stateless", "None"], "answer": "B"},
            {"question": "Default port for HTTP?", "options": ["21", "443", "80", "22"], "answer": "C"},
            {"question": "IP address is in which OSI layer?", "options": ["Transport", "Network", "Session", "Physical"], "answer": "B"},
            {"question": "Which protocol resolves IP to MAC?", "options": ["DNS", "DHCP", "ARP", "ICMP"], "answer": "C"},
        ],
        "os": [
            {"question": "Which scheduling is non-preemptive?", "options": ["Round Robin", "SRTF", "FCFS", "Priority (preemptive)"], "answer": "C"},
            {"question": "Deadlock requires how many conditions?", "options": ["2", "3", "4", "5"], "answer": "C"},
            {"question": "Virtual memory uses which storage?", "options": ["RAM only", "Disk + RAM", "Cache", "ROM"], "answer": "B"},
            {"question": "Which is not a page replacement algorithm?", "options": ["FIFO", "LRU", "Optimal", "SJF"], "answer": "D"},
            {"question": "Semaphore is used for?", "options": ["Scheduling", "Synchronization", "Memory mgmt", "File mgmt"], "answer": "B"},
        ],
    }
    return fallbacks.get(subject.lower(), fallbacks["dsa"])

# ==========================================
# 4. SECURE AUTHENTICATION UI
# ==========================================
def auth_screen():
    st.markdown("""
    <style>
    @keyframes borderGlow {
        0%, 100% { border-color: rgba(99,102,241,0.4); box-shadow: 0 0 30px rgba(99,102,241,0.1); }
        50% { border-color: rgba(139,92,246,0.6); box-shadow: 0 0 50px rgba(139,92,246,0.15); }
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 520px !important; margin: 40px auto !important; padding: 44px 40px !important; border-radius: 20px !important;
        background: rgba(15,15,26,0.85) !important; backdrop-filter: blur(20px);
        border: 1px solid rgba(99,102,241,0.3) !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        animation: borderGlow 4s ease-in-out infinite;
    }
    .auth-title {
        text-align: center; font-size: 1.8rem; font-weight: bold; margin-bottom: 0.3rem;
        background: linear-gradient(135deg, #a5b4fc, #8b5cf6, #6366f1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .auth-subtitle { text-align: center; color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='auth-title'>Academic Portal</div>", unsafe_allow_html=True)
    st.markdown("<div class='auth-subtitle'>Student Success & Readiness Platform</div>", unsafe_allow_html=True)
    
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

        /* ===== GLOBAL & SCROLLBAR ===== */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: #0a0a0f; }}
        ::-webkit-scrollbar-thumb {{ background: #6366f1; border-radius: 8px; }}
        .stApp {{ background: linear-gradient(160deg, #0a0a0f 0%, #10101a 50%, #0d0d1a 100%); }}
        header[data-testid="stHeader"] {{ background: rgba(10,10,15,0.8); backdrop-filter: blur(12px); border-bottom: 1px solid rgba(99,102,241,0.15); }}

        /* ===== ANIMATED HERO ===== */
        @keyframes heroShift {{ 0%{{background-position:0% 50%}} 50%{{background-position:100% 50%}} 100%{{background-position:0% 50%}} }}
        .hero {{
            background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa, #6366f1);
            background-size: 300% 300%; animation: heroShift 8s ease infinite;
            color: white; border-radius: 16px; padding: 28px 34px; margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(99,102,241,0.25); border: 1px solid rgba(255,255,255,0.08);
            position: relative; overflow: hidden;
        }}
        .hero::before {{ content:''; position:absolute; inset:0; background:radial-gradient(circle at 20% 80%, rgba(139,92,246,0.3) 0%, transparent 50%),radial-gradient(circle at 80% 20%, rgba(99,102,241,0.2) 0%,transparent 50%); pointer-events:none; }}
        .hero h2 {{ margin:0 0 0.5rem 0; font-size:1.9rem; font-weight:bold; letter-spacing:-0.5px; text-shadow:0 2px 8px rgba(0,0,0,0.3); position:relative; z-index:1; }}
        .hero p {{ margin:0; opacity:0.92; font-size:1.05rem; font-weight:normal; position:relative; z-index:1; }}

        /* ===== GLASS KPI CARDS ===== */
        .kpi-card {{
            border-radius: 16px; padding: 22px; min-height: 120px;
            background: rgba(20,20,32,0.6); backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            transition: all 0.35s cubic-bezier(.4,0,.2,1);
        }}
        .kpi-card:hover {{ transform: translateY(-5px) scale(1.02); box-shadow: 0 12px 36px rgba(0,0,0,0.4); border-color: rgba(255,255,255,0.12); }}
        .kpi-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
        .kpi-title {{ font-size:0.78rem; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:1px; }}
        .kpi-value {{ font-size:2.2rem; font-weight:bold; line-height:1.1; margin-bottom:8px; color:#f1f5f9; }}
        .kpi-caption {{ color:#64748b; font-size:0.8rem; font-weight:normal; }}

        .kpi-card.teal {{ border-left:4px solid #14b8a6; box-shadow:0 4px 20px rgba(20,184,166,0.12); }}
        .kpi-card.blue {{ border-left:4px solid #3b82f6; box-shadow:0 4px 20px rgba(59,130,246,0.12); }}
        .kpi-card.amber {{ border-left:4px solid #f59e0b; box-shadow:0 4px 20px rgba(245,158,11,0.12); }}
        .kpi-card.rose {{ border-left:4px solid #f43f5e; box-shadow:0 4px 20px rgba(244,63,94,0.12); }}
        .kpi-card.emerald {{ border-left:4px solid #10b981; box-shadow:0 4px 20px rgba(16,185,129,0.12); }}
        .kpi-card.purple {{ border-left:4px solid #8b5cf6; box-shadow:0 4px 20px rgba(139,92,246,0.12); }}

        /* ===== SECTION TITLES ===== */
        .section-title {{
            margin:1.8rem 0 1rem 0; font-weight:bold; font-size:1.4rem; padding-bottom:0.6rem;
            border-bottom:2px solid transparent;
            border-image: linear-gradient(90deg, #6366f1, #8b5cf6, transparent) 1;
            color: #e2e8f0;
        }}
        .chart-title {{ font-size:1.15rem; font-weight:600; margin-bottom:10px; color:#cbd5e1; }}

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0f0f1a, #141420) !important;
            border-right: 1px solid rgba(99,102,241,0.15);
        }}
        section[data-testid="stSidebar"] .stButton > button {{
            background: linear-gradient(135deg, #ef4444, #dc2626) !important;
            color: white !important; border: none !important; border-radius: 10px !important;
            font-weight: 600 !important; transition: all 0.3s ease !important;
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
            box-shadow: 0 6px 20px rgba(239,68,68,0.4) !important; transform: translateY(-2px) !important;
        }}

        /* ===== CONTAINERS & EXPANDERS ===== */
        div[data-testid="stExpander"] {{
            background: rgba(20,20,32,0.5) !important; border: 1px solid rgba(255,255,255,0.06) !important;
            border-radius: 12px !important; backdrop-filter: blur(8px);
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(20,20,32,0.6); border-radius: 10px 10px 0 0; border: 1px solid rgba(255,255,255,0.06);
            color: #94a3b8; padding: 10px 24px;
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(99,102,241,0.15) !important; color: #a5b4fc !important;
            border-bottom: 2px solid #6366f1 !important;
        }}

        /* ===== INPUTS ===== */
        .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div {{
            background: rgba(20,20,32,0.7) !important; border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 10px !important; color: #e2e8f0 !important;
            transition: border-color 0.3s ease !important;
        }}
        .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {{
            border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
        }}

        /* ===== BUTTONS ===== */
        .stButton > button[kind="primary"], button[data-testid="stFormSubmitButton"] {{
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            color: white !important; border: none !important; border-radius: 10px !important;
            font-weight: 600 !important; transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            box-shadow: 0 8px 25px rgba(99,102,241,0.45) !important; transform: translateY(-2px) !important;
        }}
        .stDownloadButton > button {{
            background: rgba(20,20,32,0.7) !important; border: 1px solid rgba(99,102,241,0.3) !important;
            color: #a5b4fc !important; border-radius: 10px !important;
        }}

        /* ===== DATAFRAMES ===== */
        .stDataFrame {{ border-radius: 12px !important; overflow: hidden; }}

        /* ===== CHAT ===== */
        .stChatMessage {{ background: rgba(20,20,32,0.5) !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.05) !important; }}
        .stChatInputContainer {{ border-color: rgba(99,102,241,0.3) !important; }}

        /* ===== PLOTLY CONTAINERS ===== */
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            border-color: rgba(255,255,255,0.06) !important; border-radius: 14px !important;
        }}

        /* ===== DIVIDER ===== */
        hr {{ border-color: rgba(99,102,241,0.15) !important; }}

        </style>
        """, unsafe_allow_html=True)
    
    # GATEKEEPER CHECK
    if not st.session_state.logged_in:
        auth_screen()
        return
    
    # Reset auth-only CSS override for dashboard views
    st.markdown("""
    <style>
    [data-testid="stMainBlockContainer"] {
        max-width: none !important; margin: 0 auto !important; padding: 1rem 1rem !important;
        border-radius: 0 !important; background: transparent !important;
        border: none !important; box-shadow: none !important; animation: none !important;
        backdrop-filter: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # SIDEBAR PROFILE & LOGOUT
    with st.sidebar:
        st.markdown(f"### Profile")
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
                fig_sgpa.update_traces(line_color='#818cf8', marker=dict(size=8, color='#a5b4fc'))
                
                if predicted_sgpa is not None:
                    fig_sgpa.add_scatter(x=["Sem 5 (Est)"], y=[predicted_sgpa], mode='markers', marker=dict(color='#c084fc', size=12, symbol='star'), name="AI Prediction")
                    
                fig_sgpa.update_layout(height=320, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', xaxis=dict(showgrid=False, color='#94a3b8'), yaxis=dict(range=[0, 10], title=None, gridcolor='rgba(148,163,184,0.08)', color='#94a3b8'))
                st.plotly_chart(fig_sgpa, use_container_width=True)
        with c2:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>Subject Performance</div>", unsafe_allow_html=True)
                subject_student = pd.DataFrame({"Subject": ["Math", "DSA", "Networks", "OS"], "Marks": [student_row["math"], student_row["dsa"], student_row["networks"], student_row["os"]]})
                avg_score = round(subject_student["Marks"].mean(), 1)
                fig_donut = px.pie(subject_student, names="Subject", values="Marks", hole=0.7, color_discrete_sequence=["#14b8a6", "#3b82f6", "#8b5cf6", "#f43f5e"])
                fig_donut.update_traces(textinfo='percent', textposition='inside', insidetextorientation='horizontal', textfont_color='#f1f5f9')
                fig_donut.update_layout(height=320, margin=dict(t=10, b=20, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(color='#94a3b8')), annotations=[dict(text=f"{avg_score}%<br>Avg", x=0.5, y=0.5, font_size=18, font_color='#e2e8f0', showarrow=False)])
                st.plotly_chart(fig_donut, use_container_width=True)
        
        # --- NEW: IMPROVEMENT TRACKER & PROGRESS ---
        st.markdown("<div class='section-title'>Performance Insights</div>", unsafe_allow_html=True)
        
        it1, it2, it3 = st.columns(3)
        
        # Improvement Tracker
        sgpa_change = float(student_row['sgpa4'] - student_row['sgpa1'])
        recent_change = float(student_row['sgpa4'] - student_row['sgpa3'])
        change_pct = (sgpa_change / max(float(student_row['sgpa1']), 0.01)) * 100
        
        change_label = f"+{sgpa_change:.2f}" if sgpa_change >= 0 else f"{sgpa_change:.2f}"
        change_tone = "emerald" if sgpa_change >= 0 else "rose"
        with it1: kpi_card("Overall SGPA Change", change_label, f"Sem 1 to Sem 4 ({change_pct:+.1f}%)", change_tone)
        
        recent_label = f"+{recent_change:.2f}" if recent_change >= 0 else f"{recent_change:.2f}"
        recent_tone = "emerald" if recent_change >= 0 else "rose"
        with it2: kpi_card("Recent Momentum", recent_label, "Sem 3 to Sem 4", recent_tone)
        
        # Academic Progress Bar
        readiness_val = float(student_row["career_readiness_score"])
        progress_color = "#10b981" if readiness_val >= 70 else "#f59e0b" if readiness_val >= 45 else "#f43f5e"
        with it3:
            st.markdown(f"""
            <div class="kpi-card teal">
                <div class="kpi-header"><span class="kpi-title">Career Readiness</span></div>
                <div class="kpi-value" style="font-size:1.6rem;">{readiness_val:.0f}%</div>
                <div style="background:rgba(255,255,255,0.06); border-radius:8px; height:10px; margin-top:8px; overflow:hidden;">
                    <div style="width:{min(readiness_val, 100):.0f}%; height:100%; background:{progress_color}; border-radius:8px; transition: width 0.5s ease;"></div>
                </div>
                <div class="kpi-caption" style="margin-top:6px;">progress toward readiness</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # --- NEW: PEER COMPARISON & STRENGTH/WEAKNESS ---
        pc1, pc2 = st.columns(2)
        
        with pc1:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>Peer Comparison</div>", unsafe_allow_html=True)
                
                # Load full class data for comparison
                try:
                    conn_full = get_db_connection()
                    class_df = conn_full.query("SELECT cgpa, attendance, studytime, math, dsa, networks, os FROM fact_student_performance;", ttl=120)
                    class_avg_cgpa = float(class_df['cgpa'].mean()) * 10  # normalize to 0-100
                    class_avg_att = float(class_df['attendance'].mean())
                    class_avg_study = min(float(class_df['studytime'].mean()) * 10, 100)
                    class_avg_subj = float(class_df[['math','dsa','networks','os']].mean(axis=1).mean())
                except:
                    class_avg_cgpa, class_avg_att, class_avg_study, class_avg_subj = 60, 70, 40, 55
                
                categories = ['CGPA', 'Attendance', 'Study Effort', 'Subject Avg']
                student_vals = [
                    float(student_row['cgpa']) * 10,
                    float(student_row['attendance']),
                    min(float(student_row['studytime']) * 10, 100),
                    float(student_row['subject_avg'])
                ]
                class_vals = [class_avg_cgpa, class_avg_att, class_avg_study, class_avg_subj]
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=student_vals + [student_vals[0]], theta=categories + [categories[0]], fill='toself', name='You', line_color='#818cf8', fillcolor='rgba(129,140,248,0.15)'))
                fig_radar.add_trace(go.Scatterpolar(r=class_vals + [class_vals[0]], theta=categories + [categories[0]], fill='toself', name='Class Avg', line_color='#64748b', fillcolor='rgba(100,116,139,0.08)'))
                fig_radar.update_layout(
                    polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(148,163,184,0.1)', color='#64748b'), angularaxis=dict(gridcolor='rgba(148,163,184,0.1)', color='#94a3b8')),
                    height=340, margin=dict(l=40, r=40, t=30, b=30), paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        
        with pc2:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>Subject Strengths</div>", unsafe_allow_html=True)
                
                subjects = ['Math', 'DSA', 'Networks', 'OS']
                marks = [float(student_row['math']), float(student_row['dsa']), float(student_row['networks']), float(student_row['os'])]
                colors = ['#10b981' if m >= 60 else '#f59e0b' if m >= 40 else '#f43f5e' for m in marks]
                
                fig_bars = go.Figure()
                fig_bars.add_trace(go.Bar(y=subjects, x=marks, orientation='h', marker_color=colors, text=[f"{m:.0f}" for m in marks], textposition='outside', textfont_color='#94a3b8'))
                fig_bars.add_vline(x=40, line_dash="dash", line_color="rgba(244,63,94,0.4)", annotation_text="Pass", annotation_font_color="#f43f5e")
                fig_bars.update_layout(height=340, margin=dict(l=10, r=40, t=30, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', xaxis=dict(range=[0, 105], gridcolor='rgba(148,163,184,0.08)', color='#94a3b8'), yaxis=dict(color='#94a3b8'), showlegend=False)
                st.plotly_chart(fig_bars, use_container_width=True)
                
                strongest = subjects[marks.index(max(marks))]
                weakest = subjects[marks.index(min(marks))]
                st.markdown(f"<div style='color:#94a3b8; font-size:0.85rem;'>Strongest: <b style='color:#10b981'>{strongest} ({max(marks):.0f})</b> &nbsp; Weakest: <b style='color:#f43f5e'>{weakest} ({min(marks):.0f})</b></div>", unsafe_allow_html=True)

        with st.expander("View Personal Summary Table"):
            personal_summary = pd.DataFrame({
                "Metric": ["Study Time (hrs/day)", "Career Preference", "Part-time Job", "Readiness Score", "AI Risk Level", "AI Stress Level", "10th Mark", "12th Mark"],
                "Value": [round(float(student_row["studytime"]), 2), student_row["career_preference"].title(), student_row["part_timejob"].title(), round(float(student_row["career_readiness_score"]), 2), student_row["risk_band"], student_row["stress_band"], round(float(student_row["10th_mark"]), 2), round(float(student_row["12th_mark"]), 2)]
            })
            personal_summary["Value"] = personal_summary["Value"].astype(str)
            st.dataframe(personal_summary, width="stretch", hide_index=True)
        
        # --- AI RECOMMENDATIONS PANEL ---
        st.markdown("<div class='section-title'>AI-Powered Recommendations</div>", unsafe_allow_html=True)
        
        # Cache recommendations in session state
        if "ai_recs" not in st.session_state or st.button("Refresh Recommendations", key="refresh_recs"):
            with st.spinner("Generating personalized recommendations..."):
                st.session_state.ai_recs = generate_ai_recommendations(student_row)
        
        rec_colors = ["#818cf8", "#14b8a6", "#f59e0b", "#f43f5e"]
        rec_cols = st.columns(len(st.session_state.ai_recs))
        for i, rec in enumerate(st.session_state.ai_recs):
            color = rec_colors[i % len(rec_colors)]
            with rec_cols[i]:
                st.markdown(f"""
                <div class="kpi-card" style="border-left: 3px solid {color}; min-height: 140px;">
                    <div style="font-weight:700; color:{color}; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">{rec['title']}</div>
                    <div style="color:#cbd5e1; font-size:0.82rem; line-height:1.5;">{rec['text']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.write("")
        
        # --- PDF REPORT DOWNLOAD ---
        dl1, dl2 = st.columns([3, 1])
        with dl2:
            pdf_bytes = generate_student_pdf_report(student_row, predicted_sgpa)
            st.download_button(
                label="Download Report Card (PDF)",
                data=pdf_bytes,
                file_name=f"report_card_{int(student_row.get('id', 0))}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        # --- WEEKLY QUIZ SECTION ---
        st.divider()
        st.markdown("<div class='section-title'>Weekly Self-Assessment Quiz</div>", unsafe_allow_html=True)
        
        # Detect weakest subject
        subj_map = {'math': float(student_row['math']), 'dsa': float(student_row['dsa']), 'networks': float(student_row['networks']), 'os': float(student_row['os'])}
        weak_subj = min(subj_map, key=subj_map.get)
        weak_score = subj_map[weak_subj]
        
        st.markdown(f"<div style='color:#94a3b8; margin-bottom:12px;'>Your weakest subject is <b style='color:#f43f5e'>{weak_subj.upper()} ({weak_score:.0f}/100)</b>. This quiz is tailored to help you improve.</div>", unsafe_allow_html=True)
        
        # Quiz state management
        if "quiz_questions" not in st.session_state:
            st.session_state.quiz_questions = None
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = 0
            st.session_state.quiz_subject = None
        
        qcol1, qcol2 = st.columns([1, 4])
        with qcol1:
            if st.button("Start Quiz", use_container_width=True, key="start_quiz"):
                with st.spinner(f"Generating {weak_subj.upper()} questions..."):
                    st.session_state.quiz_questions = generate_quiz_questions(weak_subj, weak_score)
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_subject = weak_subj.upper()
                    st.rerun()
        
        if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
            st.markdown(f"<div style='color:#a5b4fc; font-weight:600; margin:12px 0 8px;'>Subject: {st.session_state.quiz_subject} — 5 Questions</div>", unsafe_allow_html=True)
            
            with st.form("quiz_form"):
                answers = {}
                for idx, q in enumerate(st.session_state.quiz_questions):
                    st.markdown(f"**Q{idx+1}. {q['question']}**")
                    option_labels = [f"A) {q['options'][0]}", f"B) {q['options'][1]}", f"C) {q['options'][2]}", f"D) {q['options'][3]}"]
                    answers[idx] = st.radio("Select answer:", option_labels, key=f"q_{idx}", label_visibility="collapsed", index=None)
                    st.write("")
                
                submitted = st.form_submit_button("Submit Quiz", use_container_width=True)
                if submitted:
                    # Check if all questions are answered
                    unanswered = [i+1 for i in range(len(st.session_state.quiz_questions)) if answers[i] is None]
                    if unanswered:
                        st.warning(f"Please answer all questions. Unanswered: Q{', Q'.join(map(str, unanswered))}")
                    else:
                        score = 0
                        results = []
                        for idx, q in enumerate(st.session_state.quiz_questions):
                            selected_letter = answers[idx][0]  # Gets 'A', 'B', 'C', or 'D'
                            selected_text = answers[idx][3:]   # Gets the option text after "A) "
                            correct = q['answer'].strip().upper()
                            is_correct = selected_letter == correct
                            if is_correct:
                                score += 1
                            results.append({"correct": is_correct, "selected": selected_letter, "selected_text": selected_text, "answer": correct})
                        
                        st.session_state.quiz_score = score
                        st.session_state.quiz_submitted = True
                        st.session_state.quiz_results = results
                        st.rerun()
        
        if st.session_state.quiz_submitted and st.session_state.quiz_questions:
            score = st.session_state.quiz_score
            total = len(st.session_state.quiz_questions)
            pct = (score / total) * 100
            
            if pct >= 80:
                score_color, score_msg = "#10b981", "Excellent"
            elif pct >= 50:
                score_color, score_msg = "#f59e0b", "Good effort"
            else:
                score_color, score_msg = "#f43f5e", "Keep practicing"
            
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center; border: 1px solid {score_color};">
                <div style="font-size:2rem; font-weight:800; color:{score_color};">{score}/{total}</div>
                <div style="color:#94a3b8; font-size:0.9rem; margin-top:4px;">{score_msg} — {pct:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show answers review
            with st.expander("Review Answers"):
                for idx, q in enumerate(st.session_state.quiz_questions):
                    result = st.session_state.quiz_results[idx]
                    icon = "✓" if result['correct'] else "✗"
                    color = "#10b981" if result['correct'] else "#f43f5e"
                    st.markdown(f"<span style='color:{color}; font-weight:700;'>{icon}</span> **Q{idx+1}. {q['question']}**", unsafe_allow_html=True)
                    correct_idx = ord(result['answer']) - ord('A')
                    correct_text = q['options'][correct_idx] if 0 <= correct_idx < len(q['options']) else "?"
                    if result['correct']:
                        st.markdown(f"<span style='color:#10b981; font-size:0.85rem;'>Your answer: {result['selected']}) {result['selected_text']} — Correct!</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color:#f43f5e; font-size:0.85rem;'>Your answer: {result['selected']}) {result['selected_text']}</span><br><span style='color:#10b981; font-size:0.85rem;'>Correct answer: {result['answer']}) {correct_text}</span>", unsafe_allow_html=True)
                    st.write("")
            
            if st.button("Take New Quiz", key="new_quiz"):
                st.session_state.quiz_questions = None
                st.session_state.quiz_submitted = False
                st.rerun()
            
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
                fig_cohort_sgpa.update_traces(line_color='#a78bfa', marker=dict(size=8, color='#c4b5fd'))
                fig_cohort_sgpa.update_layout(height=320, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', xaxis=dict(showgrid=False, color='#94a3b8'), yaxis=dict(range=[0, 10], title=None, gridcolor='rgba(148,163,184,0.08)', color='#94a3b8'))
                st.plotly_chart(fig_cohort_sgpa, use_container_width=True)
        with g2:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>AI Risk Level Distribution</div>", unsafe_allow_html=True)
                risk_dist = filtered["risk_band"].value_counts().rename_axis("Risk Level").reset_index(name="Students")
                color_mapping = {"High Risk": "#f43f5e", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"}
                fig_risk = px.bar(risk_dist, x="Risk Level", y="Students", color="Risk Level", color_discrete_map=color_mapping)
                fig_risk.update_layout(height=320, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', showlegend=False, xaxis=dict(showgrid=False, color='#94a3b8'), yaxis=dict(title=None, gridcolor='rgba(148,163,184,0.08)', color='#94a3b8'))
                st.plotly_chart(fig_risk, use_container_width=True)
        with g3:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>AI Stress Level Distribution</div>", unsafe_allow_html=True)
                stress_dist = filtered["stress_band"].value_counts().rename_axis("Stress Level").reset_index(name="Students")
                color_mapping_stress = {"High Stress": "#f43f5e", "Medium Stress": "#f59e0b", "Low Stress": "#10b981"}
                fig_stress = px.bar(stress_dist, x="Stress Level", y="Students", color="Stress Level", color_discrete_map=color_mapping_stress)
                fig_stress.update_layout(height=320, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', showlegend=False, xaxis=dict(showgrid=False, color='#94a3b8'), yaxis=dict(title=None, gridcolor='rgba(148,163,184,0.08)', color='#94a3b8'))
                st.plotly_chart(fig_stress, use_container_width=True)
        
        # --- NEW TEACHER FEATURES ---
        st.markdown("<div class='section-title'>Detailed Class Analysis</div>", unsafe_allow_html=True)
        
        tc1, tc2 = st.columns(2)
        
        # Subject Performance Donut Chart
        with tc1:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>Subject-wise Class Averages</div>", unsafe_allow_html=True)
                subj_names = ['Math', 'DSA', 'Networks', 'OS']
                subj_keys = ['math', 'dsa', 'networks', 'os']
                subj_avgs = [float(filtered[s].mean()) for s in subj_keys]
                subj_colors_t = ['#14b8a6', '#3b82f6', '#8b5cf6', '#f59e0b']
                
                fig_subj_donut = px.pie(pd.DataFrame({'Subject': subj_names, 'Avg Marks': subj_avgs}), names='Subject', values='Avg Marks', hole=0.55, color_discrete_sequence=subj_colors_t)
                fig_subj_donut.update_traces(textinfo='label+value', texttemplate='%{label}<br>%{value:.1f}', textfont_color='#f1f5f9')
                overall_avg = sum(subj_avgs) / len(subj_avgs)
                fig_subj_donut.update_layout(
                    height=340, margin=dict(l=10, r=10, t=10, b=30),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
                    showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
                    annotations=[dict(text=f'{overall_avg:.1f}<br>Avg', x=0.5, y=0.5, font_size=16, font_color='#e2e8f0', showarrow=False)]
                )
                st.plotly_chart(fig_subj_donut, use_container_width=True)
        
        # Career Readiness Donut by Band
        with tc2:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>Career Readiness Breakdown</div>", unsafe_allow_html=True)
                readiness_dist = filtered['career_readiness_band'].value_counts().reset_index()
                readiness_dist.columns = ['Band', 'Students']
                band_colors = {'Ready': '#10b981', 'Developing': '#f59e0b', 'Needs Support': '#f43f5e'}
                fig_readiness = px.pie(readiness_dist, names='Band', values='Students', hole=0.55, color='Band', color_discrete_map=band_colors)
                fig_readiness.update_traces(textinfo='label+value', texttemplate='%{label}<br>%{value}', textfont_color='#f1f5f9')
                fig_readiness.update_layout(
                    height=340, margin=dict(l=10, r=10, t=10, b=30),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
                    showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
                )
                st.plotly_chart(fig_readiness, use_container_width=True)
        
        st.write("")
        td1, td2 = st.columns(2)
        
        # Top Performers Table
        with td1:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>Top 10 Performers</div>", unsafe_allow_html=True)
                top_cols = ["id", "cgpa", "attendance", "risk_band", "stress_band", subject_assigned]
                top_performers = filtered.sort_values("cgpa", ascending=False).head(10)[top_cols].round(2)
                top_performers.columns = ["ID", "CGPA", "Attendance", "Risk", "Stress", subject_assigned.upper()]
                st.dataframe(top_performers, width="stretch", hide_index=True)
        
        # Study Time Distribution
        with td2:
            with st.container(border=True):
                st.markdown("<div class='chart-title'>Study Time Distribution</div>", unsafe_allow_html=True)
                fig_hist = px.histogram(filtered, x="studytime", nbins=12, color_discrete_sequence=["#818cf8"])
                fig_hist.update_layout(
                    height=340, margin=dict(l=10, r=20, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
                    xaxis=dict(title="Daily Study Hours", gridcolor='rgba(148,163,184,0.08)', color='#94a3b8'),
                    yaxis=dict(title="Students", gridcolor='rgba(148,163,184,0.08)', color='#94a3b8'),
                    bargap=0.1, showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        st.write("")
        c3, c4 = st.columns(2)
        with c3:
            st.subheader(f"{subject_assigned.upper()} — Difficulty Analysis")
            avg_mark = float(filtered[subject_assigned].mean())
            fail_rate = float((filtered[subject_assigned] < 40).mean() * 100)
            pass_rate = 100 - fail_rate
            score = (100 - avg_mark) * 0.7 + fail_rate * 0.3
            difficulty_df = pd.DataFrame({
                "Metric": ["Average Mark", "Pass Rate", "Fail Rate", "Difficulty Score"],
                "Value": [f"{avg_mark:.1f}", f"{pass_rate:.1f}%", f"{fail_rate:.1f}%", f"{score:.1f}"]
            })
            st.dataframe(difficulty_df, width="stretch", hide_index=True)
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
        with st.expander(f"High Risk Students ({len(high_risk)} students - Priority Interventions Needed)"):
            if high_risk.empty:
                st.info("No high-risk students in current filters.")
            else:
                st.dataframe(high_risk, width="stretch", hide_index=True)
                
        high_stress = filtered[filtered["stress_band"] == "High Stress"].sort_values(["cgpa", "attendance"], ascending=[True, True]) \
            .loc[:, common_cols].head(15)
        high_stress = high_stress.round(2)
        with st.expander(f"High Stress Students ({len(high_stress)} students - Well-being Check Needed)"):
            if high_stress.empty:
                st.info("No high-stress students in current filters.")
            else:
                st.dataframe(high_stress, width="stretch", hide_index=True)
        
        medium_risk = filtered[filtered["risk_band"] == "Medium Risk"].sort_values(["cgpa", "attendance"], ascending=[True, True]) \
            .loc[:, common_cols].head(15)
        medium_risk = medium_risk.round(2)
        with st.expander(f"Medium Risk Students ({len(medium_risk)} students - Monitor Closely)"):
            if medium_risk.empty:
                st.info("No medium-risk students in current filters.")
            else:
                st.dataframe(medium_risk, width="stretch", hide_index=True)
        
        low_risk = filtered[filtered["risk_band"] == "Low Risk"].sort_values(["cgpa"], ascending=[False]) \
            .loc[:, common_cols].head(15)
        low_risk = low_risk.round(2)
        with st.expander(f"Low Risk Students ({len(low_risk)} students - Strong Performers)"):
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