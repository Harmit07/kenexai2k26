import pandas as pd
from sqlalchemy import create_engine
import re
import numpy as np
from datetime import datetime

# ==========================================
# 1. EXTRACT
# ==========================================
file_path = 'Student_Data.csv'
print("🚀 Extracting data...")
df = pd.read_csv(file_path)

# ==========================================
# 2. TRANSFORM (Cleaning & Business Logic)
# ==========================================
print("🧹 Cleaning column names...")
def clean_column_name(name):
    name = str(name).lower().strip().replace(' ', '_').replace('&', 'and')
    return re.sub(r'[^a-z0-9_]', '', name)

df.columns = [clean_column_name(col) for col in df.columns]

print("📊 Calculating Gold Layer metrics (Risk & Readiness)...")
def compute_metrics(df):
    out = df.copy()
    # Risk Calculation
    backlog_comp = (out["active_backlogs"] * 20 + out["total_backlogs"] * 6).clip(0, 40)
    attendance_comp = ((75 - out["attendance"]).clip(lower=0) * 0.9).clip(0, 20)
    cgpa_comp = ((6.5 - out["cgpa"]).clip(lower=0) * 7).clip(0, 20)
    study_comp = np.where(out["studytime"] < 2, 10, np.where(out["studytime"] < 4, 5, 0))
    
    out["risk_score"] = (backlog_comp + attendance_comp + cgpa_comp + study_comp).clip(0, 100)
    out["risk_band"] = pd.cut(out["risk_score"], bins=[-1, 29, 54, 100], labels=["Low", "Medium", "High"])
    
    # Career Readiness
    subj_avg = out[["math", "dsa", "networks", "os"]].mean(axis=1)
    readiness = (out["cgpa"] * 8 + (out["attendance"] / 100) * 10 + (subj_avg / 100) * 10 - (out["active_backlogs"] * 4))
    out["career_readiness_score"] = readiness.clip(0, 100)
    out["career_readiness_band"] = pd.cut(out["career_readiness_score"], bins=[-1, 44, 69, 100], labels=["Needs Support", "Developing", "Ready"])
    return out

df = compute_metrics(df)

# ==========================================
# 3. LOAD (Dimensional Modeling)
# ==========================================
print("🔗 Connecting to PostgreSQL...")
engine = create_engine("postgresql://postgres:0000@localhost:5433/student_analytics_db")

try:
    # --- SILVER LAYER: Dimensions ---
    print("🥈 Creating Silver Layer (Dimensions)...")
    
    # 1. Dim_Career: Unique career paths
    dim_career = df[['career_preference', 'part_timejob']].drop_duplicates().reset_index(drop=True)
    dim_career['career_key'] = dim_career.index + 1
    dim_career.to_sql('dim_career', engine, if_exists='replace', index=False)

    # 2. Dim_Attendance: Categorizing attendance percentages
    dim_attendance = df[['attendance']].drop_duplicates().reset_index(drop=True)
    dim_attendance['attendance_key'] = dim_attendance.index + 1
    # Adding business logic for attendance buckets
    dim_attendance['attendance_category'] = np.where(dim_attendance['attendance'] >= 85, 'Excellent', 
                                            np.where(dim_attendance['attendance'] >= 75, 'Satisfactory', 'Critical'))
    dim_attendance.to_sql('dim_attendance', engine, if_exists='replace', index=False)

    # --- GOLD LAYER: Fact Table ---
    print("🥇 Creating Gold Layer (Fact Table)...")
    
    # Merge keys back into the main dataframe
    fact_df = df.merge(dim_career, on=['career_preference', 'part_timejob'])
    fact_df = fact_df.merge(dim_attendance, on=['attendance'])
    
    # Fact Table: Now includes the marks directly and uses the new attendance_key
    fact_cols = [
        'id', 'career_key', 'attendance_key', '10th_mark', '12th_mark', 'studytime',
        'sgpa1', 'sgpa2', 'sgpa3', 'sgpa4', 'cgpa', 
        'attendance', 'active_backlogs', 'total_backlogs', 'math', 'dsa', 
        'networks', 'os', 'risk_score', 'risk_band', 
        'career_readiness_score', 'career_readiness_band'
    ]
    fact_df[fact_cols].to_sql('fact_student_performance', engine, if_exists='replace', index=False)

    # --- ARCHIVE LAYER: History ---
    print("📜 Archiving snapshot for history tracking...")
    archive_df = fact_df[fact_cols].copy()
    archive_df['snapshot_date'] = datetime.now()
    archive_df.to_sql('fact_performance_archive', engine, if_exists='append', index=False)

    print("✅ ETL Complete! Star Schema is ready.")

except Exception as e:
    print(f"❌ Error during Load: {e}")