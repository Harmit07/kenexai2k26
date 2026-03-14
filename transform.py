import pandas as pd
from sqlalchemy import create_engine

# 1. Connect to your DB
engine = create_engine("postgresql://postgres:0000@localhost:5433/student_analytics_db")

# 2. Pull the raw data
df_master = pd.read_sql("SELECT * FROM master_student_data", engine)

print("🚀 Building the complete Star Schema...")

# --- TABLE 1: dim_student (The missing piece!) ---
dim_student = df_master[['id', '10th_mark', '12th_mark']].drop_duplicates()
dim_student.to_sql('dim_student', engine, if_exists='replace', index=False)
print("✅ Created dim_student")

# --- TABLE 2: dim_career ---
dim_career = df_master[['career_preference', 'part_timejob']].drop_duplicates().reset_index(drop=True)
dim_career['career_key'] = dim_career.index + 1
dim_career.to_sql('dim_career', engine, if_exists='replace', index=False)
print("✅ Created dim_career")

# --- TABLE 3: fact_student_performance ---
fact_df = df_master.merge(dim_career, on=['career_preference', 'part_timejob'])
# We include all metrics needed for the dashboard
fact_cols = [
    'id', 'career_key', 'sgpa1', 'sgpa2', 'sgpa3', 'sgpa4', 'cgpa', 
    'active_backlogs', 'total_backlogs', 'math', 'dsa', 'networks', 
    'os', 'attendance', 'studytime'
]
fact_df[fact_cols].to_sql('fact_student_performance', engine, if_exists='replace', index=False)
print("✅ Created fact_student_performance")

print("\n🎉 Your database is now fully optimized for the Star Schema!")