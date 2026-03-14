import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# 1. Load the datasets
df_research = pd.read_csv("research_student (1).csv")
df_perf = pd.read_csv("Student Performance.csv")
df_behav1 = pd.read_csv("Student_Behaviour.csv")
df_behav2 = pd.read_csv("Student Attitude and Behavior.csv")
df_lifestyle = pd.read_csv("student_lifestyle_dataset.csv")
df_perf_data = pd.read_csv("Student_performance_data _.csv")

# Core target columns (subject columns will be added dynamically)
target_columns = [
    'CGPA', 'SGPA', 'No of backlogs', 'stress level', 'study time',
    '10th percentage', '12th percentage', 'parttime job', 
    'semester academic period', 'attendance', 'career preference'
]

mapped_dfs = []

def create_mapped_df(source_df, mapping_dict):
    """Maps source columns to core target columns."""
    temp_df = pd.DataFrame(columns=target_columns)
    for target_col, source_col in mapping_dict.items():
        if source_col in source_df.columns:
            temp_df[target_col] = source_df[source_col]
    return temp_df

# ==========================================
# SMART SUBJECT HANDLING FOR PERFORMANCE DATA
# ==========================================
# Create a unique column name for each Semester + Subject combination
df_perf['Sem_Sub'] = 'Sem' + df_perf['semester'].astype(str) + '_' + df_perf['SubjectName'].astype(str)

# Pivot the data: 1 Row per Student, Columns = Subjects, Values = Grade Points
subject_marks = df_perf.pivot_table(
    index='Rollno', 
    columns='Sem_Sub', 
    values='TheoryAggGradePoint', 
    aggfunc='mean'
).reset_index()

# Aggregate the rest of the student's base stats into 1 row
base_perf = df_perf.groupby('Rollno').agg({
    'semester': 'max', 
    'SGPA': 'mean',
    'CGPA': 'mean',
    'noofbacklog': 'max',
    'ThIsPresent': 'mean'
}).reset_index()

# Merge back together to create a wide-format dataset for this specific CSV
df_perf_wide = pd.merge(base_perf, subject_marks, on='Rollno', how='left')

# Map the base columns
map_perf = {
    'semester academic period': 'semester', 'SGPA': 'SGPA', 'CGPA': 'CGPA', 
    'No of backlogs': 'noofbacklog', 'attendance': 'ThIsPresent'
}
temp_perf = create_mapped_df(df_perf_wide, map_perf)

# Dynamically attach the newly created subject columns to this mapped dataframe
dynamic_subject_cols = [c for c in df_perf_wide.columns if c.startswith('Sem')]
for col in dynamic_subject_cols:
    temp_perf[col] = df_perf_wide[col]

mapped_dfs.append(temp_perf)

# ==========================================
# MAPPING THE REST OF THE DATASETS
# ==========================================
# Mapping Dataset 1: Research Student
map_research = {
    '10th percentage': 'Marks[10th]', '12th percentage': 'Marks[12th]',
    'CGPA': 'CGPA', 'No of backlogs': 'Current Back'
}
mapped_dfs.append(create_mapped_df(df_research, map_research))

# Mapping Dataset 3 & 4: Behavior
map_behav = {
    '10th percentage': '10th Mark', '12th percentage': '12th Mark',
    'study time': 'daily studing time', 'stress level': 'Stress Level ',
    'parttime job': 'part-time job', 'career preference': 'willingness to pursue a career based on their degree  '
}
mapped_dfs.append(create_mapped_df(pd.concat([df_behav1, df_behav2]), map_behav))

# Mapping Dataset 5: Lifestyle
map_lifestyle = {
    'study time': 'Study_Hours_Per_Day', 'CGPA': 'GPA', 'stress level': 'Stress_Level'
}
mapped_dfs.append(create_mapped_df(df_lifestyle, map_lifestyle))

# Mapping Dataset 6: Performance Data
map_perf_data = {
    'study time': 'StudyTimeWeekly', 'attendance': 'Absences', 'CGPA': 'GPA'
}
mapped_dfs.append(create_mapped_df(df_perf_data, map_perf_data))

# 3. Combine into one master DataFrame
# Pandas will automatically create the subject columns for all other datasets and fill them with NaN
master_df = pd.concat(mapped_dfs, ignore_index=True)

# 4. Clean up study time
def clean_study_time(val):
    if pd.isna(val): return np.nan
    val = str(val).lower().replace('minute', '').replace('minutes', '').replace('hour', '').replace('s', '').strip()
    if '-' in val:
        parts = val.split('-')
        try: return (float(parts[0]) + float(parts[1])) / 2
        except: return np.nan
    try: return float(val)
    except: return np.nan

master_df['study time'] = master_df['study time'].apply(clean_study_time)

# 5. Smart Imputation Preparation
label_encoders = {}
categorical_cols = ['stress level', 'parttime job', 'career preference']

for col in categorical_cols:
    mask = master_df[col].notna() & (master_df[col].astype(str).str.lower() != 'nan')
    le = LabelEncoder()
    encoded_values = le.fit_transform(master_df.loc[mask, col].astype(str))
    master_df[col] = np.nan 
    master_df.loc[mask, col] = encoded_values
    label_encoders[col] = le

master_df = master_df.apply(pd.to_numeric, errors='coerce')

# 6. Apply Smart Filler (MICE Algorithm)
print(f"Applying Machine Learning Imputation across {len(master_df.columns)} columns... (This will take a moment)")
imputer = IterativeImputer(max_iter=10, random_state=42, sample_posterior=True)
imputed_data = imputer.fit_transform(master_df)

final_df = pd.DataFrame(imputed_data, columns=master_df.columns)

# 7. Post-Processing & Decoding
# Decode categoricals
for col in categorical_cols:
    le = label_encoders[col]
    rounded_vals = np.clip(np.round(final_df[col]), 0, len(le.classes_) - 1).astype(int)
    final_df[col] = le.inverse_transform(rounded_vals)

# Clean Numerical columns
final_df['No of backlogs'] = np.round(np.clip(final_df['No of backlogs'], 0, None)).astype(int)
final_df['CGPA'] = np.clip(np.round(final_df['CGPA'], 2), 0, 10) 
final_df['SGPA'] = np.clip(np.round(final_df['SGPA'], 2), 0, 10)
final_df['10th percentage'] = np.clip(np.round(final_df['10th percentage'], 2), 0, 100)
final_df['12th percentage'] = np.clip(np.round(final_df['12th percentage'], 2), 0, 100)
final_df['attendance'] = np.clip(np.round(final_df['attendance'], 0), 0, 100)
final_df['semester academic period'] = np.round(np.clip(final_df['semester academic period'], 1, 8)).astype(int)

# Clean up dynamically generated subject marks (Assuming Grade Points operate on a 0-10 scale)
for col in dynamic_subject_cols:
    if col in final_df.columns:
        final_df[col] = np.clip(np.round(final_df[col], 1), 0, 10)

# 8. Overwrite ID Column (1, 2, 3, 4...)
final_df.insert(0, 'ID', range(1, len(final_df) + 1))

# 9. Save
output_filename = "Smart_Merged_Subject_Data.csv"
final_df.to_csv(output_filename, index=False)
print(f"Success! Final dataset with dynamic subject columns saved as: {output_filename}")