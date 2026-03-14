import pyrebase
import pandas as pd
from sqlalchemy import create_engine
import time

# --- 1. PASTE YOUR CONFIG FROM STEP 1 HERE ---
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
print("Initializing Firebase...")
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

# --- 2. CONNECT TO YOUR POSTGRESQL DATABASE ---
print("Connecting to PostgreSQL...")
engine = create_engine("postgresql://postgres:0000@localhost:5433/student_analytics_db")

# Fetch all student IDs from your fact table
df = pd.read_sql("SELECT id FROM fact_student_performance", engine)
student_ids = df['id'].tolist()

print(f"Found {len(student_ids)} students in PostgreSQL. Starting Registration...")

# --- 3. CREATE ACCOUNTS ---
success_count = 0

for student_id in student_ids:
    # Generate a standard email and password for each student
    email = f"student{student_id}@college.edu"
    password = f"Pass@{student_id}2026" 
    
    try:
        # A. Create the user in Firebase Authentication
        user = auth.create_user_with_email_and_password(email, password)
        
        # B. Save their role and ID to the Realtime Database
        data = {
            "role": "student",
            "access_id": int(student_id),
            "email": email
        }
        db.child("users").child(user['localId']).set(data)
        
        success_count += 1
        print(f"✅ Registered: {email}")
        
        # Brief pause so Firebase doesn't block us for spamming requests
        time.sleep(0.5) 
        
    except Exception as e:
        # If the email already exists, it will catch the error here and continue
        print(f"⚠️ Skipped ID {student_id} (Might already exist)")

print(f"\n🎉 Finished! Successfully registered {success_count} students.")