import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_generator import generate_library_data
from src.db_manager import save_data_to_db, load_data_from_db
from src.model_engine import train_and_evaluate, make_prediction

# Page Config
st.set_page_config(page_title="Lib-Count Dashboard", layout="wide")

st.title("Lib-Count: Library Usage Prediction System")
st.markdown("Hackathon 3 Project | Predictive Analytics for University Library")

# Sidebar for Info and Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/library.png") # Or a local logo
    st.title("Admin Control Panel")
    st.info(f"Target Campus Population: 350 Students") # Using your university data
    st.markdown("---")


# Tabs for Navigation
tab1, tab2, tab3, tab4 = st.tabs(["Data & Analysis", " Model Training", " Prediction", " Logs"])

#  TAB 1: DATA 
with tab1:
    st.header("Data Management & Insights")
    # NEW: Summary Metrics for a quick "Health Check"
    df = load_data_from_db()
    if not df.empty:
        m1, m2, m3 = st.columns(3)
        avg_attendance = df['LibraryStudentCount'].mean()
        peak_day_idx = df.groupby('DayOfWeek')['LibraryStudentCount'].mean().idxmax()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        m1.metric("Avg. Daily Attendance", f"{int(avg_attendance)} Students")
        m2.metric("Busiest Day", days[peak_day_idx])
        m3.metric("Campus Capacity", "350", "Total") # Grounded in university data
        st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate & Reset Database (Simulate New Data)"):
            with st.spinner("Generating 2 years of data..."):
                df = generate_library_data()
                save_data_to_db(df)
            st.success("Data Generated and Saved to SQL Database!")

    df = load_data_from_db()
    
    if not df.empty:
        st.dataframe(df.head())
        
        st.subheader("Data Visualization")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Library Usage by Day of Week**")
            fig1, ax1 = plt.subplots()
            sns.barplot(x='DayOfWeek', y='LibraryStudentCount', data=df, ax=ax1, palette="viridis")
            ax1.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            st.pyplot(fig1)

        with col_b:
            st.markdown("**Impact of Exam Weeks**")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x='IsExamWeek', y='LibraryStudentCount', data=df, ax=ax2)
            ax2.set_xticklabels(['Normal Week', 'Exam Week'])
            st.pyplot(fig2)
    else:
        st.warning("No data found. Please generate data first.")


#  TAB 2: MODEL TRAINING OF app.py
with tab2:
    st.header("Model Performance & Evaluation")
    st.write("Compare different algorithms to find the most accurate attendance predictor.")

    if st.button("Train Models"):
        df = load_data_from_db()
        if not df.empty:
            with st.spinner("Training & Evaluating with RMSE, MAE, MAPE..."):
                # This function returns your results
                results_df, best_model_name = train_and_evaluate(df) 
            
            # 1. Success Message
            st.success(f" Best Model Identified: {best_model_name}")
            
            # 2. Results Table with styling
            st.subheader("Model Leaderboard")
            st.dataframe(results_df) 
            st.write("Columns found in results:", results_df.columns.tolist())

            # 3. Explanatory Expander (The "Brain" of the dashboard)
            with st.expander("What do these metrics mean?!"):
                st.markdown("""
                - **RMSE (Root Mean Squared Error):** This is our primary 'penalty' score. It shows how many students we are 'off' by on average.
                - **MAE (Mean Absolute Error):** This is the average absolute difference between predicted and actual attendance.
                - **MAPE:** This shows the error as a percentage of the total student count.
                """)
        else:
            st.warning("Please generate data in Tab 1 before training!")


#  TAB 3: PREDICTION 
with tab3:
    st.header("Real-time Prediction")
    st.write("Enter daily parameters to predict library student count.")
    
    col1, col2 = st.columns(2)
    with col1:
        day = st.selectbox("Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
        total_students = st.slider("Total Students on Campus", 0, 350, 300)
        librarian = st.selectbox("Librarian Present?", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    
    with col2:
        is_holiday = st.checkbox("Is it a Holiday?")
        is_exam = st.checkbox("Is it Exam Week?")
    

if st.button("Predict Headcount"):
        result = make_prediction(day, int(is_holiday), int(is_exam), librarian, total_students)
        
        # 1. Visual Display of the Prediction
        st.markdown("---")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(label="Predicted Students in Library", value=f"{int(result)} / 350")
        
        with col_res2:
            # Calculate occupancy percentage
            occupancy = (result / 350) * 100
            st.metric(label="Estimated Occupancy", value=f"{int(occupancy)}%")

        # 2. Smart Recommendations based on the prediction
        st.subheader("Administrative Recommendations")
        
        if result > 120:
            st.error("**High Traffic Alert**")
            st.write("""
            - **Seating:** Open the overflow study hall.
            - **Environment:** Ensure the Air Conditioning is set to maximum (Room H510-207 standards!).
            - **Staffing:** Deploy an extra student assistant at the entry gate.
            """)
        elif is_exam:
            st.warning("**Exam Season Protocol**")
            st.write("""
            - **Quiet Zones:** Enforce 'No-Talk' zones in the main wing.
            - **Resources:** Check if charging ports and Wi-Fi routers are handling the load.
            """)
        else:
            st.success("**Normal Operations**")
            st.write("- Standard maintenance and lighting schedules apply.")


#  TAB 4: LOGS
with tab4:
    st.header("System Logs")
    try:
        with open('models/model_metrics.json', 'r') as f:
            logs = json.load(f)
            st.json(logs)
    except FileNotFoundError:
        st.write("No training logs available yet.")


        # --- FOOTER & SYSTEM STATUS ---
st.markdown("---")
footer_col1, footer_col2 = st.columns([3, 1])

with footer_col1:
    st.caption("© 2026 Lib-Count Predictive Systems")
    st.caption("Data sources: University Student Portal & Library Entry Logs")

with footer_col2:
    # A small 'live' indicator
    st.success("System Online")
    