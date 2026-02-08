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

st.title("📚 Lib-Count: Library Usage Prediction System")
st.markdown("Hackathon 3 Project | Predictive Analytics for University Library")

# Tabs for Navigation
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data & Analysis", "⚙️ Model Training", "🔮 Prediction", "📝 Logs"])

# --- TAB 1: DATA ---
with tab1:
    st.header("Data Management")
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

# --- TAB 2: TRAINING ---
with tab2:
    st.header("Model Engine")
    st.write("Train Linear Regression, Decision Tree, and Random Forest models. The system will automatically select the best performing one.")
    
    if st.button("Train Models"):
        df = load_data_from_db()
        if not df.empty:
            with st.spinner("Training and Tuning Hyperparameters..."):
                logs = train_and_evaluate(df)
            st.success(f"Training Complete! Best Model: {logs['best_model']}")
            st.json(logs)
        else:
            st.error("No data to train on.")

# --- TAB 3: PREDICTION ---
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
        st.metric(label="Predicted Students in Library", value=result)
        
        if result > 100:
            st.warning("⚠️ High Traffic Expected! Arrange extra chairs.")
        else:
            st.info("Normal Traffic.")

# --- TAB 4: LOGS ---
with tab4:
    st.header("System Logs")
    try:
        with open('models/model_metrics.json', 'r') as f:
            logs = json.load(f)
            st.json(logs)
    except FileNotFoundError:
        st.write("No training logs available yet.")