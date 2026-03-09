import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # Added XGBoost
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

MODEL_PATH = 'models/best_model.pkl'
METRICS_PATH = 'models/model_metrics.json'

def train_and_evaluate(df):
    """
    Trains 4 models with hyperparameter tuning and evaluates using MAE, RMSE, and MAPE.
    """
    # --- 1. PREPROCESSING PIPELINE ---
    # Mapping features to ensure consistency with app.py inputs
    X = df[['DayOfWeek', 'IsHoliday', 'IsExamWeek', 'IsLibrarianPresent', 'TotalCampusStudents']]
    y = df['LibraryStudentCount']

    # 80/20 Split to ensure the model generalizes to new library data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 2. MODEL SELECTION & HYPERPARAMETERS ---
    models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'Random Forest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [10, 20]
            }
        },
        'XGBoost': {  # New Model Added
            'model': XGBRegressor(objective='reg:squarederror'),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6]
            }
        }
    }

    best_model_name = ""
    best_rmse = np.inf # Using RMSE as the primary selector for "Best Model"
    best_model_obj = None
    all_metrics = {}

    print("Starting Training Pipeline with expanded metrics...")
    
    for name, config in models.items():
        # Hyperparameter Tuning
        clf = GridSearchCV(config['model'], config['params'], cv=3, scoring='neg_mean_squared_error')
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        
        # --- 3. ADVANCED EVALUATION METRICS ---
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Root Mean Squared Error
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100 # Mean Absolute Percentage Error
        r2 = r2_score(y_test, y_pred)
        
        all_metrics[name] = {
            'MAE (Students)': round(mae, 2),
            'RMSE (Penalty)': round(rmse, 2),
            'MAPE (%)': f"{round(mape, 2)}%",
            'R2_Score': round(r2, 3)
        }
        
        # Select best model based on lowest RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model_obj = clf.best_estimator_

    # --- 4. LIFECYCLE MANAGEMENT (Saving) ---
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model_obj, f)

    final_log = {
        "best_model": best_model_name,
        "leaderboard": all_metrics, # Renamed for clarity in the dashboard
        "last_trained": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(final_log, f, indent=4)
        
    # Convert the leaderboard dictionary into a DataFrame for Streamlit
    leaderboard_df = pd.DataFrame.from_dict(all_metrics, orient='index').reset_index()
    leaderboard_df.rename(columns={'index': 'Model'}, inplace=True)
        
    # THE FIX: Return TWO values to match app.py
    return leaderboard_df, best_model_name

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def make_prediction(day_of_week, is_holiday, is_exam_week, is_librarian, total_students):
    model = load_model()
    if not model:
        return "Model not trained yet."
    
    # Matching feature names used in training
    feature_names = ['DayOfWeek', 'IsHoliday', 'IsExamWeek', 'IsLibrarianPresent', 'TotalCampusStudents']
    input_df = pd.DataFrame([[day_of_week, is_holiday, is_exam_week, is_librarian, total_students]], 
                            columns=feature_names)
    
    prediction = model.predict(input_df)
    return int(max(0, prediction[0])) # Ensure we don't predict negative students