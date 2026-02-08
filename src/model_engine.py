import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_PATH = 'models/best_model.pkl'
METRICS_PATH = 'models/model_metrics.json'

def train_and_evaluate(df):
    """
    Trains 3 models, tunes hyperparameters, and saves the best one.
    """
    # Feature Selection
    X = df[['DayOfWeek', 'IsHoliday', 'IsExamWeek', 'IsLibrarianPresent', 'TotalCampusStudents']]
    y = df['LibraryStudentCount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            }
        }
    }

    best_model_name = ""
    best_score = -np.inf
    best_model_obj = None
    all_metrics = {}

    print("Starting training...")
    
    for name, config in models.items():
        # Hyperparameter Tuning using GridSearch
        clf = GridSearchCV(config['model'], config['params'], cv=3, scoring='r2')
        clf.fit(X_train, y_train)
        
        # Evaluation
        y_pred = clf.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        all_metrics[name] = {'MAE': round(mae, 2), 'R2_Score': round(r2, 3), 'Best_Params': clf.best_params_}
        
        if r2 > best_score:
            best_score = r2
            best_model_name = name
            best_model_obj = clf.best_estimator_

    # Save Best Model
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model_obj, f)

    # Save Metrics
    final_log = {
        "best_model": best_model_name,
        "metrics": all_metrics,
        "last_trained": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(final_log, f, indent=4)
        
    return final_log

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def make_prediction(day_of_week, is_holiday, is_exam_week, is_librarian, total_students):
    model = load_model()
    if not model:
        return "Model not trained yet."
    
    # Ensure input shape matches training shape
    input_data = [[day_of_week, is_holiday, is_exam_week, is_librarian, total_students]]
    prediction = model.predict(input_data)
    return int(prediction[0])