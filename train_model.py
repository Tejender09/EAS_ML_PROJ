import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_model():
    print("Loading Dataset...")
    try:
        df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    except FileNotFoundError:
        print("Error: Dataset 'WA_Fn-UseC_-HR-Employee-Attrition.csv' not found.")
        return

    # 1. Preprocessing
    print("Preprocessing Data...")
    
    # Drop irrelevant columns
    columns_to_drop = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # Target Variable
    if 'Attrition' not in df.columns:
        print("Column 'Attrition' not found.")
        return
        
    # Map Attrition only if needed (it usually is 'Yes'/'No' in this dataset)
    if df['Attrition'].dtype == 'object':
         df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Identify columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Calculate defaults for filling missing user inputs
    defaults = {}
    for col in X.columns:
        if col in categorical_cols:
            defaults[col] = X[col].mode()[0]
        else:
            defaults[col] = X[col].mean()

    # Save metadata for the app
    # This dictionary will store column names, types, unique values, and defaults
    meta_data = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'cat_options': {col: X[col].unique().tolist() for col in categorical_cols},
        'columns': X.columns.tolist(),
        'defaults': defaults
    }
    joblib.dump(meta_data, 'model_metadata.joblib')
    print("Metadata (with defaults) saved to 'model_metadata.joblib'")

    # Encode Categorical Features
    label_encoders = {}
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    joblib.dump(label_encoders, 'label_encoders.joblib')
    print("Label Encoders saved to 'label_encoders.joblib'")
        
    # Scale Numerical Features
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
    
    joblib.dump(scaler, 'scaler.joblib')
    print("Scaler saved to 'scaler.joblib'")
    
    # Split Data (Optional for training final model, but good practice to validate)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest with Optimization
    print("Training Random Forest Classifier with Grid Search...")
    # We prioritize 'recall' because in attrition, missing a leaver (False Negative) is worse than flagging a stayer (False Positive).
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    rf_model = grid_search.best_estimator_
    print(f"Best Parameters found: {grid_search.best_params_}")

    # Save Model
    joblib.dump(rf_model, 'rf_model.joblib')
    print("Random Forest Model saved to 'rf_model.joblib'")

    # Save X_train for SHAP explainer baseline (optional but good for TreeExplainer)
    # We'll save a small sample for SHAP background if needed, though TreeExplainer doesn't strictly need it always, 
    # passing data to it is good.
    joblib.dump(X_train, 'X_train.joblib')
    print("Training data sample saved for SHAP.")

    print("Training Complete. All artifacts saved.")

if __name__ == "__main__":
    train_model()
