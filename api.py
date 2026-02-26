import pandas as pd
import numpy as np
import joblib
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI(title="Attrition Prediction API")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Artifacts
def load_artifacts():
    try:
        model = joblib.load('rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        encoders = joblib.load('label_encoders.joblib')
        metadata = joblib.load('model_metadata.joblib')
        return model, scaler, encoders, metadata
    except FileNotFoundError:
        return None, None, None, None

model, scaler, encoders, metadata = load_artifacts()

class EmployeeInput(BaseModel):
    Age: int
    Gender: str
    MaritalStatus: str
    DistanceFromHome: int
    Department: str
    JobRole: str
    JobLevel: int
    BusinessTravel: str
    MonthlyIncome: float
    OverTime: str
    JobSatisfaction: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsSinceLastPromotion: int

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.get("/metadata")
def get_metadata():
    if metadata:
        # Return options for dropdowns
        return {
            "options": metadata['cat_options'],
            "defaults": metadata.get('defaults', {})
        }
    return {"error": "Metadata not loaded"}

@app.post("/predict")
def predict_churn(data: EmployeeInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model artifacts not found")
    
    # Create DataFrame from input
    input_dict = data.dict()
    
    # Prepare full input with defaults for missing columns
    full_input = metadata['defaults'].copy()
    full_input.update(input_dict)
    
    input_df = pd.DataFrame([full_input])
    # Ensure column order
    input_df = input_df[metadata['columns']]
    
    # Preprocessing
    encoded_df = input_df.copy()
    
    # Encode
    try:
        for col, le in encoders.items():
            if col in encoded_df.columns:
                val = encoded_df.iloc[0][col]
                # Handle unknown categories
                if val in le.classes_:
                    encoded_df[col] = le.transform([val])
                else:
                    encoded_df[col] = le.transform([le.classes_[0]])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Encoding error: {str(e)}")
        
    # Scale
    try:
        num_cols = metadata['numerical_cols']
        encoded_df[num_cols] = scaler.transform(encoded_df[num_cols])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scaling error: {str(e)}")
        
    # Predict
    try:
        prediction = int(model.predict(encoded_df)[0])
        probability = float(model.predict_proba(encoded_df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
    # SHAP Explanation
    explanation = []
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(encoded_df)
        
        # Debug Prints
        print(f"SHAP Values Type: {type(shap_values)}")
        if isinstance(shap_values, list):
             print(f"SHAP List Len: {len(shap_values)}")
             print(f"SHAP[0] Shape: {shap_values[0].shape}")
        else:
             print(f"SHAP Shape: {shap_values.shape}")

        if isinstance(shap_values, list):
            # For binary classification, index 1 is usually the positive class
            sv = shap_values[1][0]
        else:
            # SHAP returns (n_samples, n_features, n_classes) for RF
            if len(shap_values.shape) == 3:
                # Shape (1, 30, 2) -> We want sample 0, all features, class 1
                sv = shap_values[0, :, 1]
            elif len(shap_values.shape) == 2:
                # Shape (1, 30) or (30, 2)? 
                if shap_values.shape[0] == 1:
                     sv = shap_values[0]
                else:
                     # Fallback
                     sv = shap_values[:, 1]
            else:
                sv = shap_values

        # Top 3 factors
        indices = np.argsort(np.abs(sv))[::-1][:3]
        feature_names = encoded_df.columns
        
        for idx in indices:
            feat = feature_names[idx]
            impact = sv[idx]
            val = input_dict.get(feat, full_input[feat])
            
            explanation.append({
                "feature": feat,
                "value": str(val),
                "impact": "increase" if impact > 0 else "decrease",
                "importance": float(abs(impact))
            })
        print(f"Generated Explanation: {explanation}")
            
    except Exception as e:
        print(f"SHAP Error Details: {e}")
        import traceback
        traceback.print_exc()
        
    # Determine Risk Label
    if probability >= 0.50:
        label = "High Risk"
    elif probability >= 0.30:
        label = "Potential Risk"
    else:
        label = "Low Risk"

    return {
        "prediction": label,
        "probability": probability,
        "explanation": explanation
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
