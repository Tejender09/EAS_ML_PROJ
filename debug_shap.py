import joblib
import shap
import pandas as pd
import numpy as np

try:
    model = joblib.load('rf_model.joblib')
    # Create a dummy input matching the API
    meta_data = joblib.load('model_metadata.joblib')
    defaults = meta_data['defaults']
    
    # Create a single row DF from defaults
    input_df = pd.DataFrame([defaults])
    input_df = input_df[meta_data['columns']]
    
    # Encode and scale (rough approximation, using raw defaults might fail encoding if not handled)
    # Actually, we need to load encoders too to be precise, but let's just test the SHAP explainer on the raw model + some random data of correct shape
    # Wait, the model expects encoded/scaled data.
    
    # Let's just load X_train sample
    X_train = joblib.load('X_train.joblib')
    
    # Take one row
    sample = X_train.iloc[[0]]
    
    print(f"Sample shape: {sample.shape}")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    
    print(f"SHAP Type: {type(shap_values)}")
    print(f"SHAP Shape: {shap_values.shape}")
    if isinstance(shap_values, list):
        print(f"List len: {len(shap_values)}")
        print(f"Item 0 shape: {shap_values[0].shape}")
        
    print("Execution complete.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
