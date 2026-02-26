import joblib
import sys
import pandas as pd
import sklearn

print(f"Python version: {sys.version}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")

try:
    print("Loading model...")
    model = joblib.load('rf_model.joblib')
    print("Model loaded.")
    
    print("Loading scaler...")
    scaler = joblib.load('scaler.joblib')
    print("Scaler loaded.")
    
    print("Loading encoders...")
    encoders = joblib.load('label_encoders.joblib')
    print("Encoders loaded.")
    
    print("Loading metadata...")
    metadata = joblib.load('model_metadata.joblib')
    print("Metadata loaded.")
    
    print("All artifacts loaded successfully!")
except Exception as e:
    print(f"FAILED to load artifacts: {e}")
    import traceback
    traceback.print_exc()
