import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def generate_feature_report():
    print("Loading Model Artifacts...")
    try:
        model = joblib.load('rf_model.joblib')
        metadata = joblib.load('model_metadata.joblib')
        X_train = joblib.load('X_train.joblib') # Load training data to get feature names correct
    except FileNotFoundError:
        print("Error: Model artifacts not found. Run train_model.py first.")
        return

    # Extract Feature Importance
    importances = model.feature_importances_
    feature_names = metadata['columns'] # Columns used in training
    
    # Create a DataFrame for nice formatting
    # Ensure feature_names matches importances length
    # Note: X_encoded columns in train_model might differ if OneHot was used, but we used LabelEncoder, so column count is preserved 1:1 usually.
    # Let's verify lengths.
    
    # In train_model loop:
    # X_encoded = X.copy() ... le.fit_transform ...
    # So X_encoded has same columns as X.
    
    if len(importances) != len(feature_names):
        print(f"Warning: Feature count mismatch. Model has {len(importances)}, Metadata has {len(feature_names)}")
        # We might need to look at X_train columns if available
        # But actually X_train is a numpy array in the split? No, it's typically a dataframe if split wasn't converting it, 
        # but train_test_split on dataframe usually preserves it. 
        # Wait, StandardScaler converts to numpy array.
        # So X_train is likely a numpy array.
        # However, metadata['columns'] should be the list of columns in X passed to the encoder/scaler.
        pass

    feature_data = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance
    feature_data = feature_data.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # Generate Report Text
    report_lines = []
    report_lines.append("# Employee Attrition Model - Feature Importance Report")
    report_lines.append("=========================================================\n")
    report_lines.append("This report outlines which employee factors drive the AI's decision-making process, ranked by influence.\n")
    
    report_lines.append("## Top 10 Most Influential Factors (The 'Big Drivers')")
    report_lines.append("These features have the massive impact on whether an employee is flagged as High Risk.\n")
    
    for rank, row in feature_data.head(10).iterrows():
        name = row['Feature']
        score = row['Importance']
        # Add human-friendly context based on general HR knowledge for this dataset
        context = get_feature_context(name)
        report_lines.append(f"{rank + 1}. **{name}** (Impact Score: {score:.4f})")
        report_lines.append(f"   - *What it means*: {context}\n")

    report_lines.append("\n## Moderate Influencers (The 'Context' Builders)")
    report_lines.append("These features provide context to the main drivers.\n")
    
    for rank, row in feature_data.iloc[10:20].iterrows():
         report_lines.append(f"{rank + 1}. {row['Feature']} ({row['Importance']:.4f})")

    report_lines.append("\n## Low Impact Features (The 'Noise')")
    report_lines.append("These factors are rarely the deciding reason for attrition in this specific model.\n")
    
    for rank, row in feature_data.iloc[20:].iterrows():
         report_lines.append(f"{rank + 1}. {row['Feature']} ({row['Importance']:.4f})")

    # Write to file
    with open('feature_importance_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    print("Report generated: feature_importance_report.txt")
    print("\n--- TOP 5 PREVIEW ---")
    print(feature_data.head(5))

def get_feature_context(feature_name):
    # Dictionary of simple explanations for the IBM dataset features
    explanations = {
        'MonthlyIncome': "Financial stability is often the #1 retainer. Lower income relative to role increases risk.",
        'OverTime': "Burnout indicator. Employees working overtime are significantly more likely to quit.",
        'Age': "Career stage proxy. Younger employees explore; older employees settle. High predictive power.",
        'TotalWorkingYears': "Experience level. Senior staff are generally more stable than freshers.",
        'DistanceFromHome': "Commute stress. Long commutes correlate with higher attrition.",
        'YearsAtCompany': "Loyalty/Stability metric. The 'Lifer' effect—longer tenure drastically reduces risk.",
        'HourlyRate': "Granular pay metric. Surprisingly high importance in some Random Forest splits.",
        'DailyRate': "Daily pay rate (similar to HourlyRate/MonthlyIncome).",
        'MonthlyRate': "Another monthly pay metric distinct from Income.",
        'NumCompaniesWorked': "Job hopping history. People who have changed jobs often are likely to do it again.",
        'StockOptionLevel': "Golden handcuffs. Vesting stock is a powerful retention tool.",
        'JobSatisfaction': "Engagement score. Unhappy people leave, but 'content' people might stay depending on other factors."
    }
    return explanations.get(feature_name, "Quantitative metric affecting employee profile.")

if __name__ == "__main__":
    generate_feature_report()
