import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
import shap

# 1. Load the Dataset
# Please ensure 'WA_Fn-UseC_-HR-Employee-Attrition.csv' is in the same directory.
try:
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset 'WA_Fn-UseC_-HR-Employee-Attrition.csv' not found.")
    print("Please download it (e.g., from Kaggle) and place it in the same directory.")
    exit()

# 2. Data Preprocessing

# Drop irrelevant columns as per requirements
columns_to_drop = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Define Target Variable
# Attrition: Yes = 1, No = 0
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Separate Features and Target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Identify Categorical and Numerical Columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Handle Categorical Features using Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit and transform, replacing the column in X
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale Numerical Features using StandardScaler
# We only scale the original numerical columns, not the label encoded ones (though for RF it matters less, for LR it's better)
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Train-Test Split with Stratification
# Stratify=y ensures the class distribution is preserved in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 3. Modeling

# Baseline Model: Logistic Regression
# Class_weight='balanced' handles the class imbalance (Attrition is usually the minority class)
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1]

# Main Model: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# 4. Evaluation

def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n--- {name} Performance ---")
    print(classification_report(y_true, y_pred))
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

evaluate_model("Logistic Regression (Baseline)", y_test, lr_preds, lr_probs)
evaluate_model("Random Forest (Main)", y_test, rf_preds, rf_probs)

# 4.1 Feature Importance Analysis
print("\n--- Top 10 Key Features (Random Forest) ---")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

for f in range(10):
    print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")


# 5. Explainability with SHAP

print("\nGenerating SHAP Summary Plot...")
# Initialize SHAP explainer for Random Forest
# TreeExplainer is optimized for tree-based models
explainer = shap.TreeExplainer(rf_model)
# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Handling SHAP values structure for binary classification
# shap_values[1] corresponds to the positive class (Attrition=1)
if isinstance(shap_values, list):
    shap_values_target = shap_values[1]
else:
    shap_values_target = shap_values

# Generate Summary Plot
plt.figure()
shap.summary_plot(shap_values_target, X_test, show=False)
plt.title("SHAP Summary Plot for Random Forest (Attrition Features)")
plt.tight_layout()
plt.savefig('shap_summary_plot.png')
print("SHAP summary plot saved as 'shap_summary_plot.png'.")
# plt.show() # Commented out to run non-interactively if needed
