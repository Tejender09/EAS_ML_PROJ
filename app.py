import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px  # Adding plotly for better charts

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Attrition Risk Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, card-based look
st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #f0f2f6;
    }
    p, div, label, span {
        font-family: 'Inter', sans-serif;
        color: #c9d1d9;
    }
    
    /* Card Container */
    .stCard {
        background-color: #161b22;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 24px;
    }
    
    /* Result Cards */
    .result-card-high {
        background: linear-gradient(135deg, #4a1c1c 0%, #2a0f0f 100%);
        border: 1px solid #ff4b4b;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.2);
    }
    .result-card-low {
        background: linear-gradient(135deg, #1c4a2e 0%, #0f2a1a 100%);
        border: 1px solid #2ecc71;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(46, 204, 113, 0.2);
    }
    
    /* Metrics */
    .metric-value {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 10px 0;
        color: #ffffff;
    }
    .metric-label {
        font-size: 1.1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOAD TRAINED ARTIFACTS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        encoders = joblib.load('label_encoders.joblib')
        metadata = joblib.load('model_metadata.joblib')
        # Check defaults existence
        if 'defaults' not in metadata:
            st.error("Model metadata missing default values. Please re-run training.")
            st.stop()
        return model, scaler, encoders, metadata
    except FileNotFoundError as e:
        st.error(f"System Error: Artifacts not found ({e}). Run train_model.py first.")
        st.stop()

model, scaler, encoders, metadata = load_artifacts()

# -----------------------------------------------------------------------------
# 3. SIDEBAR: EMPLOYEE INPUTS
# -----------------------------------------------------------------------------
with st.sidebar:

    st.title("Employee Profile")
    st.markdown("---")
    
    # Section 1: Personal
    st.subheader("Demographics")
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", metadata['cat_options'].get('Gender', ['Male', 'Female']))
    marital = st.selectbox("Marital Status", metadata['cat_options'].get('MaritalStatus', ['Single', 'Married', 'Divorced']))
    dist = st.slider("Distance From Home (km)", 1, 30, 5)
    
    # Section 2: Job Details
    st.subheader("Job Details")
    dept = st.selectbox("Department", metadata['cat_options'].get('Department', []))
    role = st.selectbox("Job Role", metadata['cat_options'].get('JobRole', []))
    level = st.slider("Job Level", 1, 5, 1)
    travel = st.selectbox("Business Travel", metadata['cat_options'].get('BusinessTravel', []))
    
    # Section 3: Compensation & Satisfaction
    st.subheader("Work Factors")
    # INR Conversion: Model trained on USD. We accept INR and convert for prediction.
    # Assumed rate: 1 USD = 83 INR
    income_inr = st.number_input("Monthly Income (₹)", min_value=10000, value=50000, step=5000)
    income = income_inr / 83 # Convert back to USD scale for model
    
    overtime = st.selectbox("OverTime", metadata['cat_options'].get('OverTime', ['No', 'Yes']))
    satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=3,
                                    help="Score from latest Employee Engagement Survey (1=Low, 4=High)")
    wlb = st.select_slider("Work Life Balance", options=[1, 2, 3, 4], value=3,
                           help="Score from latest Performance Review or Pulse Survey (1=Bad, 4=Good)")
    
    # Section 4: Experience
    st.subheader("Tenure")
    years_at_co = st.slider("Years at Company", 0, 40, 5)
    years_promo = st.slider("Years Since Last Promotion", 0, 20, 1)
    
    st.markdown("---")
    analyze_btn = st.button("Analyze Risk Profile", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 4. MAIN APP LOGIC
# -----------------------------------------------------------------------------

st.title("Employee Attrition Risk Prediction")


if analyze_btn:
    # A. PREPARE DATA
    # Start with defaults for all hidden columns (DailyRate, etc.)
    input_data = metadata['defaults'].copy()
    
    # Update with User Inputs
    input_data.update({
        'Age': age,
        'Gender': gender,
        'MaritalStatus': marital,
        'DistanceFromHome': dist,
        'Department': dept,
        'JobRole': role,
        'JobLevel': level,
        'BusinessTravel': travel,
        'MonthlyIncome': income,
        'OverTime': overtime,
        'JobSatisfaction': satisfaction,
        'WorkLifeBalance': wlb,
        'YearsAtCompany': years_at_co,
        'YearsSinceLastPromotion': years_promo
    })
    
    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[metadata['columns']]
    
    # B. ENCODING
    encoded_df = input_df.copy()
    try:
        for col, le in encoders.items():
            if col in encoded_df.columns:
                # Handle unknown categories gracefully if possible, else default to mode
                val = encoded_df.iloc[0][col]
                if val in le.classes_:
                    encoded_df[col] = le.transform([val])
                else:
                    # Fallback for safe inference
                    encoded_df[col] = le.transform([le.classes_[0]])
    except Exception as e:
        st.error(f"Preprocessing Error: {e}")
        st.stop()

    # C. SCALING
    try:
        num_cols = metadata['numerical_cols']
        encoded_df[num_cols] = scaler.transform(encoded_df[num_cols])
    except Exception as e:
        st.error(f"Scaling Error: {e}")
        st.stop()

    # D. PREDICTION
    try:
        prediction = model.predict(encoded_df)[0]
        prob = model.predict_proba(encoded_df)[0][1] # Probability of Attrition (1)
    except Exception as e:
        st.error(f"Model Inference Error: {e}")
        st.stop()

    # E. DISPLAY RESULTS
    st.divider()
    
    # Layout: Top Row (Result Card + Gauge), Bottom Row (Feature Importance)
    col_res, col_gauge = st.columns([1, 1])
    
    with col_res:
        if prediction == 1:
            # High Risk
            st.markdown(f"""
            <div class="result-card-high">
                <h3 style="color:#ffcccc; margin:0;">High Attrition Risk</h3>
                <div class="metric-value">{prob:.1%}</div>
                <div class="metric-label" style="color:#ffcccc;">Probability of Leaving</div>
                <hr style="border-color: #ffcccc; opacity: 0.3; margin: 15px 0;">
                <p style="color: #f0f0f0;">
                    <strong>Recommendation:</strong><br>
                    Immediate intervention recommended. Review engagement and satisfaction metrics.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Low Risk
            st.markdown(f"""
            <div class="result-card-low">
                <h3 style="color:#ccffdd; margin:0;">Low Attrition Risk</h3>
                <div class="metric-value">{prob:.1%}</div>
                <div class="metric-label" style="color:#ccffdd;">Probability of Leaving</div>
                <hr style="border-color: #ccffdd; opacity: 0.3; margin: 15px 0;">
                <p style="color: #f0f0f0;">
                    <strong>Recommendation:</strong><br>
                    Employee is likely stable. Maintain current career development path.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_gauge:
        # Create a Gauge Chart using Plotly
        fig_gauge = px.bar(
            x=[prob], 
            y=["Risk"], 
            orientation='h', 
            range_x=[0, 1],
            text=[f"{prob:.1%}"],
            color=[prob],
            color_continuous_scale=['#2ecc71', '#f1c40f', '#e74c3c']
        )
        fig_gauge.update_layout(
            title="Risk Confidence Meter",
            xaxis_title="Probability",
            yaxis_title="",
            showlegend=False,
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )
        fig_gauge.update_xaxes(showgrid=False, tickfont=dict(color='white'))
        fig_gauge.update_yaxes(showticklabels=False)
        fig_gauge.update_yaxes(showticklabels=False)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Placeholder for AI Justification (to be filled after calculation)
    justification_placeholder = st.empty()


    # -------------------------------------------------------------------------
    # F. AI JUSTIFICATION
    # -------------------------------------------------------------------------


    # Feature Importance (SHAP)
    st.markdown("### Model Interpretation")
    st.markdown("<p style='font-size:0.9em; opacity:0.8;'>What factors drove this prediction?</p>", unsafe_allow_html=True)
    
    with st.spinner("Calculating feature importance..."):
        # SHAP Calculation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(encoded_df)
        
        # Handle list output for binary classification
        # Handle SHAP output structure
        # Debug output showed shape (1, 30, 2) -> (samples, features, classes)
        if len(shap_values.shape) == 3:
            # (1, 30, 2) -> We want sample 0, all features, class 1 (Attrition)
            sv = shap_values[0, :, 1]
        elif isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
            
        sv = np.array(sv).flatten()

        # Create DataFrame for Plotting
        df_shap = pd.DataFrame({
            "Feature": encoded_df.columns,
            "Impact": sv
        })
        df_shap['AbsImpact'] = df_shap['Impact'].abs()
        df_shap = df_shap.sort_values(by="AbsImpact", ascending=False).head(10)
        
        # Color coding
        df_shap['Type'] = df_shap['Impact'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
        color_map = {'Increases Risk': '#ff4b4b', 'Decreases Risk': '#2ecc71'}
        
        # Plot
        fig_shap = px.bar(
            df_shap, 
            x="Impact", 
            y="Feature", 
            color="Type",
            orientation='h',
            color_discrete_map=color_map,
            title="Top 10 Influential Factors"
        )
        fig_shap.update_layout(
            yaxis={'categoryorder':'total ascending'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#c9d1d9"),
            margin=dict(l=20, r=20, t=40, b=20),
            height=400,
            legend_title_text=""
        )
        fig_shap.update_xaxes(gridcolor='#30363d')

        st.plotly_chart(fig_shap, use_container_width=True)

        # Display Justification
        top_idx = df_shap['AbsImpact'].idxmax()
        top_feature = df_shap.loc[top_idx, 'Feature']
        top_impact = df_shap.loc[top_idx, 'Impact']
        
        # Natural Language Logic
        if "OverTime" in top_feature:
            readable_feature = "Working Overtime"
        elif "Income" in top_feature:
            readable_feature = "Monthly Income Level"
        elif "Age" in top_feature:
            readable_feature = "Age"
        elif "YearsAtCompany" in top_feature:
            readable_feature = "Tenure at Company"
        elif "StockOptionLevel" in top_feature:
            readable_feature = "Stock Option Level"
        else:
            readable_feature = top_feature.replace("_", " ")

        if top_impact > 0:
            message = (
                f"**Main Concern:** {readable_feature} is the strongest factor increasing risk. "
                "Improving this area could have the biggest impact on retention."
            )
            color = "#ffcccc" # Light red
        else:
            message = (
                f"**Top Strength:** {readable_feature} is a strong stabilizer. "
                "This factor is actively helping to keep the employee at the company."
            )
            color = "#ccffdd" # Light green
            
        justification_placeholder.markdown(
            f"<div style='background-color: {color}; padding: 15px; border-radius: 10px; color: #161b22;'>"
            f"{message}</div>", 
            unsafe_allow_html=True
        )


else:
    # Welcome Screen
    st.info("Please enter the employee's details in the sidebar and click **Analyze Risk Profile** to start.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stCard">
            <h3>Demographic Analysis</h3>
            <p>Evaluates the impact of age, distance from home, and marital status on retention.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stCard">
            <h3>Job Satisfaction</h3>
            <p>Correlates satisfaction scores and work-life balance with employee churn.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stCard">
            <h3>Compensation</h3>
            <p>Assesses how monthly income, overtime, and salary hikes influence decisions.</p>
        </div>
        """, unsafe_allow_html=True)
