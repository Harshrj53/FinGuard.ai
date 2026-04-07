import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time
import io
import os
from dotenv import load_dotenv

# Load local environment variables (if running locally via .env file)
load_dotenv()

# Read from OS Environment Variables (Render, Heroku, etc.)
APP_VERSION = os.getenv("APP_VERSION", "v1.0.0")
APP_ENV = os.getenv("APP_ENV", "Production")


# ==========================================
# PAGE CONFIG & CSS
# ==========================================
st.set_page_config(
    page_title="FinGuard AI | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium dark fintech look
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Metrics / Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2e86de;
    }
    
    .prediction-card {
        padding: 24px;
        border-radius: 12px;
        background-color: #1e2130;
        border: 1px solid #32364a;
        margin-bottom: 24px;
    }
    
    .high-risk { border-left: 5px solid #ff4757; }
    .medium-risk { border-left: 5px solid #ffa502; }
    .low-risk { border-left: 5px solid #2ed573; }
    
    .risk-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .high-risk-text { color: #ff4757; }
    .medium-risk-text { color: #ffa502; }
    .low-risk-text { color: #2ed573; }
    
    .reason-box {
        background-color: rgba(255, 71, 87, 0.1);
        border: 1px solid rgba(255, 71, 87, 0.3);
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f1f2f6;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2e86de, #00d2d3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #a4b0be;
        margin-bottom: 30px;
    }
    
    hr {
        margin-top: 10px;
        margin-bottom: 30px;
        border-color: #2f3542;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_model_data():
    try:
        model_data = joblib.load('models/fraud_model.pkl')
        return model_data
    except FileNotFoundError:
        st.error("Model file not found! Please run `python train_model.py` first.")
        st.stop()

@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv('data/fraud_data.csv')
    except FileNotFoundError:
        return None

MODEL_BUNDLE = load_model_data()
PIPELINE = MODEL_BUNDLE['pipeline']
MODEL_METRICS = MODEL_BUNDLE['metrics']
BEST_MODEL_NAME = MODEL_BUNDLE['model_name']
EXPECTED_FEATURES = MODEL_BUNDLE['features']
SAMPLE_DATA = load_sample_data()

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def predict_fraud(input_df):
    """Returns probability and prediction."""
    proba = PIPELINE.predict_proba(input_df)[0][1]
    pred = PIPELINE.predict(input_df)[0]
    return proba, pred

def evaluate_risk(proba, inputs):
    """Business logic for risk assignment and explanation."""
    if proba > 0.70:
        risk = "High"
        color = "high"
        action = "Block"
    elif proba > 0.35:
        risk = "Medium"
        color = "medium"
        action = "Review"
    else:
        risk = "Low"
        color = "low"
        action = "Allow"
        
    # Generate explanations based on inputs
    reasons = []
    if inputs['transaction_amount'].iloc[0] > 2000:
        reasons.append("Unusually high transaction amount.")
    if inputs['location_mismatch'].iloc[0] == 1:
        reasons.append("Transaction location does not match user's usual location.")
    if inputs['failed_login_attempts'].iloc[0] >= 3:
        reasons.append("Multiple failed login attempts prior to transaction.")
    if inputs['is_international'].iloc[0] == 1:
        reasons.append("International transaction detected.")
    if inputs['account_age_days'].iloc[0] < 30:
        reasons.append("Account is relatively new (less than 30 days).")
        
    if not reasons and risk in ["Medium", "High"]:
        reasons.append("Pattern matches historical fraudulent behavior.")
        
    return risk, color, action, reasons

def parse_bulk_data(df):
    """Runs prediction on bulk dataframe."""
    try:
        # Ensure all columns exist
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0 # Default padding
                
        # Filter only expected columns
        df_clean = df[EXPECTED_FEATURES]
        
        probas = PIPELINE.predict_proba(df_clean)[:, 1]
        preds = PIPELINE.predict(df_clean)
        
        df_result = df.copy()
        df_result['Fraud Probability'] = probas
        df_result['Prediction'] = preds
        df_result['Risk Level'] = pd.cut(probas, bins=[-0.1, 0.35, 0.70, 1.1], labels=['Low', 'Medium', 'High'])
        return df_result
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2592/2592476.png", width=60)
    st.markdown("<h2 style='margin-bottom:0;'>FinGuard AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a4b0be; font-size:14px;'>Enterprise Fraud Detection</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    sections = [
        "Home Dashboard",
        "Single Transaction Analysis",
        "Bulk CSV Processing",
        "Visual Analytics",
        "Model Performance",
        "About Project"
    ]
    
    choice = st.radio("Navigation", sections)
    
    st.markdown("---")
    st.info(f"🟢 **System Online**\n\nActive Model: {BEST_MODEL_NAME}\n\nApp Env: {APP_ENV}\n\nVersion: {APP_VERSION}")

# ==========================================
# 1. HOME DASHBOARD
# ==========================================
if choice == "Home Dashboard":
    st.markdown('<p class="main-header">FinGuard AI – Command Center</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered financial fraud detection and risk assessment platform.</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Transactions Analyzed", value="15,000+", delta="Live")
    with col2:
        st.metric(label="Protected Value", value="$4.2M", delta="Active", delta_color="normal")
    with col3:
        st.metric(label="Fraud Prevented", value="1,213", delta="Incidents", delta_color="off")
    with col4:
        st.metric(label="System Latency", value="12ms", delta="-2ms", delta_color="inverse")
        
    st.markdown("### System Overview")
    st.markdown("""
    Welcome to FinGuard AI. This specialized dashboard empowers analysts with real-time machine learning predictions to identify and mitigate financial fraud.
    
    **To begin, select an option from the sidebar:**
    - 🔍 **Single Transaction Analysis:** Manually input data for a suspect transaction.
    - 📁 **Bulk CSV Processing:** Upload logs to scan thousands of transactions at once.
    - 📊 **Visual Analytics:** View macro-trends in the synthetic dataset.
    """)
    
    # Mini chart
    if SAMPLE_DATA is not None:
        st.markdown("### Recent Network Activity (Sample)")
        sample_chart = SAMPLE_DATA.sample(1000)
        fig = px.scatter(sample_chart, x='transaction_time', y='transaction_amount', 
                         color='risk_flag', color_discrete_map={0: '#2e86de', 1: '#ff4757'},
                         labels={'transaction_time': 'Hour of Day', 'transaction_amount': 'Amount ($)', 'risk_flag': 'Fraud'},
                         title="Fraud Distribution by Time & Amount")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#f1f2f6')
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 2. SINGLE TRANSACTION PREDICTION
# ==========================================
elif choice == "Single Transaction Analysis":
    st.markdown('<p class="main-header">Single Transaction Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Input transaction parameters to compute instant risk scores.</p>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.markdown("#### Transaction Input")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0)
            merchant = st.selectbox("Merchant Category", ['Retail', 'Travel', 'Food', 'Entertainment', 'Online', 'Other'])
            time_hr = st.slider("Transaction Time (Hour)", 0, 23, 12)
            acc_age = st.number_input("Account Age (Days)", min_value=0, value=365)
            
        with col2:
            payment = st.selectbox("Payment Method", ['Credit Card', 'Debit Card', 'Bank Transfer', 'Digital Wallet'])
            device = st.selectbox("Device Type", ['Mobile', 'Desktop', 'Tablet', 'Unknown'])
            loc_mismatch = st.radio("Location Mismatch?", ["No", "Yes"])
            unusual_score = st.slider("Unusual Spending Score (0-100)", 0, 100, 20)
            
        with col3:
            freq = st.number_input("Transaction Frequency (Past 24h)", min_value=0, value=3)
            failed_logins = st.number_input("Recent Failed Logins", min_value=0, value=0)
            is_intl = st.radio("International Transaction?", ["No", "Yes"])
            
        submit_btn = st.form_submit_button("Compute Risk Score")
        
    if submit_btn:
        with st.spinner("Analyzing transaction patterns..."):
            time.sleep(0.5) # Simulate API latency
            
            # Map inputs to DataFrame
            input_dict = {
                'transaction_amount': amount,
                'transaction_time': time_hr,
                'merchant_category': merchant,
                'payment_method': payment,
                'device_type': device,
                'location_mismatch': 1 if loc_mismatch == "Yes" else 0,
                'failed_login_attempts': failed_logins,
                'unusual_spending_score': unusual_score,
                'transaction_frequency': freq,
                'account_age_days': acc_age,
                'is_international': 1 if is_intl == "Yes" else 0
            }
            input_df = pd.DataFrame([input_dict])
            
            prob, pred = predict_fraud(input_df)
            risk, color_class, action, reasons = evaluate_risk(prob, input_df)
            
            # Display Results
            st.markdown("### Analysis Results")
            
            html_output = f"""
            <div class="prediction-card {color_class}-risk">
                <div class="risk-title {color_class}-risk-text">Risk Level: {risk.upper()}</div>
                <h3 style="margin-top:0;">Fraud Probability: {prob:.1%}</h3>
                <p style="font-size: 1.1rem; margin-top: 10px;">Recommended Action: <strong>{action}</strong></p>
            """
            
            if reasons and risk != "Low":
                html_output += """<div class="reason-box"><strong>⚠️ Flag Reasons:</strong><ul>"""
                for r in reasons:
                    html_output += f"<li>{r}</li>"
                html_output += "</ul></div>"
            elif risk == "Low":
                html_output += """<div style="margin-top: 10px; color: #2ed573;">✓ Transaction pattern appears normal.</div>"""
                
            html_output += "</div>"
            
            st.markdown(html_output, unsafe_allow_html=True)

# ==========================================
# 3. BULK CSV PROCESSING
# ==========================================
elif choice == "Bulk CSV Processing":
    st.markdown('<p class="main-header">Bulk Batch Processing</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a CSV of multiple transactions to process them simultaneously.</p>', unsafe_allow_html=True)
    
    st.info("The CSV must contain columns similar to the training set: `transaction_amount`, `merchant_category`, etc.")
    
    uploaded_file = st.file_uploader("Upload Transaction Data (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_bulk = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df_bulk)} rows.")
            
            if st.button("Process Batch"):
                with st.spinner("Processing batch using ML engine..."):
                    result_df = parse_bulk_data(df_bulk)
                    
                    if result_df is not None:
                        st.success("Batch processing complete!")
                        
                        # Generate metrics
                        total = len(result_df)
                        high_risk = len(result_df[result_df['Risk Level'] == 'High'])
                        med_risk = len(result_df[result_df['Risk Level'] == 'Medium'])
                        low_risk = len(result_df[result_df['Risk Level'] == 'Low'])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total", total)
                        col2.metric("High Risk", high_risk, delta=f"{high_risk/total:.1%}", delta_color="inverse")
                        col3.metric("Medium Risk", med_risk)
                        col4.metric("Low Risk", low_risk)
                        
                        st.markdown("### Flagged Transactions Preview (High & Medium Risk)")
                        risky_df = result_df[result_df['Risk Level'].isin(['High', 'Medium'])].sort_values(by='Fraud Probability', ascending=False)
                        st.dataframe(risky_df, use_container_width=True)
                        
                        # Download button
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Processed Results CSV",
                            data=csv,
                            file_name='fraud_predictions_output.csv',
                            mime='text/csv',
                        )
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ==========================================
# 4. VISUAL ANALYTICS
# ==========================================
elif choice == "Visual Analytics":
    st.markdown('<p class="main-header">Fraud Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Global insights derived from historical transaction data.</p>', unsafe_allow_html=True)
    
    if SAMPLE_DATA is None:
        st.warning("No historical data found to load analytics.")
    else:
        df = SAMPLE_DATA
        
        tab1, tab2 = st.tabs(["Distributions", "Categorical Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Transaction Amount Distribution")
                # Add tiny noise to avoid issues where hist shows zero
                fig = px.histogram(df, x="transaction_amount", color="risk_flag", barmode="overlay",
                                   color_discrete_map={0: '#2e86de', 1: '#ff4757'},
                                   nbins=50)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#f1f2f6')
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.markdown("#### Transactions by Hour of Day")
                fig2 = px.histogram(df, x="transaction_time", color="risk_flag", barmode="group",
                                   color_discrete_map={0: '#2e86de', 1: '#ff4757'})
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#f1f2f6')
                st.plotly_chart(fig2, use_container_width=True)
                
        with tab2:
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### Fraud by Payment Method")
                grouped = df.groupby(['payment_method', 'risk_flag']).size().reset_index(name='count')
                fig3 = px.bar(grouped, x="payment_method", y="count", color="risk_flag", 
                              color_discrete_map={0: '#2e86de', 1: '#ff4757'},
                              barmode="group")
                fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#f1f2f6')
                st.plotly_chart(fig3, use_container_width=True)
                
            with col4:
                st.markdown("#### Fraud Rate by Location Mismatch")
                rate_df = df.groupby('location_mismatch')['risk_flag'].mean().reset_index()
                rate_df['location_mismatch'] = rate_df['location_mismatch'].map({0: 'Match', 1: 'Mismatch'})
                fig4 = px.pie(rate_df, values='risk_flag', names='location_mismatch', hole=0.5,
                              color='location_mismatch', color_discrete_map={'Match':'#2e86de', 'Mismatch':'#ffa502'})
                fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#f1f2f6')
                st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# 5. MODEL PERFORMANCE
# ==========================================
elif choice == "Model Performance":
    st.markdown('<p class="main-header">Under The Hood: ML Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluating the algorithms powering FinGuard AI.</p>', unsafe_allow_html=True)
    
    st.markdown(f"### Currently Deployed Pipeline: **{BEST_MODEL_NAME}**")
    
    # We load the metrics saved during training
    best_metrics = MODEL_METRICS[BEST_MODEL_NAME]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{best_metrics['Accuracy']:.3f}")
    col2.metric("Precision", f"{best_metrics['Precision']:.3f}")
    col3.metric("Recall", f"{best_metrics['Recall']:.3f}")
    col4.metric("F1 Score", f"{best_metrics['F1 Score']:.3f}")
    col5.metric("ROC-AUC", f"{best_metrics['ROC-AUC']:.3f}")
    
    st.markdown("---")
    st.markdown("### Model Comparison")
    
    # Present as a dataframe
    compare_data = []
    for m_name, m_stats in MODEL_METRICS.items():
        row = {'Model': m_name, 
               'Accuracy': m_stats['Accuracy'],
               'Precision': m_stats['Precision'],
               'Recall': m_stats['Recall'],
               'F1 Score': m_stats['F1 Score'],
               'ROC-AUC': m_stats['ROC-AUC']}
        compare_data.append(row)
        
    compare_df = pd.DataFrame(compare_data).set_index("Model")
    
    # Highlight the deployed model
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #2ed573; color: black' if v else '' for v in is_max]
    
    st.dataframe(compare_df.style.apply(highlight_max, axis=0).format("{:.4f}"), use_container_width=True)
    
    # Render confusion matrix of the best model
    st.markdown("### Confusion Matrix (Test Set)")
    cm = np.array(best_metrics['Confusion Matrix'])
    
    fig = px.imshow(cm, text_auto=True, colorscale='Blues',
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Legit', 'Fraud'], y=['Legit', 'Fraud'])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#f1f2f6')
    st.plotly_chart(fig, use_container_width=False)

# ==========================================
# 6. ABOUT
# ==========================================
elif choice == "About Project":
    st.markdown('<p class="main-header">About FinGuard AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Project architecture and portfolio details.</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Objective
    FinGuard AI was built to demonstrate a production-ready approach to tabular data classification. 
    It tackles the inherently imbalanced problem of financial fraud detection by simulating risky 
    behaviors, preprocessing them effectively, training a robust selection of algorithms (Random Forest, Gradient Boosting, etc.), 
    and exposing the best model via a highly interactive web interface.
    
    ### ⚙️ Machine Learning Workflow
    1. **Data Generation (`utils.py`):** Automatically maps distributions and probabilities to create a heavily imbalanced, realistic synthetic dataset. 
    2. **Preprocessing (`preprocess.py`):** Utilizes `sklearn.compose.ColumnTransformer` and `Pipeline` to impute missing values, scale continuous variables (`StandardScaler`), and encode categorical limits (`OneHotEncoder`).
    3. **Model Training (`train_model.py`):** Evaluates Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting. Automatically selects the model with the best **F1 Score** and exports the entire pipeline object via `joblib`.
    4. **Frontend Service (`app.py`):** This Streamlit application serves the saved pipeline and translates probability outputs into actionable business risk rules.
    
    ### 🛠️ Tech Stack
    - **Language:** Python 3.9+
    - **Data Stack:** Pandas, NumPy
    - **Machine Learning:** Scikit-Learn
    - **UI & Visualization:** Streamlit, Plotly Express
    
    ---
    *Built for demonstration purposes. Do not use synthetic models for live production financial operations without further tuning and real-world data validation.*
    """)
