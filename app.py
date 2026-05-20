import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Website ka Title aur Theme set kar rahe hain
st.set_page_config(page_title="AI Customer Retention System", layout="wide")
st.title("🚀 Enterprise AI: Customer Churn & Segmentation Dashboard")
st.write("This system predicts user churn and segments customers using advanced Machine Learning.")

# Saved Data aur Models ko load kar rahe hain
@st.cache_data
def load_data():
    df = pd.read_csv('rfm_dashboard_data.csv', index_col='Customer ID')
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return df, model, scaler

try:
    rfm, model, scaler = load_data()
    
    # KPI Metrics (Dukan ki live halat upar hi dikhegi)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(rfm))
    col2.metric("At Risk Customers", len(rfm[rfm['Segment'] == 'At Risk']))
    col3.metric("Avg Monetary Value", f"${round(rfm['Monetary'].mean(), 2)}")
    
    st.markdown("---")
    
    # LIVE PREDICTION SECTION (Yahan recruiter khel sakta hai)
    st.subheader("🔮 Live Customer Risk Predictor")
    customer_list = rfm.index.unique()
    selected_customer = st.selectbox("Select a Customer ID to test live:", customer_list)
    
    if selected_customer:
        cust_data = rfm.loc[selected_customer]
        st.write(f"**Current Segment:** `{cust_data['Segment']}`")
        
        # Model ke liye data scale kar rahe hain
        input_data = np.array([[cust_data['Recency'], cust_data['Frequency'], cust_data['Monetary']]])
        input_scaled = scaler.transform(input_data)
        
        # Live Prediction
        risk_prob = model.predict_proba(input_scaled)[0][1]
        
        st.write(f"**AI Churn Risk Probability:**")
        st.progress(float(risk_prob))
        st.write(f"⚠️ `{round(risk_prob * 100, 2)}%` chance that this customer will leave.")
        
        if risk_prob > 0.6:
            st.error("🔴 HIGH RISK: Action required! Send an automated discount coupon immediately.")
        else:
            st.success("🟢 LOW RISK: Customer is healthy and active.")

except Exception as e:
    st.error(f"Please run the Jupyter cells first to generate files. Error: {e}")