import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from src.models.predict_model import evaluate_model


# Page setup
st.set_page_config(page_title="Real Estate Price Estimator", layout="centered")

# Title and intro
st.title("Real Estate Price Estimator")
st.markdown("""
Welcome, data explorers!
This app leverages a **Linear Regression model** to estimate the market price of a **Condo** or **Bungalow**, based on the property characteristics you provide below.
""")

# UX Hint
st.caption("Tip: Enter the property specs and let the model estimate its market value using historical data patterns.")

# Load model and MAE
with open("models/LRmodel.pkl", "rb") as lr_pickle:
    lr_model = pickle.load(lr_pickle)

with open("models/LR_MAE.pkl", "rb") as mae_pickle:
    stored_mae = pickle.load(mae_pickle)

# --- Input Form ---
with st.form("property_inputs"):
    st.markdown("### 📋 Enter Property Details")

    col1, col2 = st.columns(2)
    with col1:
        year_sold = st.selectbox("Year Sold", options=list(range(2000, 2025)))
        property_tax = st.number_input("Annual Property Tax ($)", min_value=10, step=10)
        insurance = st.number_input("Annual Insurance Cost ($)", min_value=10, step=10)
        sqft = st.number_input("Square Footage", min_value=100, step=10)
        lot_size = st.number_input("Lot Size (sq ft)", min_value=0, step=100)
        property_age = st.number_input("Property Age (years)", min_value=0, step=1)

    with col2:
        beds = st.selectbox("Number of Bedrooms", options=list(range(1, 11)))
        baths = st.selectbox("Number of Bathrooms", options=list(range(1, 11)))
        basement = st.selectbox("Basement", options=["No", "Yes"])
        popular = st.selectbox("In a Popular Area?", options=["No", "Yes"])
        recession = st.selectbox("Sold During Recession?", options=["No", "Yes"])
        property_type = st.selectbox("Property Type", options=["Condo", "Bungalow"])

    submitted = st.form_submit_button("🚀 Estimate Price")

# --- Prediction Logic ---
if submitted:
    # Encode inputs
    basement = 1 if basement == "Yes" else 0
    popular = 1 if popular == "Yes" else 0
    recession = 1 if recession == "Yes" else 0
    property_type_Condo = 1 if property_type == "Condo" else 0

    prediction_input = [[
        year_sold, property_tax, insurance, beds, baths, sqft,
        lot_size, basement, popular, recession, property_age, property_type_Condo
    ]]

    # Prediction
    prediction = lr_model.predict(prediction_input)

    # Display result
    st.markdown("---")
    st.subheader("💰 Estimated Property Price")
    st.success(f"Estimated Market Price: **${prediction[0]:,.2f}**")

    #Display MAE
    st.caption(f"Model Performance: Mean Absolute Error (MAE) = **${stored_mae:,.2f}**, evaluated on holdout test data.")
    
    with st.expander("Model Info"):
        st.markdown("""
        - **Model**: Linear Regression  
        - **Target**: Property sale price  
        - **Features Used**: Year sold, taxes, sqft, bedrooms, etc.  
        - **Evaluation Metric**: MAE on 20% holdout test set
        """)
    
# Footer
st.markdown("---")
st.caption("Built by Mary Grace Lunar | Educational Use Only")

