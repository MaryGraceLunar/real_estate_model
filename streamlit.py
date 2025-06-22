import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Set the page title and description
st.title("Real Estate Price Predictor")
st.write("""
This app predicts price of a property 
based on its characteristics.
""")

# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()

# Load the pre-trained model
rf_pickle = open("models/LRmodel.pkl", "rb")
rf_model = pickle.load(rf_pickle)
rf_pickle.close()


# Prepare the form to collect user inputs
with st.form("property_inputs"):
    st.subheader("Enter Property Details")

    # Year Sold
    year_sold = st.selectbox("Year Sold", options=list(range(2000, 2025)))

    # Property Tax
    property_tax = st.number_input("Annual Property Tax ($)", min_value=0, step=10)

    # Insurance
    insurance = st.number_input("Annual Insurance Cost ($)", min_value=0, step=10)

    # Bedrooms and Bathrooms
    beds = st.selectbox("Number of Bedrooms", options=[1, 2, 3, 4, 5,6,7,8,9,10])
    baths = st.selectbox("Number of Bathrooms", options=[1, 2, 3,4,5,6,7,8,9,10])

    # Square Footage
    sqft = st.number_input("Square Footage", min_value=100, step=10)

    # Year Built
    year_built = st.selectbox("Year Built", options=list(range(1900, 2025)))

    # Lot Size
    lot_size = st.number_input("Lot Size (sq ft)", min_value=0, step=100)

    # Basement
    basement = st.selectbox("Has Basement?", options=["No", "Yes"])

    # Popular Location
    popular = st.selectbox("Is it in a popular area?", options=["No", "Yes"])

    # Sold during Recession
    recession = st.selectbox("Was it sold during a recession?", options=["No", "Yes"])

    # Property Age
    property_age = st.number_input("Property Age (years)", min_value=0, step=1)

    # Property Type
    property_type = st.selectbox("Property Type", options=["Condo", "Detached"])

    # Submit button
    submitted = st.form_submit_button("Submit")

# Handle the dummy variables and make prediction
if submitted:
    # Convert categorical to binary
    basement = 1 if basement == "Yes" else 0
    popular = 1 if popular == "Yes" else 0
    recession = 1 if recession == "Yes" else 0
    property_type_Condo = 1 if property_type == "Condo" else 0

    # Ensure input order matches training
    prediction_input = [[
        year_sold, property_tax, insurance, beds, baths, sqft, year_built,
        lot_size, basement, popular, recession, property_age, property_type_Condo
    ]]

    # Make prediction
    prediction = model.predict(prediction_input)

    # Display result
    st.subheader("🏡 Predicted Property Price:")
    st.success(f"${prediction[0]:,.2f}")

    # Optional: show processed input
    st.subheader("Inputs Used for Prediction:")
    st.json({
        "year_sold": year_sold,
        "property_tax": property_tax,
        "insurance": insurance,
        "beds": beds,
        "baths": baths,
        "sqft": sqft,
        "year_built": year_built,
        "lot_size": lot_size,
        "basement": basement,
        "popular": popular,
        "recession": recession,
        "property_age": property_age,
        "property_type_Condo": property_type_Condo
    })


