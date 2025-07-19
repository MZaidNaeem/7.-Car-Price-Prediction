import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# Load saved model and preprocessing files
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Title and description
st.markdown("<h1 style='text-align: center; color: black;'>🚗 Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Fill in the car details below to estimate its market price.</p>", unsafe_allow_html=True)

# Input form title
st.markdown("### 🔧 Car Details")

# Use two columns for better layout
col1, col2 = st.columns(2)

# Store input values
encoded_input = {}
raw_input = {}

# Loop through each feature expected by model
for i, col in enumerate(feature_columns):
    with (col1 if i % 2 == 0 else col2):
        display_name = "Model Year" if col == "model" else col.replace('_', ' ').capitalize()

        # Categorical columns
        if col in label_encoders:
            le = label_encoders[col]
            value = st.selectbox(f"{display_name}", le.classes_)
            encoded_value = le.transform([value])[0]

        # Numeric columns
        else:
            value = st.number_input(f"{display_name}", step=1.0)
            # Handle special case: cap 'model' year at 2023
            if col == 'model':
                if value > 2023:
                    value = 2023
            encoded_value = value

        # Save both encoded and raw input
        encoded_input[col] = encoded_value
        raw_input[col] = value

# Convert input to DataFrame
input_df = pd.DataFrame([encoded_input])
input_df = input_df[model.feature_names_in_]

# Raw input DataFrame for optional display
raw_input_df = pd.DataFrame([raw_input])

# Optional debug info (remove or comment out in production)
# st.write("Model expects features:", list(model.feature_names_in_))
# st.write("Your input columns:", list(input_df.columns))

# Prediction button
if st.button("🔍 Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Car Price: **PKR {int(prediction):,}**")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Best Car Price Predictor</div>",
    unsafe_allow_html=True
)
