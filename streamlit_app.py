import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# Load saved files
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Page title
st.markdown("<h1 style='text-align: center; color: black;'>🚗 Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Fill in the car details below to estimate its market price.</p>", unsafe_allow_html=True)

# Input container
st.markdown("### 🔧 Car Details")

user_input = {}

# Break inputs into two columns for better layout
col1, col2 = st.columns(2)

for i, col in enumerate(feature_columns):
    with (col1 if i % 2 == 0 else col2):
        display_name = col.replace('_', ' ').capitalize()
        if col in label_encoders:
            le = label_encoders[col]
            value = st.selectbox(f"{display_name}", le.classes_)
            user_input[col] = le.transform([value])[0]
        else:
            value = st.number_input(f"{display_name}", step=1.0)
            user_input[col] = value

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("🔍 Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Car Price: **PKR {int(prediction):,}**")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Best Car Price Predictor</div>",
    unsafe_allow_html=True
)
