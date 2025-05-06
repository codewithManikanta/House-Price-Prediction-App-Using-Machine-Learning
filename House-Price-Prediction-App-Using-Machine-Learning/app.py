import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk

# Load the trained model
import os
from prediction_pipeline import PredictionPipeline

# Get the absolute path to the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model file from the same directory as the script
model_path = os.path.join(current_dir, 'House_prediction_model.pkl')
model = pk.load(open(model_path, 'rb'))

# Add a header with styled markdown
st.markdown(
    """
    <h1 style="text-align: center; color: #4CAF50;">🏠 Bangalore House Prices Predictor</h1>
    <p style="text-align: center; font-size: 18px; color: #555;">
    Predict the price of your dream house in Bangalore with ease!</p>
    """,
    unsafe_allow_html=True
)

# Load the data
data_path = os.path.join(current_dir, 'cleaned_data.csv')
data = pd.read_csv(data_path)

# Add a sidebar for user inputs
st.sidebar.header('Input Features')
loc = st.sidebar.selectbox('Choose the location', data['location'].unique())
sqft = st.sidebar.number_input('Enter the Total sqft', min_value=0.0, step=1.0)
beds = st.sidebar.number_input('Enter the Number of Bedrooms', min_value=0, step=1)
bath = st.sidebar.number_input('Enter the Number of Bathrooms', min_value=0, step=1)
balc = st.sidebar.number_input('Enter the Number of Balconies', min_value=0, step=1)

# Preprocess input data
# Ensure the column order matches the model's expected input
input = pd.DataFrame([[loc, sqft, bath, balc, beds]], columns=['location', 'total_sqft', 'bath', 'balcony', 'bedrooms'])

# Add a button to predict price
if st.button('Predict Price'):
    try:
        # Ensure location is encoded if required by the model
        if 'location' in data.columns and loc not in data['location'].unique():
            st.error("Invalid location selected. Please choose a valid location.")
        else:
            output = model.predict(input)
            out_str = f"<h3 style='text-align: center; color: #FF5722;'>💰 Price of the House: ₹{output[0] * 100000:,.2f}</h3>"
            st.markdown(out_str, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Add a footer with a logo or additional information
st.markdown(
    """
    <hr>
    <p style="text-align: center; font-size: 14px; color: #888;">
    Copyright By Manikanta. All Rights Reserved.| © 2025</p>
    """,
    unsafe_allow_html=True
)