import streamlit as st
import numpy as np
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered"
)


# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------

model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🩺 Breast Cancer Prediction System")

st.write(
    "K-Nearest Neighbors (KNN) Classification"
)

st.info(
    "Enter the five tumor measurements below."
)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("Enter Tumor Measurements")


mean_radius = st.number_input(
    "Mean Radius",
    min_value=0.0,
    max_value=50.0,
    value=14.0,
    step=0.01
)


mean_texture = st.number_input(
    "Mean Texture",
    min_value=0.0,
    max_value=50.0,
    value=20.0,
    step=0.01
)


mean_perimeter = st.number_input(
    "Mean Perimeter",
    min_value=0.0,
    max_value=300.0,
    value=90.0,
    step=0.01
)


mean_area = st.number_input(
    "Mean Area",
    min_value=0.0,
    max_value=5000.0,
    value=600.0,
    step=0.01
)


mean_smoothness = st.number_input(
    "Mean Smoothness",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.001
)

# Create full feature array (5 features expected)
input_data = np.zeros((1, 5))
#1 Row 5 columns
#input_data = [0 [0.0,0.0,0.0,0.0,0.0]]
#                  0   1   2   3   4  
input_data[0][0] = mean_radius
input_data[0][1] = mean_texture
input_data[0][2] = mean_perimeter
input_data[0][3] = mean_area
input_data[0][4] = mean_smoothness

# Scale input
scaled_input = scaler.transform(input_data)

   #True
if st.button("Predict"):
    prediction = model.predict(scaled_input)

    if prediction[0] == 0:
        st.write("Prediction: Malignant Tumor (Cancer Detected)")
    else:
        st.write("Prediction: Benign Tumor (No Cancer Detected)")



        #EDA
        #Hyper Parameter Tuning
        