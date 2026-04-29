import streamlit as st
import pickle
import numpy as np
import json
import os


# Load paths (robust)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

model_path = os.path.join(BASE_DIR, 'models', 'churn_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
feature_path = os.path.join(BASE_DIR, 'models', 'features.json')

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load feature names
with open(feature_path, 'r') as f:
    feature_names = json.load(f)


# UI

st.title("Customer Churn Prediction System")
st.write("Enter Customer Details:")

# Inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])


# Convert inputs

gender = 1 if gender == "Male" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0
phone_service = 1 if phone_service == "Yes" else 0
paperless = 1 if paperless == "Yes" else 0


# Create feature input

input_dict = dict.fromkeys(feature_names, 0)

input_dict['gender'] = gender
input_dict['Partner'] = partner
input_dict['Dependents'] = dependents
input_dict['PhoneService'] = phone_service
input_dict['PaperlessBilling'] = paperless
input_dict['tenure'] = tenure
input_dict['MonthlyCharges'] = monthly_charges
input_dict['TotalCharges'] = total_charges

# Convert to array
features = np.array([list(input_dict.values())])

# Scale
features_scaled = scaler.transform(features)


# Prediction

if st.button("Predict"):
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High Churn Risk ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Customer Likely to Stay ({(1-probability)*100:.2f}%)")