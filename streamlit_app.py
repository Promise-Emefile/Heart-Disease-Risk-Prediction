mport streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# Load trained model
model = XGBClassifier()
model.load_model("Xgb_model.json")

# Load expected feature columns from training
with open("xgb_features.txt") as f:
    expected_cols = [line.strip() for line in f]

# Streamlit UI setup
st.set_page_config(page_title='Heart Disease Predictor', layout='centered')
st.title('💓 Heart Disease Prediction App')
st.markdown("""
Welcome! This app uses an XGBoost model to estimate your risk of heart disease.

### 👉 How to Use:
1. Fill in your medical info.
2. Click *Predict*.
3. View your result instantly.

> This tool is for educational purposes only.
""")

# Input fields
age = st.number_input('Age', min_value=1, max_value=120)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200)
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
restecg = st.selectbox('Resting ECG Result', ['normal', 'lv hypertrophy', 'ST-T abnormality'])
thalch = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220)
exang = st.selectbox('Exercise Induced Angina', ['True', 'False'])
oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=6.0, step=0.1)
slope = st.selectbox('Slope of ST Segment', ['upsloping', 'flat', 'downsloping'])
ca = st.slider('Number of Major Vessels by Fluoroscopy', min_value=0, max_value=4)
thal = st.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversable defect'])

# Collect user input
input_dict = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalch': thalch,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

input_df = pd.DataFrame([input_dict])

# Encode categorical variables using the same method as training
input_encoded = pd.get_dummies(input_df)

# Add any missing columns and reorder
for col in expected_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[expected_cols]  # match column order

# Predict
if st.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1]
    result = "💔 Heart Disease Detected" if prediction == 1 else "💖 No Heart Disease Detected"
    
    st.success(f"🩺 Result: *{result}*")
    st.info(f"Risk score: {proba:.2f} (1 = highest risk)")
