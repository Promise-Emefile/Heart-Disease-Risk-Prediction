import streamlit as st
import import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

# Load JSON model
booster = xgb.Booster()
booster.load_model("Xgb_model.json")

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
oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=6.0)
slope = st.selectbox('Slope of ST Segment', ['upsloping', 'flat', 'downsloping'])
ca = st.slider('Number of Major Vessels by Fluoroscopy', min_value=0, max_value=4)
thal = st.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversable defect'])

# Encode input
input_data = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == 'Male' else 0],
    'cp': [[
        'typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'
    ].index(cp)],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [1 if fbs == 'True' else 0],
    'restecg': [[
        'normal', 'lv hypertrophy', 'ST-T abnormality'
    ].index(restecg)],
    'thalch': [thalch],
    'exang': [1 if exang == 'True' else 0],
    'oldpeak': [oldpeak],
    'slope': [[
        'upsloping', 'flat', 'downsloping'
    ].index(slope)],
    'ca': [ca],
    'thal': [[
        'normal', 'fixed defect', 'reversable defect'
    ].index(thal)],
})

# Convert to DMatrix
dmatrix = xgb.DMatrix(input_data, feature_names=input_data.columns.tolist())

# Predict
if st.button('Predict'):
    prediction = booster.predict(dmatrix)[0]
    result = "Heart Disease Detected" if prediction > 0.5 else "No Heart Disease"
    st.success(f'🩺 Result: *{result}*'
