import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# Load model
model = XGBClassifier()
model.load_model("Xgb_model.json")

# Use model's feature names directly
expected_cols = model.get_booster().feature_names

# Stage descriptions
stage_desc = {
    0: "No Disease",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Critical"
}

# Streamlit setup
st.set_page_config(page_title='Heart Disease Predictor', layout='centered')
st.title('💓 Heart Disease Prediction App')

st.markdown("""
This app uses a trained XGBoost model to classify heart disease stages from 0 (no disease) to 4 (critical).  
You can enter patient data manually or upload a CSV/Excel file for batch prediction.
""")

st.markdown("---")

# File upload
st.subheader("📁 Upload Patient Data File (Optional)")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df_encoded = pd.get_dummies(df)
    df_encoded.columns = df_encoded.columns.str.strip()

    # Add missing expected columns
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder and filter columns
    df_encoded = df_encoded[expected_cols]
    df_encoded = df_encoded.astype(np.float32)

    try:
        preds = model.predict(df_encoded)
        probas = model.predict_proba(df_encoded)

        df['Prediction'] = preds
        df['Stage Description'] = df['Prediction'].map(stage_desc)

        st.success("✅ Predictions complete!")
        st.write(df[['Prediction', 'Stage Description']].head())

        csv_out = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Results as CSV", data=csv_out, file_name="heart_disease_predictions.csv", mime="text/csv")
    
    except Exception as e:
        st.error("❌ Error during prediction")
        st.code(str(e))

else:
    st.subheader("🧍‍♂ Enter Patient Information")

    with st.form("patient_form"):
        age = st.number_input('Age', min_value=1, max_value=120)
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200)
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False', 'Unknown'])
        restecg = st.selectbox('Resting ECG Result', ['normal', 'lv hypertrophy', 'ST-T abnormality'])
        thalch = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220)
        exang = st.selectbox('Exercise Induced Angina', ['True', 'False', 'Unknown'])
        oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=6.0, step=0.1)
        slope = st.selectbox('Slope of ST Segment', ['upsloping', 'flat', 'downsloping'])
        ca = st.slider('Number of Major Vessels (0–4)', min_value=0, max_value=4)
        thal = st.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversable defect'])

        submitted = st.form_submit_button("🔍 Submit")

    if submitted:
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
        input_encoded = pd.get_dummies(input_df)
        input_encoded.columns = input_encoded.columns.str.strip()

        # Add missing columns
        for col in expected_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reorder and cast dtype
        input_encoded = input_encoded[expected_cols]
        input_encoded = input_encoded.astype(np.float32)

        # DEBUG: show encoded input
        st.write("📊 Encoded Input Preview:")
        st.dataframe(input_encoded)

        try:
            pred = model.predict(input_encoded)[0]
            proba = model.predict_proba(input_encoded)[0]

            st.success(f"🩺 Prediction: {stage_desc[pred]} (Class {pred})")

            st.markdown("### 🔬 Class Probabilities:")
            for i, p in enumerate(proba):
                st.write(f"Class {i} ({stage_desc[i]}): {p:.2%}")

        except Exception as e:
            st.error("❌ Prediction failed.")
            st.code(str(e))
