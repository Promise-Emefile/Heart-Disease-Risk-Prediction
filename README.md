# Heart Disease Prediction & Risk Analysis Project
This project uses a machine learning model trained on a medical dataset to predict the likelihood of heart disease based on user-input clinical features. It also integrates a dashboard built in Power BI for deep visual insights and deploys the predictive model via a user-friendly Streamlit web app.

## Model Development (Jupyter Notebook)
#### Data Loading
Reads from heart_disease_uci.csv

Dataset includes 920 entries and 16 medical features (age, chest pain, cholesterol, etc.)
<img width="738" height="329" alt="image" src="https://github.com/user-attachments/assets/d2ac6b7a-707d-4a30-8578-9f2fc62581cf" />

#### Data Exploration & Cleaning
Identifies missing values in key features: slope, ca, thal, etc.

Converts categorical text (e.g. sex, chest pain type) to numerical format

Drops irrelevant columns such as id and dataset

### Model Building
Trains a XGBoost Classifier using features (X) and target (y = num)

Saves model to disk using:

<img width="206" height="49" alt="image" src="https://github.com/user-attachments/assets/cc94599b-9966-4a00-affc-ab1d65085dbf" />

## Power BI Dashboard Insights

To complement the model’s predictions, a Power BI dashboard highlights key patterns from the dataset:

#### Age-Based Risk Distribution

Majority of low-risk individuals are aged 45–60

Minimal representation in very high risk group for ages 75+

#### Gender-Based Analysis

Men show higher prevalence across all risk categories—especially low and high risk zones

#### Overall Risk Profile

About 44% of users fall under No Risk

Only 3.05% classified as Very High Risk, confirming strong model calibration

#### Patient-Level Risk Viewer

Tabular view includes key metrics (Cholesterol, FluoroVessel count, Max HR)

Adds transparency into how risk levels are assigned per patient

## Feature Importance — Model Explainability

Top influential features based on the model’s logic:

<img width="417" height="304" alt="image" src="https://github.com/user-attachments/assets/0eec2f67-21c6-4c92-b2cc-e3a3e339e420" />

These features play the biggest role in how the model predicts heart disease likelihood.

## Streamlit Deployment

The streamlit_app.py file launches an interactive web app where users can input medical data and receive real-time predictions.

#### How to Use
Fill in your medical info (age, cholesterol, chest pain type, etc.)

Click Predict

View result: Heart Disease Detected or No Heart Disease

## Disclaimer
This model offers statistical insights based on medical data and should not be used as a substitute for professional medical advice or diagnosis.

## Author
Built with love by Promise Emefile in Lagos, Nigeria Empowering health decisions through data and AI.
