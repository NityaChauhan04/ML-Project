import streamlit as st
import joblib
import numpy as np

model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_sleep_disorder(model, scaler, input_dict):
    class_labels = {0: 'Healthy', 1: 'Sleep Apnea', 2: 'Insomnia'}
    input_values = np.array([
        input_dict['Gender'],
        input_dict['Age'],
        input_dict['Sleep Duration'],
        input_dict['Quality of Sleep'],
        input_dict['Physical Activity Level'],
        input_dict['Stress Level'],
        input_dict['BMI Category'],
        input_dict['Heart Rate']
    ]).reshape(1, -1)
    input_scaled = scaler.transform(input_values)
    predicted_class = model.predict(input_scaled)[0]
    return class_labels[predicted_class]

st.title("Sleep Disorder Prediction")
st.write("Enter the details below to predict the sleep disorder:")

gender = st.selectbox("Gender", ["Male", "Female"], help="Select Male or Female")
gender = 1 if gender == "Male" else 0  

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0)
quality_of_sleep = st.slider("Quality of Sleep", min_value=1, max_value=10, value=7)
physical_activity_level = st.number_input("Physical Activity Level", min_value=0, max_value=100, value=50)
stress_level = st.slider("Stress Level", min_value=1, max_value=10, value=5)

bmi_category = st.selectbox("BMI Category", ["Normal Weight", "Normal", "Overweight", "Obese"], help="Select your BMI category")
bmi_mapping = {
    "Normal Weight": 0,"Normal": 1,"Overweight": 2,"Obese": 3            
}
bmi_category = bmi_mapping[bmi_category]
heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=70)

if st.button("Predict"):
    input_dict = {
        'Gender': gender,
        'Age': age,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity_level,
        'Stress Level': stress_level,
        'BMI Category': bmi_category,
        'Heart Rate': heart_rate
    }
    result = predict_sleep_disorder(model, scaler, input_dict)
    
    st.subheader(f"Predicted Sleep Disorder: {result}")