import pickle
import numpy as np
import gradio as gr
from sklearn.preprocessing import LabelEncoder

def predict_heart_disease(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    # Load saved LabelEncoders
    sex_encoder = LabelEncoder()
    sex_encoder.classes_ = np.array(['F', 'M'])

    chest_pain_encoder = LabelEncoder()
    chest_pain_encoder.classes_ = np.array(['ATA', 'NAP', 'ASY', 'TA'])

    resting_ecg_encoder = LabelEncoder()
    resting_ecg_encoder.classes_ = np.array(['Normal', 'ST', 'LVH'])

    exercise_angina_encoder = LabelEncoder()
    exercise_angina_encoder.classes_ = np.array(['N', 'Y'])

    st_slope_encoder = LabelEncoder()
    st_slope_encoder.classes_ = np.array(['Up', 'Flat', 'Down'])

    # Transform categorical inputs using LabelEncoders
    sex_encoded = sex_encoder.transform([sex])[0]
    chest_pain_encoded = chest_pain_encoder.transform([chest_pain])[0]
    resting_ecg_encoded = resting_ecg_encoder.transform([resting_ecg])[0]
    exercise_angina_encoded = exercise_angina_encoder.transform([exercise_angina])[0]
    st_slope_encoded = st_slope_encoder.transform([st_slope])[0]

    # Create input data array
    input_data = [
        age, sex_encoded, chest_pain_encoded, resting_bp, cholesterol, fasting_bs,
        resting_ecg_encoded, max_hr, exercise_angina_encoded, oldpeak, st_slope_encoded
    ]

    # Reshape input data for prediction
    input_data = np.array([input_data])

    # Load the model
    with open("heart_failure.pkl", "rb") as file:
        load_model = pickle.load(file)

    # Make prediction
    prediction = load_model.predict(input_data)[0]

    # Return prediction result
    if prediction == 1:
        return "Yes (Heart Disease Detected)"
    else:
        return "No (No Heart Disease)"

# Define Gradio interface
iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Enter Age", minimum=0, maximum=120, value=50),
        gr.Radio(label="Sex", choices=["M", "F"]),
        gr.Radio(label="Chest Pain Type", choices=["ATA", "NAP", "ASY", "TA"]),
        gr.Number(label="Resting Blood Pressure", minimum=0, maximum=200, value=120),
        gr.Number(label="Cholesterol", minimum=0, maximum=500, value=200),
        gr.Radio(label="Fasting Blood Sugar", choices=["0", "1"]),
        gr.Radio(label="Resting ECG", choices=["Normal", "ST", "LVH"]),
        gr.Number(label="Max Heart Rate", minimum=60, maximum=220, value=150),
        gr.Radio(label="Exercise Angina", choices=["N", "Y"]),
        gr.Number(label="Oldpeak", minimum=0.0, maximum=6.0, value=1.0),
        gr.Radio(label="ST Slope", choices=["Up", "Flat", "Down"])
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Heart Disease Prediction",
    description="Enter patient details to predict heart disease."
)

# Launch the interface
iface.launch()