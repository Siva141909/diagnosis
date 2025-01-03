import streamlit as st
import joblib
import pandas as pd

# Load saved model and encoders
loaded_rf_model = joblib.load('random_forest_model1.pkl')
loaded_risk_encoder = joblib.load('risk_encoder.pkl')
loaded_gender_encoder = joblib.load('gender_encoder.pkl')
required_columns = joblib.load('model_columns.pkl')  # Load saved column order

# Function to preprocess input data
def preprocess_input_data(data):
    # Normalize Gender
    data['Gender'] = data['Gender'].str.lower()
    
    # Encode 'Gender'
    encoded_gender = loaded_gender_encoder.transform(data[['Gender']])

    # Drop 'Gender' from data and append encoded columns
    data = data.drop(['Gender'], axis=1)
    encoded_gender_df = pd.DataFrame(encoded_gender, columns=loaded_gender_encoder.get_feature_names_out(['Gender']))
    data = pd.concat([data.reset_index(drop=True), encoded_gender_df], axis=1)
    
    # Ensure feature order matches training
    data = data.reindex(columns=required_columns, fill_value=0)  # Fill missing columns with 0
    return data

# Interface setup
st.title("Medical Diagnosis Risk Prediction")
st.write("Fill in the details below to predict the risk category of a patient.")

# User Inputs
heart_rate = st.number_input("Heart Rate")
respiratory_rate = st.number_input("Respiratory Rate")
body_temp = st.number_input("Body Temperature (°C)")
oxygen_saturation = st.number_input("Oxygen Saturation (%)")
systolic_bp = st.number_input("Systolic Blood Pressure")
diastolic_bp = st.number_input("Diastolic Blood Pressure")
age = st.number_input("Age", min_value=0)
gender = st.selectbox("Gender", ['Male', 'Female'])
hrv = st.number_input("Derived HRV")
pulse_pressure = st.number_input("Derived Pulse Pressure")
bmi = st.number_input("Derived BMI")
map = st.number_input("Derived MAP")

# Collecting all inputs into a dataframe
new_data = pd.DataFrame({
    'Heart Rate': [heart_rate],
    'Respiratory Rate': [respiratory_rate],
    'Body Temperature': [body_temp],
    'Oxygen Saturation': [oxygen_saturation],
    'Systolic Blood Pressure': [systolic_bp],
    'Diastolic Blood Pressure': [diastolic_bp],
    'Age': [age],
    'Gender': [gender],
    'Derived_HRV': [hrv],
    'Derived_Pulse_Pressure': [pulse_pressure],
    'Derived_BMI': [bmi],
    'Derived_MAP': [map]
})

# When the user clicks the "Predict" button
if st.button('Predict Risk Category'):
    # Preprocess the input data
    preprocessed_data = preprocess_input_data(new_data)

    # Make prediction
    prediction = loaded_rf_model.predict(preprocessed_data)

    # Map the predicted risk category
    category_mapping = {0.0: 'Low Risk', 1.0: 'High Risk'}
    predicted_risk = category_mapping[float(prediction[0])]
    
    # Display the result
    st.success(f"The predicted risk category is: {predicted_risk}")
