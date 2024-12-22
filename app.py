import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the Random Forest model and Label Encoder
rf_model = joblib.load('random_forest_model.pkl')
le = joblib.load('label_encoder.pkl')

# Load the diet data
diet_df = pd.read_pickle('diet_data.pkl')

# Function to convert symptoms to binary array
def get_symptom_array(selected_symptoms, all_symptoms):
    symptom_array = [0] * len(all_symptoms)
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            symptom_index = all_symptoms.index(symptom)
            symptom_array[symptom_index] = 1
    return symptom_array

# Function to get diet and medication for predicted disease
def get_diet_for_disease(predicted_disease):
    disease_data = diet_df[diet_df['Disease'] == predicted_disease]
    if not disease_data.empty:
        suggested_diet = disease_data['Suggested Diet'].values[0]
        foods_to_eat = disease_data['Foods to Eat'].values[0]
        foods_to_avoid = disease_data['Foods to Avoid'].values[0]
        medication = disease_data['Medication'].values[0]
        additional_tips = disease_data['Additional Tips'].values[0] if 'Additional Tips' in disease_data.columns else "No additional tips available."
        return {
            'Disease': predicted_disease,
            'Suggested Diet': suggested_diet,
            'Foods to Eat': foods_to_eat,
            'Foods to Avoid': foods_to_avoid,
            'Medication': medication,
            'Additional Tips': additional_tips
        }
    else:
        return {"Error": "Disease not found in the dataset"}

# Full list of symptoms (133)
all_symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 
    'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination',
    'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 
    'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
    'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell',
    'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation',
    'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 
    'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

# Streamlit Interface
st.title("Disease Prediction & Diet Suggestion")

# Patient Details
st.sidebar.header("Patient Information")
patient_name = st.sidebar.text_input("Name")
patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25)
patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
contact_info = st.sidebar.text_area("Contact Information")

# Check if patient details and past medical history are stored in session state
if 'patient_history' not in st.session_state:
    # This is a new patient (first-time visit)
    st.session_state.patient_history = {}

# Display past medical history if available
patient_key = f"{patient_name}_{patient_age}_{patient_gender}"
if patient_key in st.session_state.patient_history:
    st.sidebar.write(f"Welcome back, {patient_name}!")
    past_history = st.session_state.patient_history[patient_key]
    st.sidebar.write("Past History:", past_history)
else:
    st.sidebar.write(f"Hello {patient_name}, this seems to be your first visit!")
    past_history = None

# Allow users to select symptoms
selected_symptoms = st.multiselect("Select Symptoms", all_symptoms)

# Prediction button
if st.button('Predict Disease'):
    if len(selected_symptoms) < 3:
        st.warning("Please select at least 3 symptoms for a valid prediction.")
        st.info("Selecting more symptoms improves the accuracy of the prediction.")
    else:
        # Convert selected symptoms to binary array
        input_data = get_symptom_array(selected_symptoms, all_symptoms)
        input_data = np.array(input_data).reshape(1, -1)

        # Predict disease
        disease_prediction = rf_model.predict(input_data)
        predicted_disease = le.inverse_transform(disease_prediction)[0]

        # Get diet, medication, and drug details
        diet_info = get_diet_for_disease(predicted_disease)

        # Display prediction results
        st.subheader(f"Predicted Disease: {predicted_disease}")

        if 'Error' in diet_info:
            st.error(diet_info['Error'])
        else:
            st.write(f"**Suggested Diet**: {diet_info['Suggested Diet']}")
            st.write(f"**Foods to Eat**: {diet_info['Foods to Eat']}")
            st.write(f"**Foods to Avoid**: {diet_info['Foods to Avoid']}")
            st.write(f"**Medication**: {diet_info['Medication']}")
            st.write(f"**Additional Tips**: {diet_info['Additional Tips']}")

        # Save patient data with past history for future visits
        patient_data = {
            'Name': patient_name,
            'Age': patient_age,
            'Gender': patient_gender,
            'Contact Information': contact_info,
            'Symptoms': selected_symptoms,
            'Predicted Disease': predicted_disease
        }

        # Store past history for future visits
        if patient_key not in st.session_state.patient_history:
            st.session_state.patient_history[patient_key] = patient_data
        st.write("Patient Record Saved:", patient_data)
