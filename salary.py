import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoder
model = joblib.load('salary_prediction_model.joblib')
encoder = joblib.load('encoder.joblib')

# Streamlit UI
st.title("Salary Prediction App")

st.write("Fill in the details to predict if salary is >50K")

# Create input fields
age = st.number_input("Age", min_value=0, max_value=100, value=25)
workclass = st.selectbox("Workclass", encoder.categories_[0])
education = st.selectbox("Education", encoder.categories_[1])
marital_status = st.selectbox("Marital Status", encoder.categories_[2])
occupation = st.selectbox("Occupation", encoder.categories_[3])
relationship = st.selectbox("Relationship", encoder.categories_[4])
race = st.selectbox("Race", encoder.categories_[5])
sex = st.selectbox("Sex", encoder.categories_[6])
native_country = st.selectbox("Native Country", encoder.categories_[7])
hours_per_week = st.number_input("Hours per week", min_value=1, max_value=100, value=40)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

# Predict Button
if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'education': [education],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'native-country': [native_country],
        'hours-per-week': [hours_per_week],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss]
    })

    # One-hot encode the input data using the same encoder
    encoded_input = encoder.transform(input_data)
    
    # Make a prediction
    prediction = model.predict(encoded_input)
    result = "Salary >50K" if prediction == 1 else "Salary <=50K"
    
    st.write(f"Predicted: {result}")
