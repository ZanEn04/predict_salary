import pandas as pd
import streamlit as st
from joblib import load
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer

# Load the trained model, and initialize the encoder and scaler
model_file = 'RandomForest.joblib'

try:
    model = load(model_file)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    scaler = Normalizer()
except Exception as e:
    st.error(f'Error loading files: {e}')
    st.stop()  # Stop the script if there's an issue with loading files

# Fit the encoder with an example dataset containing all possible categories
categories = {
    'workclass': ['Private'],
    'education': ['Bachelors'],
    'marital-status': ['Never-married'],
    'occupation': ['Prof-specialty'],
    'relationship': ['Not-in-family'],
    'race': ['White'],
    'sex': ['Male'],
    'native-country': ['United-States']
}
training_data = pd.DataFrame(categories)

# Fit the encoder using the simulated training data
encoder.fit(training_data)

def main():
    st.title('Salary Prediction App')
    st.write('Enter details to predict the salary.')

    # Input fields for the features
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    workclass = st.selectbox('Workclass', ['Private', 'Federal-gov', 'Local-gov', 'State-gov'])
    education = st.selectbox('Education', ['Bachelors', 'Masters', 'Doctorate'])
    education_num = st.number_input('Education Number', min_value=1, max_value=16, value=10)
    marital_status = st.selectbox('Marital Status', ['Never-married', 'Married-civ-spouse'])
    occupation = st.selectbox('Occupation', ['Prof-specialty', 'Exec-managerial'])
    relationship = st.selectbox('Relationship', ['Not-in-family', 'Husband', 'Wife'])
    race = st.selectbox('Race', ['White', 'Black', 'Asian-Pac-Islander'])
    sex = st.selectbox('Sex', ['Female', 'Male'])
    capital_gain = st.number_input('Capital Gain', min_value=0, max_value=100_000, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, max_value=100_000, value=0)
    hours_per_week = st.number_input('Hours Per Week', min_value=1, max_value=100, value=40)
    native_country = st.selectbox('Native Country', ['United-States', 'Canada'])

    if st.button('Predict'):
        try:
            # Prepare the input data for prediction
            input_data = pd.DataFrame({
                'age': [age],
                'workclass': [workclass],
                'education': [education],
                'education-num': [education_num],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'relationship': [relationship],
                'race': [race],
                'sex': [sex],
                'capital-gain': [capital_gain],
                'capital-loss': [capital_loss],
                'hours-per-week': [hours_per_week],
                'native-country': [native_country]
            })
    
            # One-hot encode the categorical features
            categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
            input_data_encoded = encoder.transform(input_data[categorical_columns])
    
            # Create a DataFrame with encoded columns
            encoded_df = pd.DataFrame(input_data_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    
            # Combine encoded features with numeric features
            numeric_features = input_data.drop(columns=categorical_columns)
            final_input_data = pd.concat([numeric_features.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    
            # Align the input columns with the model’s training columns
            expected_columns = encoder.get_feature_names_out(categorical_columns).tolist()  # Get the expected feature names from encoder
            final_input_data = final_input_data.reindex(columns=expected_columns, fill_value=0)  # Ensure all columns are present
    
            # Scale the numeric features
            final_input_data_scaled = pd.DataFrame(scaler.transform(final_input_data), columns=final_input_data.columns)
    
            # Predict using the trained model
            prediction = model.predict(final_input_data_scaled)
            predicted_salary = '>50K' if prediction[0] == 1 else '<=50K'
    
            # Display prediction
            st.success(f'The predicted salary for the provided details is: {predicted_salary}')
        except Exception as e:
            st.error(f'An error occurred during prediction: {e}')

if __name__ == '__main__':
    main()
