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

# Define all possible categories (these must match those used during training)
workclass_options = ['Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']
education_options = ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college']
marital_status_options = ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']
occupation_options = ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving']
relationship_options = ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
race_options = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
sex_options = ['Female', 'Male']
native_country_options = ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 
                          'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 
                          'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 
                          'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia']

# Prepare the dummy data for fitting the OneHotEncoder
# We use lists with equal length (here we just repeat the most common categories for simplicity)
categories_data = {
    'workclass': workclass_options * (len(native_country_options) // len(workclass_options)),
    'education': education_options * (len(native_country_options) // len(education_options)),
    'marital-status': marital_status_options * (len(native_country_options) // len(marital_status_options)),
    'occupation': occupation_options * (len(native_country_options) // len(occupation_options)),
    'relationship': relationship_options * (len(native_country_options) // len(relationship_options)),
    'race': race_options * (len(native_country_options) // len(race_options)),
    'sex': sex_options * (len(native_country_options) // len(sex_options)),
    'native-country': native_country_options
}

dummy_data = pd.DataFrame(categories_data)

# Fit the OneHotEncoder on the categorical columns
encoder.fit(dummy_data)

def main():
    st.title('Salary Prediction App')
    st.write('Enter details to predict the salary.')

    # Input fields for the features
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    workclass = st.selectbox('Workclass', workclass_options)
    education = st.selectbox('Education', education_options)
    education_num = st.number_input('Education Number', min_value=1, max_value=16, value=10)
    marital_status = st.selectbox('Marital Status', marital_status_options)
    occupation = st.selectbox('Occupation', occupation_options)
    relationship = st.selectbox('Relationship', relationship_options)
    race = st.selectbox('Race', race_options)
    sex = st.selectbox('Sex', sex_options)
    capital_gain = st.number_input('Capital Gain', min_value=0, max_value=100_000, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, max_value=100_000, value=0)
    hours_per_week = st.number_input('Hours Per Week', min_value=1, max_value=100, value=40)
    native_country = st.selectbox('Native Country', native_country_options)

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
            numeric_columns = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            numeric_features = input_data[numeric_columns]
            final_input_data = pd.concat([numeric_features.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    
            # Align the input columns with the model’s training columns
            expected_columns = model.feature_names_in_  # Ensure the model gets the correct columns
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
