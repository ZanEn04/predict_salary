import streamlit as st
import pandas as pd
from joblib import load

# Load the trained model, encoder, and scaler
model_file = 'RandomForest.joblib'
encoder_file = 'OneHotEncoder.joblib'
scaler_file = 'FeatureScaling.joblib'

try:
    model = load(model_file)
    encoder = load(encoder_file)
    scaler = load(scaler_file)
except Exception as e:
    st.error(f'Error loading files: {e}')
    st.stop()  # Stop the script if there's an issue with loading files

# Define categories for categorical features
workclass_options = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                     'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
education_options = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
                     'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', 
                     '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 
                          'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation_options = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                      'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                      'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                      'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
                      'Armed-Forces']
relationship_options = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 
                         'Other-relative', 'Unmarried']
race_options = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
native_country_options = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 
                          'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 
                          'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 
                          'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 
                          'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 
                          'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 
                          'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 
                          'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 
                          'Peru', 'Hong', 'Holand-Netherlands']

def main():
    st.title('Salary Prediction App')
    st.write('Enter details to predict if the person earns more than $50K/year.')

    # Input fields for the features
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    workclass = st.selectbox('Workclass', workclass_options)
    education = st.selectbox('Education', education_options)
    education_num = st.number_input('Education Number', min_value=1, max_value=16, value=10)
    marital_status = st.selectbox('Marital Status', marital_status_options)
    occupation = st.selectbox('Occupation', occupation_options)
    relationship = st.selectbox('Relationship', relationship_options)
    race = st.selectbox('Race', race_options)
    sex = st.selectbox('Sex', ['Female', 'Male'])
    capital_gain = st.number_input('Capital Gain', min_value=0, max_value=100_000, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, max_value=100_000, value=0)
    hours_per_week = st.number_input('Hours Per Week', min_value=1, max_value=100, value=40)
    native_country = st.selectbox('Native Country', native_country_options)

    if st.button('Predict'):
        try:
            # Prepare the input data for prediction
            input_data = pd.DataFrame({
                'age': [age],
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
            categorical_columns = ['education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
            input_data_encoded = encoder.transform(input_data[categorical_columns])

            # Create a DataFrame with encoded columns
            encoded_df = pd.DataFrame(input_data_encoded, columns=encoder.get_feature_names_out(categorical_columns))

            # Combine encoded features with numeric features
            numeric_features = input_data.drop(columns=categorical_columns)
            final_input_data = pd.concat([numeric_features.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

            # Ensure the columns match the model's expected feature names
            model_feature_names = model.feature_names_in_
            final_input_data = final_input_data[model_feature_names]  # Reorder or filter columns as needed

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
