import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from joblib import load
import tkinter as tk
from tkinter import ttk

# Load trained models
knn = load('K-NearestNeighbor.joblib')
svm = load('SupportVectorMachine.joblib')
rf = load('RandomForest.joblib')
nb = load('NaiveBayes.joblib')
dt = load('DecisionTree.joblib')

# Load OneHotEncoder used for transforming categorical data
encoder = load('encoder.joblib')  # Make sure you save the encoder after training with `dump(encoder, 'encoder.joblib')`

# Non-numeric input options
workclass_options = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
education_options = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation_options = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
relationship_options = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
race_options = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
sex_options = ['Female', 'Male']
native_country_options = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']

# GUI for user input
def predict():
    # Retrieve user input
    age = int(age_entry.get())
    education_num = int(education_num_entry.get())
    capital_gain = int(capital_gain_entry.get())
    capital_loss = int(capital_loss_entry.get())
    hours_per_week = int(hours_per_week_entry.get())
    workclass = workclass_combobox.get()
    education = education_combobox.get()
    marital_status = marital_status_combobox.get()
    occupation = occupation_combobox.get()
    relationship = relationship_combobox.get()
    race = race_combobox.get()
    sex = sex_combobox.get()
    native_country = native_country_combobox.get()
    
    # Create a dataframe for the user input
    input_df = pd.DataFrame([[age, workclass, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country]],
                            columns=['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])
    
    # One-hot encode the categorical variables
    input_encoded = encoder.transform(input_df[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']])
    
    # Create a new dataframe with the encoded data
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']))
    
    # Concatenate numeric columns with the one-hot encoded categorical columns
    input_combined = pd.concat([input_df[['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']], input_encoded_df], axis=1)
    
    # Predict using each model
    knn_pred = knn.predict(input_combined)
    svm_pred = svm.predict(input_combined)
    rf_pred = rf.predict(input_combined)
    nb_pred = nb.predict(input_combined)
    dt_pred = dt.predict(input_combined)
    
    # Display predictions
    result_text.set(f"KNN: {knn_pred[0]}, SVM: {svm_pred[0]}, RF: {rf_pred[0]}, NB: {nb_pred[0]}, DT: {dt_pred[0]}")

# Create the main window
root = tk.Tk()
root.title("Income Prediction")

# Age input
tk.Label(root, text="Age:").grid(row=0, column=0)
age_entry = tk.Entry(root)
age_entry.grid(row=0, column=1)

# Education-Num input
tk.Label(root, text="Education-Num:").grid(row=1, column=0)
education_num_entry = tk.Entry(root)
education_num_entry.grid(row=1, column=1)

# Capital Gain input
tk.Label(root, text="Capital Gain:").grid(row=2, column=0)
capital_gain_entry = tk.Entry(root)
capital_gain_entry.grid(row=2, column=1)

# Capital Loss input
tk.Label(root, text="Capital Loss:").grid(row=3, column=0)
capital_loss_entry = tk.Entry(root)
capital_loss_entry.grid(row=3, column=1)

# Hours per Week input
tk.Label(root, text="Hours per Week:").grid(row=4, column=0)
hours_per_week_entry = tk.Entry(root)
hours_per_week_entry.grid(row=4, column=1)

# Dropdown lists for non-numeric inputs
tk.Label(root, text="Workclass:").grid(row=5, column=0)
workclass_combobox = ttk.Combobox(root, values=workclass_options)
workclass_combobox.grid(row=5, column=1)

tk.Label(root, text="Education:").grid(row=6, column=0)
education_combobox = ttk.Combobox(root, values=education_options)
education_combobox.grid(row=6, column=1)

tk.Label(root, text="Marital Status:").grid(row=7, column=0)
marital_status_combobox = ttk.Combobox(root, values=marital_status_options)
marital_status_combobox.grid(row=7, column=1)

tk.Label(root, text="Occupation:").grid(row=8, column=0)
occupation_combobox = ttk.Combobox(root, values=occupation_options)
occupation_combobox.grid(row=8, column=1)

tk.Label(root, text="Relationship:").grid(row=9, column=0)
relationship_combobox = ttk.Combobox(root, values=relationship_options)
relationship_combobox.grid(row=9, column=1)

tk.Label(root, text="Race:").grid(row=10, column=0)
race_combobox = ttk.Combobox(root, values=race_options)
race_combobox.grid(row=10, column=1)

tk.Label(root, text="Sex:").grid(row=11, column=0)
sex_combobox = ttk.Combobox(root, values=sex_options)
sex_combobox.grid(row=11, column=1)

tk.Label(root, text="Native Country:").grid(row=12, column=0)
native_country_combobox = ttk.Combobox(root, values=native_country_options)
native_country_combobox.grid(row=12, column=1)

# Button to trigger prediction
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=13, column=0, columnspan=2)

# Result display
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text)
result_label.grid(row=14, column=0, columnspan=2)

root.mainloop()
