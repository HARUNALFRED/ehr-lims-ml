# -*- coding: utf-8 -*-
"""Copy of Diabetes Prediction Model &  Patient Scoring.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17n7R0aw33blEgVTfk_oEwkE4fHqnaoVn
"""

import pandas as pd

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin',
                'bmi', 'diabetes_pedigree_function', 'age', 'outcome']
df = pd.read_csv(url, names=column_names)

# Show first few rows
print(df.head())

# Basic info and missing value check
print(df.info())
print(df.describe())

df

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Features and target
X = df.drop('outcome', axis=1)
y = df['outcome']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees
    'max_depth': [None, 10, 20, 30],      # Max depth of tree
    'min_samples_split': [2, 5, 10],      # Min samples to split node
    'min_samples_leaf': [1, 2, 4],        # Min samples at leaf node
    'class_weight': [None, 'balanced']    # Handle class imbalance
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# GridSearch with 5-fold cross-validation, scoring for recall (catch more diabetics)
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=5,
                           scoring='recall',
                           n_jobs=-1,
                           verbose=2)

# Fit GridSearch to training data
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best parameters:", grid_search.best_params_)

# Best model
best_rf = grid_search.best_estimator_

# Predict and evaluate with best model
y_pred_best = best_rf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_best))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))



"""**Performance on Unseen Data using the Tet Data**"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assume original full data
X_full = df.drop('outcome', axis=1)
y_full = df['outcome']

# Split full data into train_val (80%) and test (20%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

# Further split train_val into training and validation sets (e.g., 75/25 split)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# Now train your best model on X_train and y_train (simulate tuning done on val)
best_rf.fit(X_train, y_train)

# Evaluate on validation set (for tuning feedback)
y_val_pred = best_rf.predict(X_val)
print("Validation set performance:")
print(classification_report(y_val, y_val_pred))

# Finally, simulate unseen data evaluation on the test set
y_test_pred = best_rf.predict(X_test)
print("\nTest set (unseen data) performance:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

import pandas as pd

# Create a DataFrame with actual and predicted values
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred
})

# Filter to only include positive class (1)
positive_cases_df = results_df[results_df['Actual'] == 1]

# Display first few rows
print(positive_cases_df.head())

# Optionally, show counts
print(f"Total positive cases in test set: {positive_cases_df.shape[0]}")
print(f"Correctly predicted positives: {(positive_cases_df['Actual'] == positive_cases_df['Predicted']).sum()}")
print(f"Incorrectly predicted positives: {(positive_cases_df['Actual'] != positive_cases_df['Predicted']).sum()}")

import pandas as pd

# Assume X_test has the patient IDs as its index (or you can set it)
# If not, reset index or create IDs as needed.

# Create DataFrame with index, actual, and predicted
results_df = pd.DataFrame({
    'PatientID': X_test.index,
    'Actual': y_test,
    'Predicted': y_test_pred
})

# Filter for positive class (Actual == 1)
positive_cases_df = results_df[results_df['Actual'] == 1]

print(positive_cases_df.head())

# Optional: save to CSV
# positive_cases_df.to_csv('positive_cases_predictions.csv', index=False)

positive_cases_df.head(20)

import pandas as pd
import numpy as np

# Get predicted probabilities for test set (X_test)
probs = best_rf.predict_proba(X_test)

# probs is an array with shape (n_samples, 2), columns correspond to class 0 and class 1
prob_0 = probs[:, 0]
prob_1 = probs[:, 1]

# Create DataFrame with patient index, actual labels, predicted labels, probabilities
results_df = pd.DataFrame({
    'PatientID': X_test.index,
    'Actual': y_test,
    'Predicted': y_test_pred,
    'Prob_No_Diabetes': prob_0,
    'Prob_Diabetes': prob_1,
})

# Define function to set Risk flag based on your criteria
def risk_flag(row):
    if row['Predicted'] == 1:
        return 'High Risk'
    elif row['Predicted'] == 0 and row['Prob_No_Diabetes'] < 0.2:
        return 'High Risk'
    else:
        return 'Low Risk'

# Apply the function to each row
results_df['Risk_Flag'] = results_df.apply(risk_flag, axis=1)

# Show the first few rows
print(results_df.head())

# Optionally, save to CSV
# results_df.to_csv('patient_risk_flags.csv', index=False)

# results_df.shape

# df.shape

# Assuming your original dataset df and best_rf model are ready

# Separate features (exclude target 'outcome' column)
X_all = df.drop('outcome', axis=1)

# Predict class labels on entire dataset
pred_all = best_rf.predict(X_all)

# Predict probabilities on entire dataset
probs_all = best_rf.predict_proba(X_all)
prob_0_all = probs_all[:, 0]
prob_1_all = probs_all[:, 1]

# Create results DataFrame with all patient info
full_results_df = df.copy()  # keep all original columns

# Add prediction and probabilities columns
full_results_df['Predicted'] = pred_all
full_results_df['Prob_No_Diabetes'] = prob_0_all
full_results_df['Prob_Diabetes'] = prob_1_all

# Define the risk flag function as before
def risk_flag(row):
    if row['Predicted'] == 1:
        return 'High Risk'
    elif row['Predicted'] == 0 and row['Prob_No_Diabetes'] < 0.2:
        return 'High Risk'
    else:
        return 'Low Risk'

# Apply risk flag
full_results_df['Risk_Flag'] = full_results_df.apply(risk_flag, axis=1)

# Sort by patient identifier or index if you have one (optional)
# full_results_df = full_results_df.sort_index()  # if index reflects patient order

# Display all patients with their original data and predictions
print(full_results_df.head())

# Optionally save full results to CSV for doctor’s review
full_results_df.to_csv('full_patient_predictions.csv', index=False)

full_results_df

# Filter where prediction is different from actual outcome
misclassified_df = full_results_df[full_results_df['Predicted'] != full_results_df['outcome']]

# Show the misclassified records
print(misclassified_df)

misclassified_df.shape

misclassified_df

full_results_df.to_csv('full_patient_predictions.csv', index=False)

!pip install streamlit
!pip install pyngrok

# import streamlit as st
# import pandas as pd
# import altair as alt
# import numpy as np

# @st.cache_data
# def load_data():
#     return pd.read_csv('full_patient_predictions.csv')

# df = load_data()

# st.title("Fastlane Patient Risk Dashboard")

# # Create bins for numeric columns (age example)
# age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
# age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins)-1)]
# df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

# # Age filter as dropdown
# age_group_options = df['age_group'].unique().tolist()
# age_group_filter = st.sidebar.multiselect("Filter by Age Group", age_group_options, default=age_group_options)

# # For glucose - bin as example
# glucose_bins = [50, 80, 110, 140, 170, 200, 250]
# glucose_labels = [f"{glucose_bins[i]}-{glucose_bins[i+1]-1}" for i in range(len(glucose_bins)-1)]
# df['glucose_group'] = pd.cut(df['glucose'], bins=glucose_bins, labels=glucose_labels, right=False)
# glucose_group_options = df['glucose_group'].unique().tolist()
# glucose_group_filter = st.sidebar.multiselect("Filter by Glucose Level", glucose_group_options, default=glucose_group_options)

# # Risk Flag filter dropdown
# risk_options = df['Risk_Flag'].unique().tolist()
# risk_filter = st.sidebar.multiselect("Filter by Risk Level", risk_options, default=risk_options)

# # Outcome filter dropdown
# outcome_options = df['outcome'].unique().tolist()
# outcome_filter = st.sidebar.multiselect("Filter by Outcome", outcome_options, default=outcome_options)

# # Filter dataframe based on dropdown selections
# filtered_df = df[
#     (df['age_group'].isin(age_group_filter)) &
#     (df['glucose_group'].isin(glucose_group_filter)) &
#     (df['Risk_Flag'].isin(risk_filter)) &
#     (df['outcome'].isin(outcome_filter))
# ]

# st.markdown(f"### Displaying {len(filtered_df)} patients")
# st.dataframe(filtered_df)

# # Risk Level Distribution with color
# risk_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('Risk_Flag:N', title='Risk Level'),
#     y=alt.Y('count()', title='Count'),
#     color=alt.condition(
#         alt.datum.Risk_Flag == 'High Risk',
#         alt.value('red'),
#         alt.value('blue')
#     )
# ).properties(title='Risk Level Distribution')

# st.altair_chart(risk_chart, use_container_width=True)

# # Outcome Distribution
# outcome_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('outcome:N', title='Outcome'),
#     y=alt.Y('count()', title='Count'),
#     color=alt.Color('outcome:N')
# ).properties(title='Outcome Distribution')

# st.altair_chart(outcome_chart, use_container_width=True)

!ngrok authtoken 2praMSDMWCc25BTFCpWdNu4Mlsk_5aFDPn68FMBiwAJRWx6ZP

# !kill $(lsof -t -i:8501)

# !streamlit run app.py --server.port 8501

# # === app.py content ===
# app_code = """
# import streamlit as st
# import pandas as pd
# import altair as alt

# @st.cache_data
# def load_data():
#     return pd.read_csv('full_patient_predictions.csv')

# df = load_data()

# st.title("Fastlane Patient Risk Dashboard")

# # Create bins for age groups
# age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
# age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins)-1)]
# df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

# # Sidebar filters: Age Group and Risk Level
# age_group_options = df['age_group'].dropna().unique().tolist()
# age_group_filter = st.sidebar.multiselect("Filter by Age Group", age_group_options, default=age_group_options)

# risk_options = df['Risk_Flag'].dropna().unique().tolist()
# risk_filter = st.sidebar.multiselect("Filter by Risk Level", risk_options, default=risk_options)

# # Filter dataframe
# filtered_df = df[
#     (df['age_group'].isin(age_group_filter)) &
#     (df['Risk_Flag'].isin(risk_filter))
# ]

# st.markdown(f"### Displaying {len(filtered_df)} patients")
# st.dataframe(filtered_df)

# # Risk Level Distribution (no color)
# risk_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('Risk_Flag:N', title='Risk Level'),
#     y=alt.Y('count()', title='Count')
# ).properties(title='Risk Level Distribution')

# st.altair_chart(risk_chart, use_container_width=True)

# # Outcome Distribution (no color)
# outcome_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('outcome:N', title='Outcome'),
#     y=alt.Y('count()', title='Count')
# ).properties(title='Outcome Distribution')

# st.altair_chart(outcome_chart, use_container_width=True)
# """

# # Write the Streamlit app code to app.py
# with open('app.py', 'w') as f:
#     f.write(app_code)

# # === Ngrok and Streamlit runner ===
# import subprocess
# import time
# from pyngrok import ngrok

# # Define port
# port = 8501

# # Start the Streamlit app
# process = subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port', str(port), '--server.headless', 'true'])

# # Wait for the app to initialize
# time.sleep(5)

# # Open ngrok tunnel
# public_url = ngrok.connect(port).public_url

# print(f"Streamlit app live at: {public_url}")

# # === app.py content ===
# app_code = """
# import streamlit as st
# import pandas as pd
# import altair as alt

# @st.cache_data
# def load_data():
#     return pd.read_csv('full_patient_predictions.csv')

# df = load_data()

# st.title("Fastlane Patient Risk Dashboard")

# # Create bins for age groups
# age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
# age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins)-1)]
# df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

# # Sidebar filters: Age Group and Risk Level
# age_group_options = df['age_group'].dropna().unique().tolist()
# age_group_filter = st.sidebar.multiselect("Filter by Age Group", age_group_options, default=age_group_options)

# risk_options = df['Risk_Flag'].dropna().unique().tolist()
# risk_filter = st.sidebar.multiselect("Filter by Risk Level", risk_options, default=risk_options)

# # Filter dataframe
# filtered_df = df[
#     (df['age_group'].isin(age_group_filter)) &
#     (df['Risk_Flag'].isin(risk_filter))
# ]

# st.markdown(f"### Displaying {len(filtered_df)} patients")
# st.dataframe(filtered_df)

# # Risk Level Distribution (no color)
# risk_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('Risk_Flag:N', title='Risk Level'),
#     y=alt.Y('count()', title='Count')
# ).properties(title='Risk Level Distribution')

# st.altair_chart(risk_chart, use_container_width=True)

# # Outcome Distribution (no color)
# outcome_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('outcome:N', title='Outcome'),
#     y=alt.Y('count()', title='Count')
# ).properties(title='Outcome Distribution')

# st.altair_chart(outcome_chart, use_container_width=True)

# # Predicted Distribution (no color)
# predicted_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('Predicted:N', title='Predicted'),
#     y=alt.Y('count()', title='Count')
# ).properties(title='Predicted Value Distribution')

# st.altair_chart(predicted_chart, use_container_width=True)
# """

# # Write the Streamlit app code to app.py
# with open('app.py', 'w') as f:
#     f.write(app_code)

# # === Ngrok and Streamlit runner ===
# import subprocess
# import time
# from pyngrok import ngrok

# # Define port
# port = 8501

# # Start the Streamlit app
# process = subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port', str(port), '--server.headless', 'true'])

# # Wait for the app to initialize
# time.sleep(5)

# # Open ngrok tunnel
# public_url = ngrok.connect(port).public_url

# print(f"Streamlit app live at: {public_url}")

# df

# # === app.py content ===
# app_code = """
# import streamlit as st
# import pandas as pd
# import altair as alt

# @st.cache_data
# def load_data():
#     return pd.read_csv('full_patient_predictions.csv')

# df = load_data()

# st.title("Fastlane Patient Risk Dashboard")

# # Create bins for age groups
# age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
# age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins)-1)]
# df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False).astype(str)

# # Convert Risk_Flag to string (safety)
# df['Risk_Flag'] = df['Risk_Flag'].astype(str)

# # Sidebar filters: Age Group and Risk Level
# age_group_options = df['age_group'].dropna().unique().tolist()
# age_group_filter = st.sidebar.multiselect("Filter by Age Group", age_group_options, default=age_group_options)

# risk_options = df['Risk_Flag'].dropna().unique().tolist()
# risk_filter = st.sidebar.multiselect("Filter by Risk Level", risk_options, default=risk_options)

# # Filter dataframe based on selections
# filtered_df = df[
#     df['age_group'].isin(age_group_filter) &
#     df['Risk_Flag'].isin(risk_filter)
# ]

# st.markdown(f"### Displaying {len(filtered_df)} patients")
# st.dataframe(filtered_df)

# # Risk Level Distribution chart
# risk_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('Risk_Flag:N', title='Risk Level'),
#     y=alt.Y('count()', title='Count')
# ).properties(title='Risk Level Distribution')

# st.altair_chart(risk_chart, use_container_width=True)

# # Outcome Distribution chart
# outcome_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('outcome:N', title='Outcome'),
#     y=alt.Y('count()', title='Count')
# ).properties(title='Outcome Distribution')

# st.altair_chart(outcome_chart, use_container_width=True)

# # Predicted Value Distribution chart
# predicted_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('Predicted:N', title='Predicted'),
#     y=alt.Y('count()', title='Count')
# ).properties(title='Predicted Value Distribution')

# st.altair_chart(predicted_chart, use_container_width=True)
# """

# # Write the Streamlit app code to app.py
# with open('app.py', 'w') as f:
#     f.write(app_code)

# # === Ngrok and Streamlit runner ===
# import subprocess
# import time
# from pyngrok import ngrok

# # Define port
# port = 8501

# # Start the Streamlit app as a background process
# process = subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port', str(port), '--server.headless', 'true'])

# # Give the app some time to start
# time.sleep(5)

# # Open ngrok tunnel on the defined port
# public_url = ngrok.connect(port).public_url

# print(f"Streamlit app live at: {public_url}")

# # === app.py content ===
# app_code = """
# import streamlit as st
# import pandas as pd
# import altair as alt

# # Cache data loading to optimize performance
# @st.cache_data
# def load_data():
#     return pd.read_csv('full_patient_predictions.csv')

# # Load dataset
# df = load_data()

# st.title("Fastlane Patient Risk Dashboard")

# # Create bins for age groups for better filtering and visualization
# age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
# age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins)-1)]
# df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False).astype(str)

# # Ensure Risk_Flag and other categorical columns are string for filtering
# df['Risk_Flag'] = df['Risk_Flag'].astype(str)

# # Sidebar filters using checkboxes inside expanders for better interactivity

# with st.sidebar.expander("Filter by Age Group", expanded=True):
#     age_group_options = df['age_group'].dropna().unique().tolist()
#     selected_age_groups = [age for age in age_group_options if st.checkbox(age, value=True)]

# with st.sidebar.expander("Filter by Risk Level", expanded=True):
#     risk_options = df['Risk_Flag'].dropna().unique().tolist()
#     selected_risk_levels = [risk for risk in risk_options if st.checkbox(risk, value=True)]

# # Filter dataframe based on selected filters
# filtered_df = df[
#     df['age_group'].isin(selected_age_groups) &
#     df['Risk_Flag'].isin(selected_risk_levels)
# ]

# # Display number of records after filtering
# st.markdown(f"### Displaying {len(filtered_df)} patients")
# st.dataframe(filtered_df)

# # Risk Level Distribution Chart (bar chart)
# risk_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('Risk_Flag:N', title='Risk Level'),
#     y=alt.Y('count()', title='Count')
# ).properties(title='Risk Level Distribution')
# st.altair_chart(risk_chart, use_container_width=True)

# # Outcome Distribution Chart
# outcome_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('outcome:N', title='Outcome'),
#     y=alt.Y('count()', title='Count')
# ).properties(title='Outcome Distribution')
# st.altair_chart(outcome_chart, use_container_width=True)

# # Predicted Value Distribution Chart
# predicted_chart = alt.Chart(filtered_df).mark_bar().encode(
#     x=alt.X('Predicted:N', title='Predicted'),
#     y=alt.Y('count()', title='Count')
# ).properties(title='Predicted Value Distribution')
# st.altair_chart(predicted_chart, use_container_width=True)
# """

# # Write the Streamlit app code to app.py file
# with open('app.py', 'w') as f:
#     f.write(app_code)

# # === Ngrok and Streamlit runner code ===
# import subprocess
# import time
# from pyngrok import ngrok

# # Define Streamlit port
# port = 8501

# # Start the Streamlit app as a background process
# process = subprocess.Popen([
#     'streamlit', 'run', 'app.py',
#     '--server.port', str(port),
#     '--server.headless', 'true'
# ])

# # Wait for Streamlit server to start properly
# time.sleep(5)

# # Open an ngrok tunnel to the Streamlit port
# public_url = ngrok.connect(port).public_url

# print(f"Streamlit app live at: {public_url}")

import joblib

# Save the best model for later use in Streamlit app
joblib.dump(best_rf, 'diabetes_model.pkl')
print("Best model saved to diabetes_model.pkl")

"""****"""

# === app.py content ===
app_code = """
import streamlit as st
import pandas as pd
import altair as alt
import joblib

# Cache loading base dataset
@st.cache_data
def load_base_data():
    return pd.read_csv('full_patient_predictions.csv')

# Cache loading trained model
@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.pkl')

# Load base data and model
df = load_base_data()
model = load_model()

st.title("Fastlane Patient Risk Dashboard")

# Sidebar: Upload new patient data CSV
uploaded_file = st.sidebar.file_uploader("Upload New Patient Data (CSV)", type=["csv"])

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)

    # Predict on new data
    new_data['Predicted'] = model.predict(new_data)
    probs = model.predict_proba(new_data)
    new_data['Prob_No_Diabetes'] = probs[:, 0]
    new_data['Prob_Diabetes'] = probs[:, 1]

    # Risk Flag logic
    def risk_flag(row):
        if row['Predicted'] == 1 or row['Prob_No_Diabetes'] < 0.2:
            return 'High Risk'
        else:
            return 'Low Risk'

    new_data['Risk_Flag'] = new_data.apply(risk_flag, axis=1)

    # Override base df with new uploaded data
    df = new_data
    st.sidebar.success("New patient data loaded and predictions applied.")

# Create age group bins for filtering
age_bins = [20,30,40,50,60,70,80,90]
age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins)-1)]
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False).astype(str)

# Sidebar filters for Age Group and Risk Level
age_group_options = df['age_group'].dropna().unique().tolist()
selected_age_groups = st.sidebar.multiselect("Filter by Age Group", age_group_options, default=age_group_options)

risk_options = df['Risk_Flag'].dropna().unique().tolist()
selected_risk_levels = st.sidebar.multiselect("Filter by Risk Level", risk_options, default=risk_options)

# Filter dataframe based on selections
filtered_df = df[
    (df['age_group'].isin(selected_age_groups)) &
    (df['Risk_Flag'].isin(selected_risk_levels))
]

st.markdown(f"### Displaying {len(filtered_df)} patients")
st.dataframe(filtered_df, use_container_width=True)

# Charts

risk_chart = alt.Chart(filtered_df).mark_bar().encode(
    x='Risk_Flag:N',
    y='count()'
).properties(title='Risk Level Distribution')

outcome_chart = alt.Chart(filtered_df).mark_bar().encode(
    x='outcome:N',
    y='count()'
).properties(title='Outcome Distribution')

predicted_chart = alt.Chart(filtered_df).mark_bar().encode(
    x='Predicted:N',
    y='count()'
).properties(title='Predicted Class Distribution')

st.altair_chart(risk_chart, use_container_width=True)

# Only show outcome chart if outcome column exists
if 'outcome' in filtered_df.columns:
    st.altair_chart(outcome_chart, use_container_width=True)

st.altair_chart(predicted_chart, use_container_width=True)

# Download filtered data as CSV
csv = filtered_df.to_csv(index=False)
st.download_button(
    label='Download Filtered Results as CSV',
    data=csv,
    file_name='filtered_patients.csv',
    mime='text/csv'
)
"""

# Write the Streamlit app code to 'app.py'
with open('app.py', 'w') as f:
    f.write(app_code)

# === Ngrok and Streamlit runner ===
import subprocess
import time
from pyngrok import ngrok

# Define Streamlit port
port = 8501

# Start Streamlit app as a subprocess
process = subprocess.Popen([
    'streamlit', 'run', 'app.py',
    '--server.port', str(port),
    '--server.headless', 'true'
])

# Wait for Streamlit app to initialize
time.sleep(5)

# Open ngrok tunnel on the port
public_url = ngrok.connect(port).public_url

print(f"Streamlit app live at: {public_url}")

import streamlit as st
import pandas as pd
import altair as alt

# ---------------------------
# Data Loading with Caching
# ---------------------------
@st.cache_data
def load_data():
    # Load the full patient prediction dataset CSV
    return pd.read_csv('full_patient_predictions.csv')

# Load the dataset
df = load_data()

# ---------------------------
# App Title
# ---------------------------
st.title("Fastlane Patient Risk Dashboard")

# ---------------------------
# Data Preparation: Age Groups
# ---------------------------
age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins)-1)]
# Create age_group column with string labels for easier filtering
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False).astype(str)

# Ensure Risk_Flag column is string type for filtering
df['Risk_Flag'] = df['Risk_Flag'].astype(str)

# ---------------------------
# Sidebar Filters: Expanders + Checkboxes
# ---------------------------

# Age Group filter
with st.sidebar.expander("Filter by Age Group", expanded=True):
    age_group_options = df['age_group'].dropna().unique().tolist()
    selected_age_groups = []
    for age_group in age_group_options:
        checked = st.checkbox(age_group, value=True)
        if checked:
            selected_age_groups.append(age_group)

# Risk Level filter
with st.sidebar.expander("Filter by Risk Level", expanded=True):
    risk_options = df['Risk_Flag'].dropna().unique().tolist()
    selected_risk_levels = []
    for risk_level in risk_options:
        checked = st.checkbox(risk_level, value=True)
        if checked:
            selected_risk_levels.append(risk_level)

# ---------------------------
# Filter DataFrame Based on Selected Filters
# ---------------------------
filtered_df = df[
    (df['age_group'].isin(selected_age_groups)) &
    (df['Risk_Flag'].isin(selected_risk_levels))
]

# ---------------------------
# Display Filtered Data
# ---------------------------
st.markdown(f"### Displaying {len(filtered_df)} patients")
st.dataframe(filtered_df)

# ---------------------------
# Visualizations
# ---------------------------

# Risk Level Distribution Chart
risk_chart = alt.Chart(filtered_df).mark_bar().encode(
    x=alt.X('Risk_Flag:N', title='Risk Level'),
    y=alt.Y('count()', title='Count')
).properties(title='Risk Level Distribution')
st.altair_chart(risk_chart, use_container_width=True)

# Outcome Distribution Chart (only if 'outcome' column exists)
if 'outcome' in filtered_df.columns:
    outcome_chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('outcome:N', title='Outcome'),
        y=alt.Y('count()', title='Count')
    ).properties(title='Outcome Distribution')
    st.altair_chart(outcome_chart, use_container_width=True)

# Predicted Value Distribution Chart (only if 'Predicted' column exists)
if 'Predicted' in filtered_df.columns:
    predicted_chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('Predicted:N', title='Predicted'),
        y=alt.Y('count()', title='Count')
    ).properties(title='Predicted Value Distribution')
    st.altair_chart(predicted_chart, use_container_width=True)

import streamlit as st
import pandas as pd
import altair as alt
import joblib

# ---------------------------
# Cache loading base dataset
@st.cache_data
def load_base_data():
    return pd.read_csv('full_patient_predictions.csv')

# Cache loading trained model
@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.pkl')

# Load base data and model
df = load_base_data()
model = load_model()

st.title("Fastlane Patient Risk Dashboard")

# ---------------------------
# Sidebar: Upload new patient data CSV
uploaded_file = st.sidebar.file_uploader("Upload New Patient Data (CSV)", type=["csv"])

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)

    # Predict on new data using loaded model
    new_data['Predicted'] = model.predict(new_data)
    probs = model.predict_proba(new_data)
    new_data['Prob_No_Diabetes'] = probs[:, 0]
    new_data['Prob_Diabetes'] = probs[:, 1]

    # Risk Flag logic
    def risk_flag(row):
        if row['Predicted'] == 1 or row['Prob_No_Diabetes'] < 0.2:
            return 'High Risk'
        else:
            return 'Low Risk'

    new_data['Risk_Flag'] = new_data.apply(risk_flag, axis=1)

    # Override base df with new uploaded data + predictions
    df = new_data
    st.sidebar.success("New patient data loaded and predictions applied.")

# ---------------------------
# Create age group bins for filtering
age_bins = [20,30,40,50,60,70,80,90]
age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins)-1)]
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False).astype(str)

# Sidebar filters: Age Group and Risk Level (multi-select)
age_group_options = df['age_group'].dropna().unique().tolist()
selected_age_groups = st.sidebar.multiselect("Filter by Age Group", age_group_options, default=age_group_options)

risk_options = df['Risk_Flag'].dropna().unique().tolist()
selected_risk_levels = st.sidebar.multiselect("Filter by Risk Level", risk_options, default=risk_options)

# ---------------------------
# Filter dataframe based on sidebar selections
filtered_df = df[
    (df['age_group'].isin(selected_age_groups)) &
    (df['Risk_Flag'].isin(selected_risk_levels))
]

st.markdown(f"### Displaying {len(filtered_df)} patients")
st.dataframe(filtered_df, use_container_width=True)

# ---------------------------
# Charts

# Risk Level Distribution Chart
risk_chart = alt.Chart(filtered_df).mark_bar().encode(
    x='Risk_Flag:N',
    y='count()'
).properties(title='Risk Level Distribution')
st.altair_chart(risk_chart, use_container_width=True)

# Outcome Distribution Chart (show only if column exists)
if 'outcome' in filtered_df.columns:
    outcome_chart = alt.Chart(filtered_df).mark_bar().encode(
        x='outcome:N',
        y='count()'
    ).properties(title='Outcome Distribution')
    st.altair_chart(outcome_chart, use_container_width=True)

# Predicted Class Distribution Chart
predicted_chart = alt.Chart(filtered_df).mark_bar().encode(
    x='Predicted:N',
    y='count()'
).properties(title='Predicted Class Distribution')
st.altair_chart(predicted_chart, use_container_width=True)

# ---------------------------
# Download filtered data as CSV
csv = filtered_df.to_csv(index=False)
st.download_button(
    label='Download Filtered Results as CSV',
    data=csv,
    file_name='filtered_patients.csv',
    mime='text/csv'
)

# === app.py content ===
app_code = """
import streamlit as st
import pandas as pd
import altair as alt
import joblib

# Cache loading base dataset
@st.cache_data
def load_base_data():
    return pd.read_csv('full_patient_predictions.csv')

# Cache loading trained model
@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.pkl')

# Load base data and model
df = load_base_data()
model = load_model()

st.title("Fastlane Patient Risk Dashboard")

# Sidebar: Upload new patient data CSV
uploaded_file = st.sidebar.file_uploader("Upload New Patient Data (CSV)", type=["csv"])

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)

    # Predict on new data using loaded model
    new_data['Predicted'] = model.predict(new_data)
    probs = model.predict_proba(new_data)
    new_data['Prob_No_Diabetes'] = probs[:, 0]
    new_data['Prob_Diabetes'] = probs[:, 1]

    # Risk Flag logic
    def risk_flag(row):
        if row['Predicted'] == 1 or row['Prob_No_Diabetes'] < 0.2:
            return 'High Risk'
        else:
            return 'Low Risk'

    new_data['Risk_Flag'] = new_data.apply(risk_flag, axis=1)

    # Override base df with new uploaded data + predictions
    df = new_data
    st.sidebar.success("New patient data loaded and predictions applied.")

# Create age group bins for filtering
age_bins = [20,30,40,50,60,70,80,90]
age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins)-1)]
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False).astype(str)

# Sidebar filters with checkboxes inside expanders

with st.sidebar.expander("Filter by Age Group", expanded=True):
    age_group_options = df['age_group'].dropna().unique().tolist()
    selected_age_groups = []
    for age_group in age_group_options:
        if st.checkbox(age_group, value=True):
            selected_age_groups.append(age_group)

with st.sidebar.expander("Filter by Risk Level", expanded=True):
    risk_options = df['Risk_Flag'].dropna().unique().tolist()
    selected_risk_levels = []
    for risk_level in risk_options:
        if st.checkbox(risk_level, value=True):
            selected_risk_levels.append(risk_level)

# Filter dataframe based on checked filters
filtered_df = df[
    (df['age_group'].isin(selected_age_groups)) &
    (df['Risk_Flag'].isin(selected_risk_levels))
]

st.markdown(f"### Displaying {len(filtered_df)} patients")
st.dataframe(filtered_df, use_container_width=True)

# Risk Level Distribution Chart
risk_chart = alt.Chart(filtered_df).mark_bar().encode(
    x='Risk_Flag:N',
    y='count()'
).properties(title='Risk Level Distribution')
st.altair_chart(risk_chart, use_container_width=True)

# Outcome Distribution Chart (show only if column exists)
if 'outcome' in filtered_df.columns:
    outcome_chart = alt.Chart(filtered_df).mark_bar().encode(
        x='outcome:N',
        y='count()'
    ).properties(title='Outcome Distribution')
    st.altair_chart(outcome_chart, use_container_width=True)

# Predicted Class Distribution Chart
predicted_chart = alt.Chart(filtered_df).mark_bar().encode(
    x='Predicted:N',
    y='count()'
).properties(title='Predicted Class Distribution')
st.altair_chart(predicted_chart, use_container_width=True)

# Download filtered data as CSV
csv = filtered_df.to_csv(index=False)
st.download_button(
    label='Download Filtered Results as CSV',
    data=csv,
    file_name='filtered_patients.csv',
    mime='text/csv'
)
"""

# Write the Streamlit app code to 'app.py'
with open('app.py', 'w') as f:
    f.write(app_code)

# === Ngrok and Streamlit runner ===
import subprocess
import time
from pyngrok import ngrok

# Define Streamlit port
port = 8501

# Start the Streamlit app as a subprocess
process = subprocess.Popen([
    'streamlit', 'run', 'app.py',
    '--server.port', str(port),
    '--server.headless', 'true'
])

# Wait for the Streamlit app to initialize
time.sleep(7)

# Open ngrok tunnel on the same port
public_url = ngrok.connect(port).public_url

print(f"Streamlit app live at: {public_url}")

