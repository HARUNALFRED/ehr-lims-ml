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

# Sidebar filters for Region, ID Number, Practice Number, and Patient Name
region_options = df['Region'].dropna().unique().tolist()
selected_region = st.sidebar.multiselect("Filter by Region", region_options, default=region_options)

id_number_options = df['Id Number'].dropna().unique().tolist()
selected_id_number = st.sidebar.multiselect("Filter by Id Number", id_number_options, default=id_number_options)

practice_number_options = df['Practice Number'].dropna().unique().tolist()
selected_practice_number = st.sidebar.multiselect("Filter by Practice Number", practice_number_options, default=practice_number_options)

patient_name_options = df['Patient Name'].dropna().unique().tolist()
selected_patient_name = st.sidebar.multiselect("Filter by Patient Name", patient_name_options, default=patient_name_options)

# Filter dataframe based on sidebar selections
filtered_df = df[
    (df['age_group'].isin(selected_age_groups)) &
    (df['Risk_Flag'].isin(selected_risk_levels)) &
    (df['Region'].isin(selected_region)) &
    (df['Id Number'].isin(selected_id_number)) &
    (df['Practice Number'].isin(selected_practice_number)) &
    (df['Patient Name'].isin(selected_patient_name))
]

st.markdown(f"### Displaying {len(filtered_df)} patients")
st.dataframe(filtered_df, use_container_width=True)

# Visualizations
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
