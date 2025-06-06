# Fastlane Patient Risk Dashboard

## Overview
This project contains a machine learning pipeline and Streamlit app to predict diabetes risk levels based on patient data. The dashboard allows uploading new patient data, performs real-time predictions using a trained RandomForest model, and visualizes risk distributions.

## Project Structure
- \data/\: Sample and real patient CSV datasets.
- \models/\: Trained ML model(s) stored as \.pkl\ files.
- \
otebooks/\: Jupyter notebooks for exploratory data analysis and model development.
- \src/\: Python scripts for training and prediction.
- \pp.py\: Streamlit dashboard application.
- \equirements.txt\: Python dependencies.

## How to Run
1. Clone the repo
2. Install dependencies:
   \\\
   pip install -r requirements.txt
   \\\
3. Run the Streamlit app:
   \\\
   streamlit run app.py
   \\\
4. Open the displayed URL to interact with the dashboard.

## Features
- Interactive filters on age groups and risk levels
- Upload new patient CSV for real-time predictions
- Visualizations for risk level, outcome, and predicted class distributions
- Download filtered results as CSV

## Model
The model is a RandomForest classifier trained with hyperparameter tuning. It outputs probabilities and risk flags to help classify patients as high or low risk for diabetes.

## Next Steps
- Add patient profile drill-down
- Implement time-based filters
- Integrate with LIMS/EHR systems
- Deploy as a web app or mobile app for patient self-assessment

## License
MIT License

