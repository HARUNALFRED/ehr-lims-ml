import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 
                'bmi', 'diabetes_pedigree_function', 'age', 'outcome']
df = pd.read_csv(url, names=column_names)

# Create dummy data columns
dummy_data = {
    "Medical History": ["Diabetes", "Hypertension", "Healthy", "Asthma", "Diabetes"] * 20,
    "Outstandings": ["Test A, Test B", "Test C", "Test D", "Test E", "Test F"] * 20,
    "Cash/Receipts": np.random.randint(100, 1000, size=100),
    "System Admin": ["Admin1", "Admin2", "Admin3", "Admin4", "Admin5"] * 20,
    "Received Time": pd.date_range('2025-01-01', periods=100, freq='H'),
    "Region": ["Nampula", "Maputo", "Beira", "Nacala", "Pemba"] * 20,
    "Id Number": np.random.randint(100000, 999999, size=100),
    "Doctor Name": ["Dr. Smith", "Dr. Johnson", "Dr. Lee", "Dr. Brown", "Dr. Green"] * 20,
    "Doctor Number": np.random.randint(100000000, 999999999, size=100),
    "Practice Number": np.random.randint(1000, 5000, size=100),
    "Patient Name": [f"Patient {i}" for i in range(1, 101)],
    "Tests": ["Blood Test", "X-ray", "ECG", "MRI", "Ultrasound"] * 20,
    "Profiles": ["Profile 1", "Profile 2", "Profile 3", "Profile 4", "Profile 5"] * 20
}

# Convert the dummy data to DataFrame
dummy_df = pd.DataFrame(dummy_data)

# Join the dummy data with the main dataframe
df = pd.concat([df, dummy_df], axis=1)

# Show first few rows
print(df.head())

# Basic info and missing value check
print(df.info())
print(df.describe())

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

# GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced']
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# GridSearch with 5-fold cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=2)

# Fit GridSearch to training data
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best parameters:", grid_search.best_params_)

# Best model
best_rf = grid_search.best_estimator_

# Predict and evaluate with best model
y_pred_best = best_rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_best))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# Assuming original full data
X_full = df.drop('outcome', axis=1)
y_full = df['outcome']

# Split full data into train_val (80%) and test (20%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

# Further split train_val into training and validation sets (e.g., 75/25 split)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# Now train your best model on X_train and y_train
best_rf.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = best_rf.predict(X_val)
print("Validation set performance:")
print(classification_report(y_val, y_val_pred))

# Evaluate on the test set (unseen data)
y_test_pred = best_rf.predict(X_test)
print(f"Test set performance:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Results with actual and predicted values
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred
})

# Filter to only include positive class (1)
positive_cases_df = results_df[results_df['Actual'] == 1]

# Display first few rows of positive cases
print(positive_cases_df.head())

# Save results to CSV
positive_cases_df.to_csv('positive_cases_predictions.csv', index=False)

# Additional functionality to predict on new data, filter by age group, and visualize predictions
import joblib
joblib.dump(best_rf, 'diabetes_model.pkl')
print("Best model saved to diabetes_model.pkl")
