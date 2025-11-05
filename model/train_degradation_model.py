"""
train_degradation_model.py
-------------------------------------------------------
Trains a Machine Learning model to predict irradiation-induced
sensor degradation based on pre- and post-irradiation parameters.

Part of: AutomatedIrradiationDamageAssessmentSystem
Author : Deepanshu Kumar
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ================================
# Load and Combine Data
# ================================
pre_data = pd.read_csv("../data/pre_irradiation.csv")
post_data = pd.read_csv("../data/post_irradiation.csv")

# Ensure consistent sorting and alignment
pre_data = pre_data.sort_values(by="sample_id").reset_index(drop=True)
post_data = post_data.sort_values(by="sample_id").reset_index(drop=True)

# Compute degradation as difference in key parameters
data = pd.DataFrame()
data["sample_id"] = pre_data["sample_id"]
data["dose"] = post_data["dose"]
data["leakage_current_change"] = post_data["leakage_current"] - pre_data["leakage_current"]
data["capacitance_change"] = post_data["capacitance"] - pre_data["capacitance"]
data["gain_drop"] = post_data["gain"] - pre_data["gain"]

# Target variable (damage factor)
data["damage_factor"] = (
    data["leakage_current_change"].abs() * 0.6 +
    data["capacitance_change"].abs() * 0.3 +
    data["gain_drop"].abs() * 0.1
)

# ================================
# Model Training
# ================================
X = data[["dose", "leakage_current_change", "capacitance_change", "gain_drop"]]
y = data["damage_factor"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# ================================
# Evaluation
# ================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ================================
# Save Model
# ================================
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/degradation_predictor.pkl")
print("📁 Model saved as '../models/degradation_predictor.pkl'")
