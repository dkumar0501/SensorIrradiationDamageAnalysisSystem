"""
predict_degradation.py
-------------------------------------------------------
Loads the trained model and performs degradation predictions
on unseen irradiation test data.

Part of: AutomatedIrradiationDamageAssessmentSystem
Author : Deepanshu Kumar
"""

import pandas as pd
import joblib

# Load model
model = joblib.load("../models/degradation_predictor.pkl")

# Load new unseen sensor data
new_data = pd.read_csv("../data/new_irradiation_test.csv")

# Ensure features match the training ones
features = ["dose", "leakage_current_change", "capacitance_change", "gain_drop"]
X_new = new_data[features]

# Predict degradation factor
new_data["predicted_damage_factor"] = model.predict(X_new)

# Save results
output_path = "../data/predicted_damage_results.csv"
new_data.to_csv(output_path, index=False)

print(f"✅ Predictions saved to: {output_path}")
