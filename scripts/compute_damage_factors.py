"""
compute_damage_factors.py
---------------------------------
This script computes key radiation-induced degradation metrics
for sensors exposed to ionizing radiation. It compares pre- and
post-irradiation data and calculates quantitative damage factors.

Part of: Automated Irradiation Damage Assessment System (AIDAS)
Author: [Your Name]
"""

import pandas as pd
import numpy as np

# === Load the merged dataset ===
df = pd.read_csv("data/merged.csv")

# === Compute degradation metrics ===

# Signal degradation ratio (post/pre)
df["signal_retention"] = df["signal_amplitude_post"] / df["signal_amplitude_pre"]

# Leakage current increase factor
df["leakage_increase"] = df["leakage_current_post"] / df["leakage_current_pre"]

# Noise growth factor
df["noise_growth"] = df["noise_level_post"] / df["noise_level_pre"]

# Combined damage index (weighted mean)
# Example: signal loss = 0.6 weight, leakage = 0.3, noise = 0.1
df["damage_index"] = (
    (1 - df["signal_retention"]) * 0.6 +
    (df["leakage_increase"] - 1) * 0.3 +
    (df["noise_growth"] - 1) * 0.1
)

# Handle negative or unrealistic values
df["damage_index"] = df["damage_index"].clip(lower=0)

# === Save results ===
output_path = "data/damage_factors.csv"
df.to_csv(output_path, index=False)

print("✅ Damage factors computed successfully!")
print(f"Saved to: {output_path}")

# === Optional summary statistics ===
print("\n📊 Summary of Computed Damage Metrics:")
print(df[["dose", "signal_retention", "leakage_increase", "noise_growth", "damage_index"]].describe())
