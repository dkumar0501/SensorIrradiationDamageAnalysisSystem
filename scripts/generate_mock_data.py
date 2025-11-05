import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

np.random.seed(42)
n_samples = 100

# Dose range: 0 to 1 MGy (Gray)
dose = np.linspace(0, 1e6, n_samples)

# === PRE-IRRADIATION (baseline) ===
pre_signal = np.random.normal(1000, 20, n_samples)
pre_leakage = np.random.normal(0.5, 0.05, n_samples)
pre_noise = np.random.normal(5, 0.5, n_samples)

pre_df = pd.DataFrame({
    "dose": dose,
    "signal_amplitude": pre_signal,
    "leakage_current": pre_leakage,
    "noise_level": pre_noise
})

# === POST-IRRADIATION (degraded) ===
damage_factor = 1 - (dose / 1e6) * np.random.uniform(0.2, 0.6)  # 20–60% drop
post_signal = pre_signal * damage_factor
post_leakage = pre_leakage + (dose / 1e6) * np.random.uniform(2, 4)
post_noise = pre_noise + (dose / 1e6) * np.random.uniform(1, 3)

post_df = pd.DataFrame({
    "dose": dose,
    "signal_amplitude": post_signal,
    "leakage_current": post_leakage,
    "noise_level": post_noise
})

pre_df.to_csv("data/pre_irradiation.csv", index=False)
post_df.to_csv("data/post_irradiation.csv", index=False)

print("✅ Synthetic IRRAD-like data generated successfully!")
