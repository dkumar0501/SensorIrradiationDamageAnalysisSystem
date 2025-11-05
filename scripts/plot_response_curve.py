import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/merged.csv")

plt.figure(figsize=(8,6))
plt.plot(df["dose"], df["signal_amplitude_pre"], label="Pre-Irradiation")
plt.plot(df["dose"], df["signal_amplitude_post"], label="Post-Irradiation", linestyle="--")
plt.xlabel("Dose (Gy)")
plt.ylabel("Signal Amplitude (a.u.)")
plt.title("Detector Response Before vs After Irradiation")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.plot(df["dose"], df["leakage_current_post"] - df["leakage_current_pre"])
plt.xlabel("Dose (Gy)")
plt.ylabel("Leakage Current Increase (μA)")
plt.title("Leakage Current Increase vs Radiation Dose")
plt.grid(True)
plt.show()
