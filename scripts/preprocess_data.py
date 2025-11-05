import pandas as pd

pre_df = pd.read_csv("data/pre_irradiation.csv")
post_df = pd.read_csv("data/post_irradiation.csv")

# Normalize for analysis
pre_df["signal_norm"] = pre_df["signal_amplitude"] / pre_df["signal_amplitude"].max()
post_df["signal_norm"] = post_df["signal_amplitude"] / post_df["signal_amplitude"].max()

merged_df = pre_df.merge(post_df, on="dose", suffixes=("_pre", "_post"))
merged_df.to_csv("data/merged.csv", index=False)

print("✅ Preprocessing complete! Saved merged.csv")
