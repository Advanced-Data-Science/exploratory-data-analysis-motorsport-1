import pandas as pd
from pathlib import Path

# Paths
p_clean = Path(r"C:\Users\Justin\Documents\jdulyanu-tire-deg\data\processed\laps_weather_joined_clean.csv")
out = Path("data/processed/laps_valid.csv")

# Load data
df = pd.read_csv(p_clean)

# Define valid compounds and normal lap-time range
valid_compounds = ["SOFT", "MEDIUM", "HARD"]
min_lap_time = 60
max_lap_time = 120

# Apply filters
df_valid = df[
    (df["track_status_clean"] == "Track clear") &
    (df["compound_code"].isin(valid_compounds)) &
    (df["lap_time_s"].between(min_lap_time, max_lap_time)) &
    (df["matched_weather"])  # keep only laps with matched weather
].copy()

# Reset index and save
df_valid = df_valid.reset_index(drop=True)
df_valid.to_csv(out, index=False)

# Summary stats
print("✅ Saved valid laps file:", out)
print(f"Rows kept: {len(df_valid)} / {len(df)}")
print("Filters applied:")
print(f"  - track_status_clean == 'Track clear'")
print(f"  - compound_code in {valid_compounds}")
print(f"  - {min_lap_time}s <= lap_time_s <= {max_lap_time}s")
print("  - matched_weather == True")
