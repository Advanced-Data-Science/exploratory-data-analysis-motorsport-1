import pandas as pd
from pathlib import Path

WX_PATH = Path("data/processed/weather_track_conditions.csv")
CLEAN_PATH = Path("data/processed/weather_track_conditions_clean.csv")
LEGACY_PATH = Path("data/processed/weather_track_conditions_legacy_no_circuit_id.csv")

df = pd.read_csv(WX_PATH)
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# --- Drop rows missing circuit_id --- #
legacy = df[df["circuit_id"].isna() | (df["circuit_id"] == "")]
df = df.dropna(subset=["circuit_id"])
df = df[df["circuit_id"] != ""]

# --- Deduplicate on (circuit_id, timestamp_utc) --- #
before = len(df)
df = df.drop_duplicates(subset=["circuit_id", "timestamp_utc"])
after = len(df)
print(f"✅ Removed {before - after} duplicate rows")

# --- Save outputs --- #
df.to_csv(CLEAN_PATH, index=False)
legacy.to_csv(LEGACY_PATH, index=False)
print(f"Kept {after} good rows → {CLEAN_PATH.name}")
print(f"Parked {len(legacy)} legacy rows → {LEGACY_PATH.name}")