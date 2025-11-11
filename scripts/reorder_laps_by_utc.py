import pandas as pd
from pathlib import Path

# --- Config ---
LAPS_PATH = Path(r"C:\Users\Justin\Documents\jdulyanu-tire-deg\data\processed\laps.csv")
OUT_PATH = LAPS_PATH.with_name("laps_chrono.csv")

print(f"📂 Loading {LAPS_PATH.name} ...")
df = pd.read_csv(LAPS_PATH, low_memory=False)
print(f"✅ Loaded {len(df):,} rows")

# --- Convert timestamps ---
if "timestamp_utc" not in df.columns:
    raise KeyError("❌ Missing 'timestamp_utc' column — cannot sort chronologically.")

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)

# --- Normalize session order ---
session_order = {
    "FP1": 1, "Practice 1": 1,
    "FP2": 2, "Practice 2": 2,
    "FP3": 3, "Practice 3": 3
}
df["session_rank"] = df["session"].map(session_order)

if df["session_rank"].isna().any():
    print("⚠️ Some sessions have unrecognized labels; they’ll sort last.")
    print(df.loc[df["session_rank"].isna(), "session"].value_counts())

# --- Sort chronologically ---
df_sorted = df.sort_values(
    by=["year", "timestamp_utc", "session_rank"],
    ascending=[True, True, True],
    na_position="last"
).reset_index(drop=True)

# --- Save chronologically ordered copy ---
df_sorted.to_csv(OUT_PATH, index=False)
print(f"✅ Wrote chronologically ordered file: {OUT_PATH}")
print(f"🔢 Rows: {len(df_sorted):,}")
print("🕒 First few entries:")
print(df_sorted[["year", "gp_name", "session", "timestamp_utc"]].head(10))
