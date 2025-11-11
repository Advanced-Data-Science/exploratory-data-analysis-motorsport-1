# Drops source and session_rank from laps.csv
# Drops laps that have track_status != green (can be changed), and compound_code = TEST_UNKNOWN/UNKNOWN
# Creates new laps_v2.csv

import pandas as pd
from pathlib import Path
import numpy as np

# ---- paths ----
p_in  = Path(r"C:\Users\Justin\Documents\jdulyanu-tire-deg\data\processed\laps.csv")
p_out = p_in.with_name("laps_v2.csv")

# ---- configurable strictness ----
# Set to True for STRICT (only green laps), False for RELAXED (keeps yellow/VSC laps)
STRICT_MODE = True

# ---- load ----
df = pd.read_csv(p_in)
start_rows = len(df)

# ---- drop redundant columns ----
df = df.drop(columns=["source", "session_rank"], errors="ignore")

# ---- normalize session labels ----
def norm_session(s):
    s = str(s or "").strip().lower()
    if s in ("practice 1", "fp1"): return "FP1"
    if s in ("practice 2", "fp2"): return "FP2"
    if s in ("practice 3", "fp3"): return "FP3"
    return s.upper()
df["session"] = df["session"].map(norm_session)

# ---- coerce numeric types ----
num_cols = ["stint_number","tyre_age_laps","lap_number","lap_time_s"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- timestamp conversion ----
if "timestamp_utc" in df.columns:
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)

# ---- drop invalid compounds ----
bad_compounds = {"TEST_UNKNOWN", "UNKNOWN", "PROTOTYPE", "", None, np.nan}
before = len(df)
df = df[~df["compound_code"].astype(str).str.upper().isin(bad_compounds)]
dropped_compound = before - len(df)

# ---- filter lap time sanity ----
before = len(df)
df = df[df["lap_time_s"].apply(lambda x: np.isfinite(x) and 40 <= float(x) <= 200)]
dropped_laptime = before - len(df)

# ---- track status filtering ----
# FastF1: 1=green, 2=yellow, 4=SC, 5=VSC, 6=red, etc.
def track_ok(ts):
    if pd.isna(ts): return True
    ts_str = str(ts).strip()
    if STRICT_MODE:
        # keep only green or NA
        return ts_str in ("", "1")
    else:
        # relaxed: keep green, yellow, VSC
        return ts_str in ("", "1", "2", "5")

before = len(df)
df = df[df["track_status"].apply(track_ok)]
dropped_status = before - len(df)

# ---- reorder columns ----
col_order = [
    "year","gp_name","session","circuit_id","driver","team",
    "stint_number","compound_code","tyre_age_laps","lap_number",
    "lap_time_s","track_status","is_pb","timestamp_utc"
]
df = df[[c for c in col_order if c in df.columns]].copy()

# ---- export ----
df.to_csv(p_out, index=False)

print("✅ Cleaned laps saved:", p_out)
print(f"Rows start: {start_rows}")
print(f"Dropped (bad compound): {dropped_compound}")
print(f"Dropped (lap_time_s sanity): {dropped_laptime}")
print(f"Dropped (non-green track): {dropped_status}")
print(f"Rows final: {len(df)}")
print(f"STRICT_MODE = {STRICT_MODE}")
