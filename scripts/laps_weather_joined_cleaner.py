from pathlib import Path
import json
import pandas as pd
import numpy as np

JOINED = Path("data/processed/laps_weather_joined.csv")
CLEAN  = Path("data/processed/laps_weather_joined_clean.csv")
SCHEMA = Path("data/processed/laps_weather_joined_clean.schema.json")

# --- config ---
KEEP_COLS = [
    "year","gp_name","session","circuit_id",
    "driver","team","stint_number","compound_code",
    "tyre_age_laps","lap_number","lap_time_s",
    "track_status","is_pb","timestamp_utc_lap",
    "temp_c","humidity_pct","precip_mm","wind_speed_ms","local_hour",
    "weather_main","weather_time_delta_sec","matched_weather"
]

MIN_LAP_S, MAX_LAP_S = 10.0, 200.0
REQUIRE_MATCHED_WEATHER = True
MAX_ABS_WX_DT = 45 * 60  # seconds

# --- helper: normalize track status ---
def normalize_track_status(x):
    x = str(x).strip()
    mapping = {
        '1': 'Track clear',
        '2': 'Yellow flag',
        '4': 'Safety Car',
        '5': 'Red flag',
        '6': 'VSC deployed',
        '7': 'VSC ending'
    }
    if x in mapping:
        return mapping[x]
    elif any(ch in x for ch in mapping.keys()):
        # fallback: if it’s a concatenated string like “12” or “21”, take last valid code
        for ch in reversed(x):
            if ch in mapping:
                return mapping[ch]
        return 'Unknown'
    else:
        return 'Unknown'

def main():
    if not JOINED.exists():
        raise FileNotFoundError(f"Missing file: {JOINED}")

    df = pd.read_csv(JOINED)

    # ensure all expected columns exist
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"⚠️ Missing columns in joined file (will create as NA): {missing}")
        for c in missing:
            df[c] = np.nan

    df = df[KEEP_COLS].copy()

    # convert numeric columns
    num_cols = [
        "stint_number","tyre_age_laps","lap_number","lap_time_s",
        "temp_c","humidity_pct","precip_mm","wind_speed_ms",
        "local_hour","weather_time_delta_sec"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "timestamp_utc_lap" in df.columns:
        df["timestamp_utc_lap"] = pd.to_datetime(df["timestamp_utc_lap"], utc=True, errors="coerce")

    # normalize track status
    if "track_status" in df.columns:
        df["track_status_clean"] = df["track_status"].apply(normalize_track_status)

    # clean filters
    before = len(df)
    df = df[(df["lap_time_s"] > MIN_LAP_S) & (df["lap_time_s"] < MAX_LAP_S)]
    if REQUIRE_MATCHED_WEATHER and "matched_weather" in df.columns:
        df = df[df["matched_weather"] == True]
    if "weather_time_delta_sec" in df.columns:
        df = df[df["weather_time_delta_sec"].abs() <= MAX_ABS_WX_DT]
    after = len(df)

    # save clean csv + schema
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)

    schema = {
        "source": str(JOINED),
        "rows_before": int(before),
        "rows_after": int(after),
        "kept_columns": KEEP_COLS + ["track_status_clean"],
        "filters": {
            "lap_time_sec_range": [MIN_LAP_S, MAX_LAP_S],
            "require_matched_weather": REQUIRE_MATCHED_WEATHER,
            "max_abs_weather_delta_sec": MAX_ABS_WX_DT
        },
        "notes": "Adds track_status_clean (normalized flags). EDA-ready dataset."
    }
    with open(SCHEMA, "w") as f:
        json.dump(schema, f, indent=2)

    print("✅ Saved clean file:", CLEAN)
    print("ℹ️  Schema:", SCHEMA)
    print(f"Rows: {before} → {after}")
    print("Added column: track_status_clean (normalized)")

if __name__ == "__main__":
    main()
