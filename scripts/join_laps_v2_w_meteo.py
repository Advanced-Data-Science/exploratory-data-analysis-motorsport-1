# scripts/join_laps_with_openmeteo.py
# Usage: python scripts/join_laps_with_openmeteo.py

import json
from pathlib import Path
import pandas as pd

# ==== CONFIG ====
PROJECT_ROOT = Path("C:/Users/Justin/Documents/jdulyanu-tire-deg")
PROCESSED = PROJECT_ROOT / "data" / "processed"

LAPS_IN   = PROCESSED / "laps_v2.csv"                          # cleaned laps
WX_IN     = PROCESSED / "weather_conditions_openmeteo_v2.csv"  # cleaned Open-Meteo
OUT_FILE  = PROCESSED / "joined_laps_weather.csv"

# Join behavior — tweak here if needed
WINDOW_MINUTES = 90            # +/- window; e.g., 60 or 120 also fine
MERGE_DIRECTION = "nearest"    # "nearest" or "backward" (causal-only)

CIRCUITS_JSON = PROJECT_ROOT / "data" / "metadata" / "circuits.json"

# Fallback GP → circuit_id map (used only if circuits.json aliases don't cover it)
FALLBACK_EVENT_TO_CIRCUIT = {
    "Bahrain Grand Prix": "sakhir",
    "Saudi Arabian Grand Prix": "jeddah",
    "Australian Grand Prix": "albert_park",
    "Emilia Romagna Grand Prix": "imola",
    "Miami Grand Prix": "miami",
    "Spanish Grand Prix": "catalunya",
    "Monaco Grand Prix": "monaco",
    "Azerbaijan Grand Prix": "baku",
    "Canadian Grand Prix": "gilles_villeneuve",
    "British Grand Prix": "silverstone",
    "Austrian Grand Prix": "red_bull_ring",
    "Hungarian Grand Prix": "hungaroring",
    "Belgian Grand Prix": "spa",
    "Dutch Grand Prix": "zandvoort",
    "Italian Grand Prix": "monza",
    "Singapore Grand Prix": "marina_bay",
    "Japanese Grand Prix": "suzuka",
    "United States Grand Prix": "cota",
    "Mexico City Grand Prix": "hermanos_rodriguez",
    "São Paulo Grand Prix": "interlagos",
    "Abu Dhabi Grand Prix": "yas_marina",
    "Qatar Grand Prix": "lusail",
    "Las Vegas Grand Prix": "las_vegas",
    "Chinese Grand Prix": "shanghai",
    "French Grand Prix": "paul_ricard",
}

def build_event_to_circuit_from_aliases(circuits_path: Path) -> dict:
    """
    Create a case-insensitive map from EventName (e.g., 'Bahrain Grand Prix')
    to circuit_id using circuits.json aliases.
    """
    if not circuits_path.exists():
        return {}
    data = json.loads(circuits_path.read_text(encoding="utf-8"))
    m = {}
    for c in data:
        cid = c.get("circuit_id")
        for a in (c.get("aliases") or []):
            key = str(a).strip().lower()
            if key:
                m[key] = cid
    return m

def map_gp_to_circuit_id(df_laps: pd.DataFrame, circuits_path: Path) -> pd.DataFrame:
    alias_map = build_event_to_circuit_from_aliases(circuits_path)

    def resolve(gp_name: str):
        if not isinstance(gp_name, str):
            return None
        k = gp_name.strip().lower()
        return alias_map.get(k) or FALLBACK_EVENT_TO_CIRCUIT.get(gp_name)

    df = df_laps.copy()
    if "circuit_id" not in df.columns or df["circuit_id"].isna().any():
        df["circuit_id"] = df.get("circuit_id")
        needs = df["circuit_id"].isna() | (df["circuit_id"].astype(str).str.len() == 0)
        df.loc[needs, "circuit_id"] = df.loc[needs, "gp_name"].map(resolve)
    return df

def main():
    # --- Load data
    laps = pd.read_csv(LAPS_IN)
    wx   = pd.read_csv(WX_IN)

    # --- Ensure circuit_id present on laps
    laps = map_gp_to_circuit_id(laps, CIRCUITS_JSON)
    missing_cid = laps[laps["circuit_id"].isna()][["year","gp_name","session"]].drop_duplicates()
    if not missing_cid.empty:
        print("WARNING: Some laps could not be mapped to circuit_id (will drop on join):")
        print(missing_cid.head(20))

    # --- Normalize circuit_id text on both frames
    for df in (laps, wx):
        df["circuit_id"] = df["circuit_id"].astype(str).str.strip().str.lower()

    # --- Build timestamp columns (UTC) from actual source cols
    lap_ts_col = "timestamp_utc_lap" if "timestamp_utc_lap" in laps.columns else "timestamp_utc"
    if lap_ts_col not in laps.columns:
        raise KeyError(f"Need '{lap_ts_col}' in laps. Got: {list(laps.columns)}")
    if "timestamp_utc" not in wx.columns:
        raise KeyError(f"Weather must have 'timestamp_utc'. Got: {list(wx.columns)}")

    laps["_lap_ts"] = pd.to_datetime(laps[lap_ts_col], errors="coerce", utc=True)
    wx["_wx_ts"]    = pd.to_datetime(wx["timestamp_utc"], errors="coerce", utc=True)

    # Make both tz-naive in UTC (merge_asof prefers naive aligned time)
    if isinstance(laps["_lap_ts"].dtype, pd.DatetimeTZDtype):
        laps["_lap_ts"] = laps["_lap_ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        laps["_lap_ts"] = laps["_lap_ts"].dt.tz_localize(None)

    if isinstance(wx["_wx_ts"].dtype, pd.DatetimeTZDtype):
        wx["_wx_ts"] = wx["_wx_ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        wx["_wx_ts"] = wx["_wx_ts"].dt.tz_localize(None)

    # Drop any rows with missing keys
    laps = laps.dropna(subset=["circuit_id", "_lap_ts"]).copy()
    wx   = wx.dropna(subset=["circuit_id", "_wx_ts"]).copy()

    # --- Robust per-circuit asof join (avoids "left keys must be sorted")
    wanted_cols = ["circuit_id", "_wx_ts", "temp_c", "humidity_pct", "precip_mm", "wind_speed_ms"]
    wx_small = wx[wanted_cols].copy()

    frames = []
    miss_counts = {}

    for cid, laps_g in laps.groupby("circuit_id", sort=False):
        lg = laps_g.sort_values("_lap_ts", kind="mergesort").copy()
        wg = wx_small[wx_small["circuit_id"] == cid].sort_values("_wx_ts", kind="mergesort").copy()

        if lg.empty or wg.empty:
            continue

        merged_g = pd.merge_asof(
            left=lg,
            right=wg,
            left_on="_lap_ts",
            right_on="_wx_ts",
            tolerance=pd.Timedelta(minutes=WINDOW_MINUTES),
            direction=MERGE_DIRECTION,
            allow_exact_matches=True
        )

        merged_g["matched_weather"] = merged_g["_wx_ts"].notna()
        merged_g["weather_time_delta_sec"] = (merged_g["_lap_ts"] - merged_g["_wx_ts"]).dt.total_seconds()

        miss_counts[cid] = int((~merged_g["matched_weather"]).sum())
        frames.append(merged_g)

    if not frames:
        raise RuntimeError("No groups merged. Check circuit_id mapping and timestamps.")

    merged = pd.concat(frames, ignore_index=True)

    # Keep only rows that found weather within the window
    out = merged[merged["matched_weather"]].copy()

    # Optional column drop (clean final modeling table)
    for c in ["_lap_ts", "_wx_ts", "weather_main", "weather_desc", "source", "ingested_at", "matched_weather"]:
        if c in out.columns:
            out.drop(columns=c, inplace=True)

    # --- Save
    out.to_csv(OUT_FILE, index=False)
    print(f"Joined rows: {len(out)} (from {len(laps)} laps). Saved -> {OUT_FILE}")

    # Simple QA report for unmatched counts
    bad = {k: v for k, v in miss_counts.items() if v}
    if bad:
        print("Unmatched rows (outside tolerance) by circuit_id:", bad)

if __name__ == "__main__":
    main()
