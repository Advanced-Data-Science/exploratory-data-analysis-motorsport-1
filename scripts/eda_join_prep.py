# eda_join_prep.py
# Validates timestamp alignment and joins laps ↔ weather for EDA (FP sessions only)

from pathlib import Path
import json
import pandas as pd
import numpy as np

# ========= CONFIG (edit if paths differ) =========
LAPS_CSV     = "data/processed/laps.csv"
WEATHER_CSV  = "data/processed/weather_track_conditions.csv"
OUTPUT_JOIN  = "data/processed/laps_weather_joined.csv"
QUALITY_JSON = "data/processed/laps_weather_join_quality.json"

ASOF_TOLERANCE = "45min"                 # e.g., "30min" / "1h"
VALID_FP = {"FP1", "FP2", "FP3"}         # filter laps to FP sessions

# EXACT strings from laps["gp_name"] → EXACT strings from weather["circuit_id"]
GP_TO_CIRCUIT = {
    "Bahrain Grand Prix": "sakhir",
    "Saudi Arabian Grand Prix": "jeddah",
    "Australian Grand Prix": "albert_park",
    "Azerbaijan Grand Prix": "baku",
    "Miami Grand Prix": "miami",
    "Emilia Romagna Grand Prix": "imola",
    "Monaco Grand Prix": "monaco",
    "Spanish Grand Prix": "catalunya",
    "Canadian Grand Prix": "gilles_villeneuve",
    "Austrian Grand Prix": "red_bull_ring",
    "British Grand Prix": "silverstone",
    "Hungarian Grand Prix": "hungaroring",
    "Belgian Grand Prix": "spa",
    "Dutch Grand Prix": "zandvoort",
    "Italian Grand Prix": "monza",
    "Singapore Grand Prix": "marina_bay",
    "Japanese Grand Prix": "suzuka",
    "Qatar Grand Prix": "lusail",
    "United States Grand Prix": "cota",
    "Mexico City Grand Prix": "hermanos_rodriguez",
    "São Paulo Grand Prix": "interlagos",
    "Las Vegas Grand Prix": "las_vegas",
    "Abu Dhabi Grand Prix": "yas_marina",
    "French Grand Prix": "paul_ricard",
    "Chinese Grand Prix": "shanghai"
}

# ========= HELPERS =========
def normalize_session_label(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip().lower().replace("-", " ").replace("_", " ")
    t = " ".join(t.split())
    if t in {"fp1", "p1", "practice 1", "free practice 1"}:
        return "FP1"
    if t in {"fp2", "p2", "practice 2", "free practice 2"}:
        return "FP2"
    if t in {"fp3", "p3", "practice 3", "free practice 3"}:
        return "FP3"
    return s

def _read_laps(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = [
        "year","gp_name","session","driver","team","stint_number","compound_code",
        "tyre_age_laps","lap_number","lap_time_s","track_status","is_pb",
        "timestamp_utc","source","session_rank"
    ]
    _missing = [c for c in needed if c not in df.columns]
    if _missing:
        raise ValueError(f"laps missing columns: {_missing}")
    # parse times (tz-aware UTC) + FP filter
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    # NEW: normalize session labels, then filter
    df["session_norm"] = df["session"].map(normalize_session_label)
    df = df[df["session_norm"].isin({"FP1","FP2","FP3"})].copy()

    # keep 'session' as normalized for downstream consistency
    df["session"] = df["session_norm"]
    df = df.drop(columns=["session_norm"])
    
    # map circuit
    df["circuit_id"] = df["gp_name"].map(GP_TO_CIRCUIT)
    if df["circuit_id"].isna().any():
        missing = sorted(df.loc[df["circuit_id"].isna(),"gp_name"].unique())
        print("⚠️ Unmapped gp_name (update GP_TO_CIRCUIT):", missing)
    # drop unusable
    df = df.dropna(subset=["timestamp_utc","circuit_id"]).copy()
    # rename ts for clarity and sort as required by merge_asof
    df = (df.rename(columns={"timestamp_utc":"timestamp_utc_lap"})
            .sort_values(["circuit_id","timestamp_utc_lap"], kind="mergesort")
            .reset_index(drop=True))
    return df

def _read_weather(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = [
        "circuit_id","timestamp_utc","local_time","local_hour","temp_c","humidity_pct",
        "precip_mm","wind_speed_ms","weather_main","weather_desc","source","ingested_at"
    ]
    _missing = [c for c in needed if c not in df.columns]
    if _missing:
        raise ValueError(f"weather missing columns: {_missing}")
    # parse times (tz-aware UTC)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["local_time"]    = pd.to_datetime(df["local_time"],    utc=True, errors="coerce")
    # drop unusable
    df = df.dropna(subset=["circuit_id","timestamp_utc"]).copy()
    # rename ts and sort as required by merge_asof
    df = (df.rename(columns={"timestamp_utc":"timestamp_utc_wx"})
            .sort_values(["circuit_id","timestamp_utc_wx"], kind="mergesort")
            .reset_index(drop=True))
    return df

def _event_overlap_audit(laps_sorted: pd.DataFrame, weather_sorted: pd.DataFrame) -> pd.DataFrame:
    # per-event lap window
    ev = (laps_sorted.assign(
            event_key=(laps_sorted["year"].astype(str) + "|" +
                       laps_sorted["gp_name"] + "|" +
                       laps_sorted["session"]))
          .groupby(["event_key","circuit_id"])["timestamp_utc_lap"]
          .agg(lap_min="min", lap_max="max")
          .reset_index())
    # per-circuit weather window
    wx = (weather_sorted.groupby("circuit_id")["timestamp_utc_wx"]
          .agg(wx_min="min", wx_max="max")
          .reset_index())
    audit = ev.merge(wx, on="circuit_id", how="left")
    # compute overlap seconds
    overlap_start = audit[["lap_min","wx_min"]].max(axis=1)
    overlap_end   = audit[["lap_max","wx_max"]].min(axis=1)
    overlap = (overlap_end - overlap_start)
    overlap = overlap.where(overlap > pd.Timedelta(0), pd.Timedelta(0))
    audit["overlap_sec"] = overlap.dt.total_seconds()
    audit["has_any_overlap"] = audit["overlap_sec"] > 0
    return audit

def _merge_asof_strict(laps_sorted: pd.DataFrame, weather_sorted: pd.DataFrame, tol: str) -> pd.DataFrame:
    # merge_asof requires both sides sorted by ["circuit_id", ts]
    # Do a defensive re-sort to be safe (stable)
    laps_sorted = laps_sorted.sort_values(["circuit_id","timestamp_utc_lap"], kind="mergesort").reset_index(drop=True)
    weather_sorted = weather_sorted.sort_values(["circuit_id","timestamp_utc_wx"], kind="mergesort").reset_index(drop=True)

    # per-circuit join (most robust if any oddities appear)
    pieces = []
    for cid, g_laps in laps_sorted.groupby("circuit_id", sort=False):
        g_wx = weather_sorted.loc[weather_sorted["circuit_id"] == cid]
        if g_wx.empty:
            out = g_laps.copy()
            # ensure expected weather columns exist with NaN
            for col in ["timestamp_utc_wx","temp_c","humidity_pct","precip_mm","wind_speed_ms",
                        "weather_main","weather_desc","source","ingested_at","local_time","local_hour"]:
                if col not in out.columns:
                    out[col] = pd.NA
        else:
            out = pd.merge_asof(
                left=g_laps.sort_values("timestamp_utc_lap"),
                right=g_wx.sort_values("timestamp_utc_wx"),
                left_on="timestamp_utc_lap",
                right_on="timestamp_utc_wx",
                direction="nearest",
                tolerance=pd.Timedelta(tol),
                suffixes=("", "_wx"),
            )
        pieces.append(out)

    merged = pd.concat(pieces, ignore_index=True)
    merged["weather_time_delta_sec"] = (
        (merged["timestamp_utc_lap"] - merged["timestamp_utc_wx"])
        .dt.total_seconds()
        .abs()
    )
    merged["matched_weather"] = merged["temp_c"].notna()
    return merged

def main():
    laps_sorted    = _read_laps(LAPS_CSV)
    weather_sorted = _read_weather(WEATHER_CSV)

    # Overlap audit (do weather windows cover session windows?)
    audit = _event_overlap_audit(laps_sorted, weather_sorted)
    total_events = int(audit["event_key"].nunique())
    events_with_overlap = int(audit["has_any_overlap"].sum())

    if (audit["has_any_overlap"] == False).any():
        print("⚠️ Some events have zero overlap with weather windows:")
        print(audit.loc[~audit["has_any_overlap"], ["event_key","circuit_id","lap_min","lap_max","wx_min","wx_max"]])

    # Perform asof merge (robust per-circuit)
    merged = _merge_asof_strict(laps_sorted, weather_sorted, ASOF_TOLERANCE)

    # Quality summary
    total = int(len(merged))
    matched = int(merged["matched_weather"].sum())
    pct = matched / total if total else 0.0

    delta = merged.loc[merged["matched_weather"], "weather_time_delta_sec"]
    delta_summary = {
        "count": int(delta.size),
        "p50_sec": float(delta.quantile(0.5)) if delta.size else None,
        "p90_sec": float(delta.quantile(0.9)) if delta.size else None,
        "max_sec": float(delta.max()) if delta.size else None,
    }

    # Any unmapped names still lingering (should be none after dropna earlier)?
    unmapped = sorted(
        laps_sorted.loc[laps_sorted["circuit_id"].isna(), "gp_name"].unique().tolist()
    ) if laps_sorted["circuit_id"].isna().any() else []

    quality = {
        "asof_tolerance": ASOF_TOLERANCE,
        "total_laps": total,
        "matched_laps": matched,
        "pct_matched": pct,
        "delta_summary": delta_summary,
        "unmapped_gp_names": unmapped,
        "events_overlap_summary": {
            "total_events": total_events,
            "events_with_overlap": events_with_overlap
        }
    }

    # Save outputs
    Path(OUTPUT_JOIN).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_JOIN, index=False)
    with open(QUALITY_JSON, "w") as f:
        json.dump(quality, f, indent=2, default=str)

    print("\n✅ Join complete.")
    print(f"→ Joined file: {OUTPUT_JOIN}")
    print(f"→ Quality report: {QUALITY_JSON}")
    print(f"Matched {matched}/{total} laps ({pct:.1%}) within tolerance {ASOF_TOLERANCE}.")
    print(f"Events with weather overlap: {events_with_overlap}/{total_events}")

if __name__ == "__main__":
    main()
