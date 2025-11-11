# quality_report.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# CONFIG — matches your folders and columns
# ------------------------------------------------------------------
CONFIG = {
    "laps_csv": "data/processed/laps.csv",
    "weather_csv": "data/processed/weather_track_conditions.csv",
    "report_json": "reports/quality_report_all.json",
    # time & filtering
    "weather_match_tolerance_min": 60,
    "min_lap_time_s": 30.0,     # seconds
    "max_lap_time_s": 200.0,    # seconds
    # columns in *your* files
    "lap_timestamp_col": "timestamp_utc",      # laps
    "weather_timestamp_col": "timestamp_utc",  # weather
    "session_keys": ["year", "gp_name", "session"],
    "dup_key": ["year", "gp_name", "session", "driver", "lap_number"],
    # weather fields for completeness/dist
    "weather_fields": ["temp_c", "humidity_pct", "wind_speed_ms", "precip_mm", "weather_main"],
}

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _to_datetime_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")

def _parse_csv(path):
    # Try with UTF-8 BOM handling first; fall back to default if needed
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path)
    # strip spaces and remove BOM characters from headers
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)  # BOM
    )
    return df

def _ensure_lap_timestamps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure per-lap UTC timestamps exist and vary within each session.
    If a session’s timestamps are constant or NaT, rebuild them by ordering
    laps by lap_number and cumulatively summing lap_time_s from an anchor.
    """
    ts_col = CONFIG["lap_timestamp_col"]

    # Guarantee the timestamp column exists & is datetime
    if ts_col not in laps.columns:
        laps[ts_col] = pd.NaT
    laps[ts_col] = _to_datetime_utc(laps[ts_col])

    can_rebuild = "lap_time_s" in laps.columns

    def rebuild_if_constant(g: pd.DataFrame) -> pd.DataFrame:
        # g is one session (defined by CONFIG["session_keys"])
        ts = g[ts_col]
        constant_ts = ts.notna().all() and (ts.max() - ts.min() == pd.Timedelta(0))
        no_ts = ts.isna().all()
        needs = (constant_ts or no_ts) and can_rebuild

        if not needs:
            return g

        # anchor: first real timestamp if present, else epoch
        anchor = ts.dropna().iloc[0] if ts.notna().any() else pd.Timestamp("1970-01-01", tz="UTC")

        # order by lap_number if available
        if "lap_number" in g.columns:
            g = g.sort_values("lap_number")

        # cumulative elapsed time from start of session (sec)
        elapsed = pd.to_timedelta(
            pd.to_numeric(g["lap_time_s"], errors="coerce").fillna(0).cumsum().shift(fill_value=0),
            unit="s"
        )
        g[ts_col] = anchor + elapsed
        return g

    # SAFE: no .apply on groupby; iterate and concat back
    parts = []
    for _, g in laps.groupby(CONFIG["session_keys"], sort=False, group_keys=False):
        parts.append(rebuild_if_constant(g))
    fixed = pd.concat(parts, ignore_index=False)

    # sanity: ensure keys still present
    assert all(k in fixed.columns for k in CONFIG["session_keys"]), \
        f"Grouping columns missing after rebuild: {CONFIG['session_keys']}"

    return fixed

def _filter_and_tag_valid_laps(laps):
    lt = pd.to_numeric(laps.get("lap_time_s", np.nan), errors="coerce")
    valid = (lt >= CONFIG["min_lap_time_s"]) & (lt <= CONFIG["max_lap_time_s"])
    return laps.assign(_valid_lap=valid.fillna(False))

def _session_windows(laps):
    ts_col = CONFIG["lap_timestamp_col"]
    g = laps[laps[ts_col].notna()].groupby(CONFIG["session_keys"])
    win = g[ts_col].agg(["min", "max"]).reset_index().rename(columns={"min": "session_start_utc", "max": "session_end_utc"})
    win["session_mid_utc"] = win["session_start_utc"] + (win["session_end_utc"] - win["session_start_utc"]) / 2
    return win

def _weather_coverage_by_session(session_win, weather):
    tol = pd.to_timedelta(CONFIG["weather_match_tolerance_min"], unit="m")
    wts = weather[CONFIG["weather_timestamp_col"]].sort_values()
    rows = []
    for r in session_win.itertuples(index=False):
        lower = r.session_start_utc - tol
        upper = r.session_end_utc + tol
        cnt = ((wts >= lower) & (wts <= upper)).sum()
        rows.append({
            "year": getattr(r, CONFIG["session_keys"][0]),
            "gp_name": getattr(r, CONFIG["session_keys"][1]),
            "session": getattr(r, CONFIG["session_keys"][2]),
            "weather_rows_in_window": int(cnt),
            "has_weather": bool(cnt > 0)
        })
    return pd.DataFrame(rows)

def _joinability_rate(cov_df):
    checked = len(cov_df)
    ok = int(cov_df["has_weather"].sum())
    miss = cov_df.loc[~cov_df["has_weather"], ["year", "gp_name", "session"]].to_dict(orient="records")
    return (ok / checked if checked else 0.0), checked, miss

def _missingness(df, cols):
    out = {"missing_counts": {}, "missing_rates": {}, "total_records": int(len(df))}
    for c in cols:
        if c in df.columns:
            m = df[c].isna().sum()
            out["missing_counts"][c] = int(m)
            out["missing_rates"][c] = float(m / len(df)) if len(df) else 0.0
    return out

def _distributions(df, cols):
    out = {}
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            s = s.dropna()
            if len(s):
                out[c] = {"min": float(s.min()), "max": float(s.max()), "mean": float(s.mean())}
    return out

def _duplicates(laps):
    key = [k for k in CONFIG["dup_key"] if k in laps.columns]
    if not key:
        return {"composite_key": CONFIG["dup_key"], "duplicate_rows": None, "note": "Key columns missing"}
    return {"composite_key": key, "duplicate_rows": int(laps.duplicated(subset=key).sum())}

def _anomalies(laps, weather):
    issues = {"count": 0, "examples": []}
    if "lap_time_s" in laps.columns:
        lt = pd.to_numeric(laps["lap_time_s"], errors="coerce")
        bad = (lt < CONFIG["min_lap_time_s"]) | (lt > CONFIG["max_lap_time_s"])
        if bad.any():
            cnt = int(bad.sum()); issues["count"] += cnt
            ex = laps.loc[bad, CONFIG["session_keys"] + ["driver", "lap_number", "lap_time_s"]].head(5).to_dict(orient="records")
            issues["examples"].extend(ex)
    if "humidity_pct" in weather.columns:
        h = pd.to_numeric(weather["humidity_pct"], errors="coerce")
        bad_h = (h < 0) | (h > 100)
        if bad_h.any():
            cnt = int(bad_h.sum()); issues["count"] += cnt
            ex = weather.loc[bad_h, ["timestamp_utc", "humidity_pct"]].head(5).to_dict(orient="records")
            issues["examples"].extend(ex)
    return issues

def _normalize_lap_columns(laps: pd.DataFrame) -> pd.DataFrame:
    """Rename common alternates to expected names."""
    rename_map = {
        # year
        "season": "year", "Season": "year", "Year": "year",
        # grand prix name
        "grand_prix": "gp_name", "GrandPrix": "gp_name", "race_name": "gp_name",
        "raceName": "gp_name", "event_name": "gp_name", "gp": "gp_name",
        # session name
        "session_name": "session", "Session": "session",
        "sessionType": "session", "session_type": "session",
        "session_code": "session",
        # timing
        "laptime_s": "lap_time_s", "time_s": "lap_time_s",
        "ts_utc": "timestamp_utc", "TimestampUTC": "timestamp_utc",
    }
    to_rename = {k: v for k, v in rename_map.items() if k in laps.columns}
    if to_rename:
        laps = laps.rename(columns=to_rename)
    return laps

def _ensure_session_keys(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'year', 'gp_name', 'session' exist.
    Try to derive from timestamp_utc or other columns if missing.
    """
    laps = _normalize_lap_columns(laps)

    # Derive YEAR
    if "year" not in laps.columns or laps["year"].isna().all():
        if "timestamp_utc" in laps.columns:
            ts = pd.to_datetime(laps["timestamp_utc"], utc=True, errors="coerce")
            if ts.notna().any():
                laps["year"] = ts.dt.year
        # if still missing, try integer season-ish columns
        for alt in ["Season", "SEASON"]:
            if "year" not in laps.columns and alt in laps.columns:
                laps["year"] = pd.to_numeric(laps[alt], errors="coerce").astype("Int64")

    # Derive GP_NAME
    if "gp_name" not in laps.columns or laps["gp_name"].isna().all():
        # Try common alternates already normalized in _normalize_lap_columns
        for alt in ["circuit_name", "event", "race", "venue", "Circuit", "Event"]:
            if alt in laps.columns and laps[alt].notna().any():
                laps["gp_name"] = laps[alt]
                break
        # Last resort: use circuit_id as a stand-in label
        if "gp_name" not in laps.columns and "circuit_id" in laps.columns:
            laps["gp_name"] = laps["circuit_id"]
        # If still missing, create a placeholder so grouping works
        if "gp_name" not in laps.columns:
            laps["gp_name"] = "UNKNOWN_GP"

    # Derive SESSION
    if "session" not in laps.columns or laps["session"].isna().all():
        # Try alternates (already normalized): session_type, session_code, etc.
        for alt in ["SessionType", "type", "sessionid", "session_code"]:
            if alt in laps.columns and laps[alt].notna().any():
                laps["session"] = laps[alt]
                break
        # Try to infer from typical FastF1 fields (e.g., if only FP2 laps present)
        if "session" not in laps.columns:
            # If there’s a single unique value in something like 'session_name', normalize earlier.
            laps["session"] = "UNKNOWN_SESSION"

    # Final sanity check — fail fast with helpful info
    missing = [k for k in ["year", "gp_name", "session"] if k not in laps.columns]
    if missing:
        cols = list(laps.columns)
        raise KeyError(
            f"Missing required keys {missing}. Columns present: {cols[:25]}..."
            " Add a rename in _normalize_lap_columns or populate these fields in laps.csv."
        )

    return laps


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    Path("reports").mkdir(parents=True, exist_ok=True)

    laps = _parse_csv(CONFIG["laps_csv"])
    weather = _parse_csv(CONFIG["weather_csv"])

    # OPTIONAL: quick debug
    print("[DEBUG] laps cols:", list(laps.columns)[:20])

    # parse datetimes
    laps[CONFIG["lap_timestamp_col"]] = _to_datetime_utc(laps.get(CONFIG["lap_timestamp_col"]))
    weather[CONFIG["weather_timestamp_col"]] = _to_datetime_utc(weather.get(CONFIG["weather_timestamp_col"]))

    laps = _normalize_lap_columns(laps)
    
    # rebuild constant/noon lap timestamps per session using cumulative lap_time_s
    laps = _ensure_lap_timestamps(laps)
    # tag valid laps
    laps = _filter_and_tag_valid_laps(laps)
    
    print("[DEBUG] keys check:",
      "year" in laps.columns,
      "gp_name" in laps.columns,
      "session" in laps.columns)
    # print("[DEBUG] head:", laps[["year","gp_name","session"]].head(3))

    # session windows from (fixed) timestamps
    win = _session_windows(laps)
    cov = _weather_coverage_by_session(win, weather[weather[CONFIG["weather_timestamp_col"]].notna()])

    join_rate, checked_sessions, missing_sessions = _joinability_rate(cov)

    # coverage by year (using what’s actually in laps)
    seasons = {}
    for y, g in win.groupby("year"):
        seasons[str(int(y))] = {
            "events_expected": None,  # optional (fill later via Ergast)
            "sessions_expected": int(g["session"].nunique()),
            "sessions_with_any_weather_row": int(cov.loc[cov["year"] == y, "has_weather"].sum())
        }

    totals = {
        "laps_rows": int(len(laps)),
        "weather_rows": int(len(weather)),
        "sessions_with_weather_any": int(cov["has_weather"].sum())
    }

    completeness = _missingness(weather, CONFIG["weather_fields"])
    distributions = _distributions(weather, ["temp_c", "humidity_pct", "wind_speed_ms", "precip_mm"])
    dup_info = _duplicates(laps)
    anomalies = _anomalies(laps, weather)

    report = {
        "summary": {
            "total_records": totals["weather_rows"],
            "collection_success_rate": None,
            "overall_quality_score": None,
            "years": sorted(list(seasons.keys())),
        },
        "coverage": {
            "seasons": seasons,
            "unmapped_events": {},
            "totals": totals,
        },
        "joinability": {
            "joinable_rate": float(join_rate),
            "missing_sessions": missing_sessions,
            "checked_sessions": int(checked_sessions),
            "tolerance_min": CONFIG["weather_match_tolerance_min"],
        },
        "completeness_analysis": completeness,
        "data_distribution": distributions,
        "duplicates": dup_info,
        "anomaly_detection": anomalies,
        "recommendations": [
            f"Join tolerance is ±{CONFIG['weather_match_tolerance_min']} min; increase if joinable_rate < 0.95.",
            "If `precip_mm` is missing while `weather_main` ∈ {Clear, Clouds, Mist}, impute 0 (dry).",
            "Drop duplicate laps by composite key: " + ", ".join(CONFIG["dup_key"]),
            f"Exclude invalid laps with lap_time_s outside [{CONFIG['min_lap_time_s']}, {CONFIG['max_lap_time_s']}] seconds.",
        ],
    }

    Path(CONFIG["report_json"]).parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG["report_json"], "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[OK] Wrote report → {CONFIG['report_json']}")
    print(f"Laps: {totals['laps_rows']:,} | Weather rows: {totals['weather_rows']:,}")
    print(f"Joinable sessions: {report['joinability']['joinable_rate']:.3f} ({checked_sessions} checked)")

if __name__ == "__main__":
    main()
