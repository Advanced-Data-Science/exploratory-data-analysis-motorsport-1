"""
backfill_weather_2024.py
Fetches and appends only missing 2024 FP1/FP2/FP3 weather rows
to data/processed/weather_track_conditions.csv using OpenWeather API.
Safe to re-run; deduplicates automatically.
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
import requests
import fastf1
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(".env"))

# ---------------- CONFIG ---------------- #
WX_PATH = Path("data/processed/weather_track_conditions.csv")
CIRCUITS_JSON = Path("data/metadata/circuits.json")
YEARS = [2024]
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # must be set in your .env
RATE_SLEEP = 1.0  # seconds between API calls
# ---------------------------------------- #
SESSION_NAMES = ["Practice 1", "Practice 2", "Practice 3"]  # sprint weekends only have Practice 1


def load_circuits():
    with CIRCUITS_JSON.open("r", encoding="utf-8") as f:
        return {c["circuit_id"]: c for c in json.load(f)}


def load_weather():
    if WX_PATH.exists():
        df = pd.read_csv(WX_PATH)
        if "circuit_id" not in df.columns:
            df["circuit_id"] = ""
        return df
    return pd.DataFrame(columns=[
        "circuit_id","timestamp_utc","local_time","local_hour",
        "temp_c","humidity_pct","precip_mm","wind_speed_ms",
        "weather_main","weather_desc","source","ingested_at"
    ])

def ensure_utc_aware(ts_iso: str) -> pd.Timestamp:
    """Return a tz-aware UTC Timestamp from an ISO string that may be tz-naive."""
    ts = pd.Timestamp(ts_iso)
    return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')

def find_fp_windows_from_schedule(target_years: list[int], circuits_by_id: dict[str, dict]) -> list[dict]:
    """
    Extract FP1/FP2/FP3 UTC start times directly from the FastF1 event schedule.
    Returns: list of dicts with circuit_id, gp_name, session_name, start_utc
    """
    windows = []

    for year in target_years:
        schedule = fastf1.events.get_event_schedule(year, include_testing=False)

        for _, ev in schedule.iterrows():
            gp_name = ev["EventName"]

            # find circuit_id by alias match
            circuit_id = None
            lower_gp = gp_name.lower()
            for cid, c in circuits_by_id.items():
                aliases = [a.lower() for a in c.get("aliases", [])]
                aliases.append(c.get("name", "").lower())
                if any(lower_gp in a or a in lower_gp for a in aliases):
                    circuit_id = cid
                    break
            if circuit_id is None:
                continue

            # Iterate through Practice sessions 1-3 if they exist
            for i in range(1, 4):
                s_name_col = f"Session{i}"
                s_time_col = f"Session{i}DateUtc"
                if s_name_col in ev and s_time_col in ev:
                    name = str(ev[s_name_col])
                    if "Practice" not in name:
                        continue
                    ts = ev[s_time_col]
                    if pd.notnull(ts):
                        windows.append({
                            "circuit_id": circuit_id,
                            "gp_name": gp_name,
                            "session_name": name,
                            "start_utc": pd.Timestamp(ts).isoformat()
                        })

    # Deduplicate
    key = lambda w: (w["circuit_id"], w["session_name"], w["start_utc"])
    seen, deduped = set(), []
    for w in windows:
        k = key(w)
        if k not in seen:
            seen.add(k)
            deduped.append(w)

    print(f"🗓  Found {len(deduped)} FP sessions across {len(target_years)} season(s)")
    return deduped


def find_fp_windows_from_laps(laps_csv: Path, circuits_by_id: dict[str, dict], years: list[int]) -> list[dict]:
    if not laps_csv.exists():
        return []
    laps = pd.read_csv(laps_csv, usecols=["year","gp_name","session","timestamp_utc"])
    laps = laps[laps["year"].isin(years)]
    laps = laps[laps["session"].isin(["Practice 1","Practice 2","Practice 3"])]
    # session start ≈ first lap timestamp seen
    starts = (laps
              .assign(ts=pd.to_datetime(laps["timestamp_utc"], utc=True, errors="coerce"))
              .dropna(subset=["ts"])
              .groupby(["gp_name","session"], as_index=False)["ts"].min())
    windows = []
    for _, r in starts.iterrows():
        gp_name = str(r["gp_name"])
        s_name = str(r["session"])
        ts_utc = r["ts"].isoformat()
        # gp → circuit_id via aliases
        circuit_id = None
        lower_gp = gp_name.lower()
        for cid, c in circuits_by_id.items():
            aliases = [a.lower() for a in c.get("aliases", [])]
            aliases.append(str(c.get("name","")).lower())
            if any(lower_gp in a or a in lower_gp for a in aliases):
                circuit_id = cid
                break
        if circuit_id:
            windows.append({
                "circuit_id": circuit_id,
                "gp_name": gp_name,
                "session_name": s_name,
                "start_utc": ts_utc
            })
    # Dedup
    seen, deduped = set(), []
    for w in windows:
        k = (w["circuit_id"], w["session_name"], w["start_utc"])
        if k not in seen:
            seen.add(k)
            deduped.append(w)
    return deduped


def fetch_openweather(lat, lon):
    """Fetch current weather at a lat/lon."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def make_row(circuit_id, ts_iso, local_tz, ow):
    ts_utc = ensure_utc_aware(ts_iso)               # UTC, tz-aware
    ts_local = ts_utc.tz_convert(local_tz)          # convert to circuit tz

    main = ow.get("weather", [{}])[0]
    wind = ow.get("wind", {})

    return {
        "circuit_id": circuit_id,
        "timestamp_utc": ts_utc.isoformat(),        # stays in UTC (with Z offset)
        "local_time": ts_local.isoformat(),         # localized timestamp
        "local_hour": float(ts_local.hour),
        "temp_c": float(ow.get("main",{}).get("temp", float("nan"))),
        "humidity_pct": float(ow.get("main",{}).get("humidity", float("nan"))),
        "precip_mm": float((ow.get("rain",{}) or {}).get("1h", 0.0)),
        "wind_speed_ms": float(wind.get("speed", 0.0)),
        "weather_main": str(main.get("main","")),
        "weather_desc": str(main.get("description","")),
        "source": "openweather.current",
        "ingested_at": pd.Timestamp.utcnow().isoformat()
    }

if __name__ == "__main__":
    assert OPENWEATHER_API_KEY, "⚠️  Set OPENWEATHER_API_KEY in your .env file!"

    circuits = load_circuits()
    wx = load_weather()

    # determine which circuit_id + timestamps exist
    have = set(zip(wx["circuit_id"], wx["timestamp_utc"]))

    windows = find_fp_windows_from_schedule(YEARS, circuits)
    if not windows:
        # fallback to lap-derived starts if aliasing/schedule parsing fails
        windows = find_fp_windows_from_laps(Path("data/processed/laps.csv"), circuits, YEARS)
    
    print(f"Planned FP windows from schedule/laps: {len(windows)}")
    out_rows = []

    print(f"Found {len(windows)} FP windows for {YEARS} to check.")

    for w in windows:
        circuit_id = w["circuit_id"]
        ts_iso = w["start_utc"]           # e.g., '2024-02-29T11:30:00'
        lat = float(circuits[circuit_id]["lat"])
        lon = float(circuits[circuit_id]["lon"])
        local_tz = circuits[circuit_id].get("timezone", "UTC")

        ow = fetch_openweather(lat, lon)
        new_row = make_row(circuit_id, ts_iso, local_tz, ow)

        try:
            ow = fetch_openweather(lat, lon)
            new_row = make_row(circuit_id, ts_iso, local_tz, ow)
            out_rows.append(new_row)
            print(f"✅ {circuit_id} @ {ts_iso}")
            time.sleep(RATE_SLEEP)
        except Exception as e:
            print(f"❌ Failed for {circuit_id} @ {ts_iso}: {e}")


        lat = float(circuits[circuit_id]["lat"])
        lon = float(circuits[circuit_id]["lon"])
        local_tz = circuits[circuit_id].get("timezone", "UTC")

        try:
            ow = fetch_openweather(lat, lon)
            new_row = make_row(circuit_id, ts_iso, local_tz, ow)
            out_rows.append(new_row)
            print(f"✅ {circuit_id} @ {ts_iso}")
            time.sleep(RATE_SLEEP)
        except Exception as e:
            print(f"❌ Failed for {circuit_id}: {e}")

    if out_rows:
        df_new = pd.DataFrame(out_rows)
        df_new.to_csv(WX_PATH, mode="a", index=False, header=not WX_PATH.exists())
        print(f"\n✅ Appended {len(df_new)} new rows to {WX_PATH}")
    else:
        print("\n✅ Nothing new to append.")
