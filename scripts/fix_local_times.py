# save as scripts/fix_local_timezones.py (or run in a notebook cell)

import pandas as pd
from zoneinfo import ZoneInfo
from pathlib import Path

# --- paths ---
csv_path = Path("data/processed/weather_track_conditions.csv")
backup_path = csv_path.with_suffix(".pre_localfix.bak.csv")

# --- circuit -> IANA timezone map ---
TZ = {
    # Middle East
    "sakhir": "Asia/Bahrain",
    "yas_marina": "Asia/Dubai",
    "lusail": "Asia/Qatar",
    "jeddah": "Asia/Riyadh",

    # Europe
    "silverstone": "Europe/London",
    "zandvoort": "Europe/Amsterdam",
    "spa": "Europe/Brussels",
    "monza": "Europe/Rome",
    "imola": "Europe/Rome",
    "monaco": "Europe/Monaco",
    "catalunya": "Europe/Madrid",
    "red_bull_ring": "Europe/Vienna",
    "hungaroring": "Europe/Budapest",
    "paul_ricard": "Europe/Paris",

    # Americas
    "miami": "America/New_York",
    "cota": "America/Chicago",
    "hermanos_rodriguez": "America/Mexico_City",
    "gilles_villeneuve": "America/Toronto",
    "interlagos": "America/Sao_Paulo",
    "las_vegas": "America/Los_Angeles",

    # Asia-Pacific
    "suzuka": "Asia/Tokyo",
    "shanghai": "Asia/Shanghai",
    "marina_bay": "Asia/Singapore",
    "baku": "Asia/Baku",
    "albert_park": "Australia/Melbourne",
}

# --- load & make a backup ---
df = pd.read_csv(csv_path)
df.to_csv(backup_path, index=False)

# --- ensure UTC-aware timestamp_utc ---
# robust to either tz-naive or tz-aware inputs
ts_utc = pd.to_datetime(df["timestamp_utc"], utc=True)

# --- compute local_time per-row using circuit-local tz ---
def row_local_time(ts, circ):
    tz = TZ.get(str(circ))
    if not tz:
        raise KeyError(f"Missing timezone mapping for circuit_id='{circ}'")
    return ts.tz_convert(ZoneInfo(tz))

df["local_time"] = [
    row_local_time(ts, circ) for ts, circ in zip(ts_utc, df["circuit_id"])
]

# --- derive local_hour as integer (0–23) ---
df["local_hour"] = df["local_time"].dt.hour.astype(int)

# (optional) keep timestamp_utc as UTC with explicit 'Z' formatting:
# df["timestamp_utc"] = ts_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# write back
df.to_csv(csv_path, index=False)

print(f"✅ Wrote localized times to {csv_path.name}")
print(f"🛟 Backup at {backup_path.name}")
print("Sample check:")
print(df.loc[df['circuit_id'].eq('zandvoort')].head(3)[["circuit_id","timestamp_utc","local_time","local_hour"]])
