# repair_weather_csv.py
import re, json
from pathlib import Path
import pandas as pd
from zoneinfo import ZoneInfo

# === CONFIG ===
WEATHER_CSV = r"C:/Users/Justin/Documents/jdulyanu-tire-deg/data/processed/weather_track_conditions.csv"
CIRCUITS_JSON = r"C:/Users/Justin/Documents/jdulyanu-tire-deg/data/metadata/circuits.json"
BACKUP = Path(WEATHER_CSV).with_suffix(".bak")

EXPECTED_COLS = [
    "circuit_id","timestamp_utc","local_time","local_hour",
    "temp_c","humidity_pct","precip_mm","wind_speed_ms",
    "weather_main","weather_desc","source","ingested_at"
]

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")

# 1) Load raw as text to preserve malformed lines
raw = Path(WEATHER_CSV).read_text(encoding="utf-8").splitlines()

# 2) Normalize header (trust EXPECTED_COLS)
rows = []
for i, line in enumerate(raw):
    if i == 0:
        # force header to EXPECTED_COLS
        continue
    # drop trailing lone comma
    if line.endswith(","):
        line = line[:-1]
    parts = line.split(",")

    # If first column looks like an ISO timestamp, shift right by 1 (old schema)
    if parts and ISO_RE.match(parts[0]):
        parts = [""] + parts  # missing circuit_id
    # Pad/trim to expected length
    if len(parts) < len(EXPECTED_COLS):
        parts += [""] * (len(EXPECTED_COLS)-len(parts))
    elif len(parts) > len(EXPECTED_COLS):
        parts = parts[:len(EXPECTED_COLS)]
    rows.append(parts)

df = pd.DataFrame(rows, columns=EXPECTED_COLS)

# 3) Type fixes
for c in ["temp_c","humidity_pct","precip_mm","wind_speed_ms"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["local_hour"] = pd.to_numeric(df["local_hour"], errors="coerce")

# 4) Fix timestamp_utc and local_time
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)

# Load circuit tz map from circuits.json (expects each circuit to have a "timezone")
with open(CIRCUITS_JSON, "r", encoding="utf-8") as f:
    circuits = json.load(f)

# Support list or dict formats
tz_by_circuit = {}
if isinstance(circuits, list):
    for item in circuits:
        cid = item.get("id") or item.get("circuit_id")
        tz  = item.get("timezone") or item.get("tz") or item.get("iana_tz")
        if cid and tz: tz_by_circuit[cid] = tz
elif isinstance(circuits, dict):
    # could be {circuit_id: {...}}
    for cid, item in circuits.items():
        tz  = None
        if isinstance(item, dict):
            tz = item.get("timezone") or item.get("tz") or item.get("iana_tz")
        if tz: tz_by_circuit[cid] = tz

def to_local(row):
    cid = row["circuit_id"] or None
    ts  = row["timestamp_utc"]
    if pd.isna(ts) or not cid: return pd.NaT, pd.NA
    tz = tz_by_circuit.get(cid)
    if not tz:
        return pd.NaT, pd.NA
    try:
        lt = ts.tz_convert(ZoneInfo(tz))
        return lt.isoformat(), lt.hour
    except Exception:
        return pd.NaT, pd.NA

# Recompute local_time/hour where missing or clearly wrong (UTC-ish)
needs_local = (
    df["local_time"].isna()
    | df["local_time"].eq("")
    | df["local_time"].str.endswith("+00:00", na=False)
)
loc_vals = df.loc[needs_local].apply(to_local, axis=1, result_type="expand")
df.loc[needs_local, ["local_time","local_hour"]] = loc_vals

# 5) Clean strings
for c in ["circuit_id","weather_main","weather_desc","source"]:
    df[c] = df[c].astype("string").str.strip()

# 6) Drop fully invalid rows (no timestamp)
df = df[~df["timestamp_utc"].isna()].copy()

# 7) Deduplicate on (circuit_id, timestamp_utc), keep the newest ingested_at if available
# Parse ingested_at for stable ordering
df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce", utc=True)
df.sort_values(["circuit_id","timestamp_utc","ingested_at"], inplace=True)
df = df.drop_duplicates(subset=["circuit_id","timestamp_utc"], keep="last")

# 8) Final sort (seasonal chronological)
df.sort_values(["timestamp_utc","circuit_id"], inplace=True)

# 9) Backup + write
Path(WEATHER_CSV).rename(BACKUP)
df.to_csv(WEATHER_CSV, index=False)
print(f"Repaired → {WEATHER_CSV}\nBackup   → {BACKUP}")
