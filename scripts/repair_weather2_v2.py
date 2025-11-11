# repair_weather2_v2_2.py
import os, sys, time, re
import pandas as pd
from datetime import datetime
import pytz

PATH = r"data/processed/weather_track_conditions.csv"

def ts():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def main():
    print(f"[{ts()}] Working dir:", os.getcwd())
    if not os.path.exists(PATH):
        print(f"[{ts()}] ERROR: File not found: {PATH}")
        sys.exit(1)

    before_mtime = os.path.getmtime(PATH)
    print(f"[{ts()}] Target file:", os.path.abspath(PATH))
    print(f"[{ts()}] Last modified:", time.ctime(before_mtime))

    # 1) Load & normalize
    df = pd.read_csv(PATH, dtype=str)
    n_rows = len(df)
    df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)
    df.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA}, inplace=True)

    for col in ["local_hour", "temp_c", "humidity_pct", "precip_mm", "wind_speed_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["timestamp_utc", "ingested_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Baseline stats
    blanks_circuit = df["circuit_id"].isna().sum() if "circuit_id" in df.columns else 0
    dup_key = df.duplicated(subset=["circuit_id","timestamp_utc"], keep=False).sum() if {"circuit_id","timestamp_utc"} <= set(df.columns) else 0
    print(f"[{ts()}] Rows: {n_rows}")
    print(f"[{ts()}] Blank circuit_id rows: {blanks_circuit}")
    print(f"[{ts()}] Rows in duplicate key groups: {dup_key}")

    # 2) Fill circuit_id by timestamp propagation + dedupe
    if {"timestamp_utc", "ingested_at"} <= set(df.columns):
        id_fill = df.groupby("timestamp_utc")["circuit_id"].transform(
            lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA
        )
        newly_filled = df["circuit_id"].isna() & id_fill.notna()
        df.loc[newly_filled, "circuit_id"] = id_fill[newly_filled]
        print(f"[{ts()}] Filled circuit_id for {newly_filled.sum()} rows.")

        nonnull_count = df.notna().sum(axis=1).rename("nonnull_count")
        df = pd.concat([df, nonnull_count], axis=1)
        df = df.sort_values(
            by=["circuit_id", "timestamp_utc", "ingested_at", "nonnull_count"],
            ascending=[True, True, False, False]
        )
        before = len(df)
        df = df.drop_duplicates(subset=["circuit_id","timestamp_utc"], keep="first")
        df.drop(columns=["nonnull_count"], inplace=True, errors="ignore")
        print(f"[{ts()}] Deduplicated {before - len(df)} rows.")

    # 3) Backfill missing local_time / local_hour (row-wise; supports mixed timezones)
    TZ = {
        "sakhir": "Asia/Bahrain",
        "jeddah": "Asia/Riyadh",
        "albert_park": "Australia/Melbourne",
        "imola": "Europe/Rome",
        "miami": "America/New_York",
        "monaco": "Europe/Monaco",
        "catalunya": "Europe/Madrid",
        "silverstone": "Europe/London",
        "red_bull_ring": "Europe/Vienna",
        "hungaroring": "Europe/Budapest",
        "spa": "Europe/Brussels",
        "zandvoort": "Europe/Amsterdam",
        "monza": "Europe/Rome",
        "marina_bay": "Asia/Singapore",
        "suzuka": "Asia/Tokyo",
        "lusail": "Asia/Qatar",
        "interlagos": "America/Sao_Paulo",
        "cota": "America/Chicago",
        "hermanos_rodriguez": "America/Mexico_City",
        "baku": "Asia/Baku",
        "paul_ricard": "Europe/Paris",
        "shanghai": "Asia/Shanghai",
        "las_vegas": "America/Los_Angeles",
        "yas_marina": "Asia/Dubai",
        "gilles_villeneuve": "America/Toronto",
    }

    def fmt_local(dt_utc, track):
        if pd.isna(dt_utc) or pd.isna(track):
            return (pd.NA, pd.NA)
        tzname = TZ.get(str(track), "UTC")
        try:
            ts_local = pd.Timestamp(dt_utc).tz_convert(pytz.timezone(tzname))
        except Exception:
            return (pd.NA, pd.NA)
        s = ts_local.strftime("%Y-%m-%dT%H:%M:%S%z")
        s = re.sub(r"(\d{2})(\d{2})$", r"\1:\2", s)  # add colon in offset
        return (s, int(ts_local.hour))

    if "local_time" in df.columns:
        need_local = df["local_time"].isna() | (df["local_time"] == "")
        count = int(need_local.sum())
        if count:
            # compute per row; returns tuple (local_time_str, local_hour_int)
            vals = df.loc[need_local, ["timestamp_utc","circuit_id"]].apply(
                lambda r: fmt_local(r["timestamp_utc"], r["circuit_id"]), axis=1
            )
            # assign back
            df.loc[need_local, "local_time"] = [v[0] for v in vals]
            df.loc[need_local, "local_hour"] = [v[1] for v in vals]
            # cast hour to nullable int
            df["local_hour"] = pd.to_numeric(df["local_hour"], errors="coerce").astype("Int64")
            print(f"[{ts()}] Backfilled {count} local_time/local_hour values.")

    # 4) QA date range BEFORE stringifying datetimes
    ts_min = df["timestamp_utc"].min()
    ts_max = df["timestamp_utc"].max()

    # 5) Normalize timestamp strings for CSV
    fmt = "%Y-%m-%dT%H:%M:%S%z"
    df["timestamp_utc"] = df["timestamp_utc"].dt.strftime(fmt).str.replace(
        r"(\d{2})(\d{2})$", r"\1:\2", regex=True
    )
    if "ingested_at" in df.columns:
        df["ingested_at"] = df["ingested_at"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z").str.replace(
            r"(\d{2})(\d{2})$", r"\1:\2", regex=True
        )

    # 6) Backup & write
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    backup = PATH.replace(".csv", f".backup-{stamp}.csv")
    df.to_csv(backup, index=False)
    df.to_csv(PATH, index=False)
    print(f"[{ts()}] Backup written: {backup}")
    print(f"[{ts()}] Overwrote original: {PATH}")

    # 7) QA summary
    print("\n=== QA SUMMARY ===")
    print("Row count:", len(df))
    if "circuit_id" in df.columns:
        print("Unique circuits:", df['circuit_id'].nunique())
    print("Missing values by column:")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print("Date range (UTC):", ts_min, "→", ts_max)

if __name__ == "__main__":
    main()
