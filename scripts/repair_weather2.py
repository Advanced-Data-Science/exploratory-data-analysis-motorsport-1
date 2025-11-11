# repair_weather2.py
import os, sys, time
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

    # 1) Load + normalize blanks to NaN
    df = pd.read_csv(PATH, dtype=str)  # keep raw strings to detect blanks
    n_rows = len(df)

    # Normalize whitespace and empty strings
    df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)
    df.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA}, inplace=True)

    # Coerce types where useful
    for col in ["local_hour", "temp_c", "humidity_pct", "precip_mm", "wind_speed_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse timestamps
    for col in ["timestamp_utc", "ingested_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # 2) Baseline stats
    blanks_circuit = df["circuit_id"].isna().sum() if "circuit_id" in df.columns else None
    dup_key = df.duplicated(subset=["circuit_id","timestamp_utc"], keep=False).sum() if {"circuit_id","timestamp_utc"} <= set(df.columns) else None

    print(f"[{ts()}] Rows: {n_rows}")
    if blanks_circuit is not None:
        print(f"[{ts()}] Blank circuit_id rows: {blanks_circuit}")
    if dup_key is not None:
        print(f"[{ts()}] Rows participating in duplicate (circuit_id,timestamp_utc) keys: {dup_key}")

    # 3) Fix pattern in your sample:
    # There are pairs at the SAME timestamp_utc:
    #  - one row has circuit_id = NA, and some weather values X
    #  - another row has circuit_id = <track>, with slightly different weather + later ingested_at
    # Goal: collapse to a single canonical row per (track, timestamp_utc), preferring the row:
    #    (a) with non-null circuit_id,
    #    (b) with the latest ingested_at,
    #    (c) with max non-null fields (tie-breaker).
    if not {"timestamp_utc", "ingested_at"}.issubset(df.columns):
        print(f"[{ts()}] WARNING: Missing required columns for repair. No changes applied.")
    else:
        # First, for any timestamp that has at least one non-null circuit_id, propagate that id
        # to the null-id sibling(s) with the exact same timestamp_utc.
        if "circuit_id" in df.columns:
            id_fill = df.groupby("timestamp_utc")["circuit_id"].transform(
                lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA
            )
            newly_filled = df["circuit_id"].isna() & id_fill.notna()
            filled_count = newly_filled.sum()
            df.loc[newly_filled, "circuit_id"] = id_fill[newly_filled]
            if filled_count:
                print(f"[{ts()}] Filled circuit_id for {filled_count} rows using same-timestamp propagation.")

        # Now that circuit_id is filled where possible, collapse duplicates by (circuit_id, timestamp_utc)
        key_cols = ["circuit_id", "timestamp_utc"]
        if df["circuit_id"].notna().any():
            # Score rows: prefer newer ingested_at, then more non-nulls
            nonnull_count = df.notna().sum(axis=1).rename("nonnull_count")
            scored = pd.concat([df, nonnull_count], axis=1)

            # Sort so the "best" record comes first per group
            scored = scored.sort_values(
                by=["circuit_id", "timestamp_utc", "ingested_at", "nonnull_count"],
                ascending=[True, True, False, False]
            )

            # Keep first per group
            before = len(scored)
            scored = scored.drop_duplicates(subset=key_cols, keep="first")
            after = len(scored)

            removed = before - after
            if removed > 0:
                print(f"[{ts()}] Deduplicated {removed} rows across (circuit_id, timestamp_utc).")
            df = scored.drop(columns=["nonnull_count"], errors="ignore")

    # 4) After stats
    after_blanks = df["circuit_id"].isna().sum() if "circuit_id" in df.columns else None
    print(f"[{ts()}] After: rows {len(df)} (was {n_rows})")
    if after_blanks is not None:
        print(f"[{ts()}] After: blank circuit_id rows: {after_blanks}")

    # 5) Backup then write
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    backup = PATH.replace(".csv", f".backup-{stamp}.csv")
    df.to_csv(backup, index=False)
    print(f"[{ts()}] Wrote backup: {backup}")

    df.to_csv(PATH, index=False)
    print(f"[{ts()}] Overwrote original: {PATH}")
    print(f"[{ts()}] Done.")

if __name__ == "__main__":
    main()
