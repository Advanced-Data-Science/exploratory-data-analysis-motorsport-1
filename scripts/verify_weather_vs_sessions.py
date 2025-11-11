# verify_weather_vs_sessions.py
# Audit whether weather rows align with FP session windows derived from laps.
# Outputs a CSV report with per-session alignment checks.

from pathlib import Path
from typing import Tuple, Iterable
import pandas as pd

# ========= CONFIG (edit if paths differ) =========
LAPS_CSV    = "data/processed/laps.csv"
WEATHER_CSV = "data/processed/weather_track_conditions.csv"
REPORT_CSV  = "data/processed/_audit_weather_session_alignment.csv"

ASOF_TOL = pd.Timedelta("45min")  # tolerance for "nearest" weather rows to session start/end

# EXACT mapping: laps['gp_name'] -> weather['circuit_id']
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
    """
    Map various practice labels to {'FP1','FP2','FP3'}.
    Accepts e.g. 'FP-1', 'Free Practice 1', 'Practice 1', 'P1' -> 'FP1', etc.
    Non-FP returns original (so they’ll be filtered out later).
    """
    if not isinstance(s, str):
        return s
    t = s.strip().lower().replace("-", " ").replace("_", " ")
    t = " ".join(t.split())  # squeeze spaces
    # common patterns
    if t in {"fp1", "p1", "practice 1", "free practice 1"}:
        return "FP1"
    if t in {"fp2", "p2", "practice 2", "free practice 2"}:
        return "FP2"
    if t in {"fp3", "p3", "practice 3", "free practice 3"}:
        return "FP3"
    return s  # leave as-is for non-FP sessions

def nearest_delta(ts: pd.Timestamp, series: pd.Series) -> Tuple[pd.Timestamp, pd.Timedelta]:
    """Return (nearest_timestamp_in_series, absolute_delta). Both may be NaT if series empty or ts NaT."""
    if series.empty or pd.isna(ts):
        return pd.NaT, pd.NaT
    # series must be sorted for searchsorted to be meaningful
    series = series.sort_values(ignore_index=True)
    i = series.searchsorted(ts)
    candidates: Iterable[pd.Timestamp] = []
    if i > 0:
        candidates.append(series.iloc[i - 1])
    if i < len(series):
        candidates.append(series.iloc[i])
    if not candidates:
        return pd.NaT, pd.NaT
    deltas = [abs(c - ts) for c in candidates]
    j = min(range(len(deltas)), key=lambda k: deltas[k])
    best = candidates[j]
    return best, abs(best - ts)

# ========= MAIN =========
def main():
    # --- Load ---
    laps = pd.read_csv(LAPS_CSV)
    weather = pd.read_csv(WEATHER_CSV)

    # --- Parse timestamps (tz-aware UTC) ---
    laps["timestamp_utc"] = pd.to_datetime(laps["timestamp_utc"], utc=True, errors="coerce")
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True, errors="coerce")

    # --- Normalize session labels & filter to FP only ---
    if "session" not in laps.columns:
        raise ValueError("laps.csv missing 'session' column.")
    laps["session_norm"] = laps["session"].map(normalize_session_label)
    laps = laps[laps["session_norm"].isin({"FP1", "FP2", "FP3"})].copy()

    # --- Map circuits and drop unusable rows ---
    if "gp_name" not in laps.columns:
        raise ValueError("laps.csv missing 'gp_name' column.")
    laps["circuit_id"] = laps["gp_name"].map(GP_TO_CIRCUIT)
    # Diagnostics for unmapped GP names
    unmapped = sorted(laps.loc[laps["circuit_id"].isna(), "gp_name"].unique().tolist())
    if unmapped:
        print("⚠️ Unmapped gp_name values (add to GP_TO_CIRCUIT):", unmapped)
    laps = laps.dropna(subset=["timestamp_utc", "circuit_id"]).copy()

    # --- Fast guard for empty set after filtering ---
    if laps.empty:
        print("⚠️ No laps remaining after FP filter and mapping. Check session labels and GP_TO_CIRCUIT mapping.")
        print("   Unique raw session labels:", pd.Series(laps["session"].unique() if "session" in laps else []).tolist())
        return

    # --- Session windows (per event) ---
    # Keep the original year and name fields if present
    if "year" not in laps.columns:
        laps["year"] = pd.NaT  # keep column for sorting/reporting

    laps["event_key"] = (
        laps.get("year").astype(str).fillna("")
        + "|" + laps["gp_name"].astype(str)
        + "|" + laps["session_norm"].astype(str)
    )

    sess = (
        laps.groupby(["event_key", "year", "gp_name", "session_norm", "circuit_id"])["timestamp_utc"]
            .agg(session_start="min", session_end="max")
            .reset_index()
    )

    # --- Weather per circuit (sorted) ---
    weather = weather.dropna(subset=["timestamp_utc", "circuit_id"]).copy()
    weather = weather.sort_values(["circuit_id", "timestamp_utc"]).reset_index(drop=True)

    # --- Build audit rows ---
    rows = []
    for _, r in sess.iterrows():
        cid = r["circuit_id"]
        wx_ts = weather.loc[weather["circuit_id"] == cid, "timestamp_utc"]
        start_near, start_delta = nearest_delta(r["session_start"], wx_ts)
        end_near, end_delta = nearest_delta(r["session_end"], wx_ts)

        # Compute robust overlap-with-tolerance
        if wx_ts.empty:
            overlap = False
            wx_min = wx_max = pd.NaT
        else:
            wx_min, wx_max = wx_ts.iloc[0], wx_ts.iloc[-1]

        # Treat instantaneous sessions as valid if *any* weather row falls within tolerance
        # around the session window (expanded by ASOF_TOL).
        window_start = r["session_start"] - ASOF_TOL
        window_end   = r["session_end"]   + ASOF_TOL

        if wx_ts.empty:
            overlap = False
        else:
            has_any_wx_in_window = not wx_ts[(wx_ts >= window_start) & (wx_ts <= window_end)].empty
            overlap = has_any_wx_in_window

        ok_start = (pd.notna(start_delta) and start_delta <= ASOF_TOL)
        ok_end   = (pd.notna(end_delta)   and end_delta   <= ASOF_TOL)

        rows.append({
            "event_key": r["event_key"],
            "year": None if pd.isna(r["year"]) else int(r["year"]),
            "gp_name": r["gp_name"],
            "session": r["session_norm"],
            "circuit_id": cid,
            "session_start_utc": r["session_start"],
            "session_end_utc": r["session_end"],
            "wx_min_utc": wx_min,
            "wx_max_utc": wx_max,
            "nearest_wx_at_start_utc": start_near,
            "delta_start_sec": None if pd.isna(start_delta) else int(start_delta.total_seconds()),
            "nearest_wx_at_end_utc": end_near,
            "delta_end_sec": None if pd.isna(end_delta) else int(end_delta.total_seconds()),
            "has_overlap": bool(overlap),
            "ok_start<=tol": bool(ok_start),
            "ok_end<=tol": bool(ok_end),
            # pass if start & end are within tolerance AND there is at least one wx point in window
            "passes_all_checks": bool(overlap and ok_start and ok_end),
        })

    # --- Assemble report DataFrame (robust to empties) ---
    audit_cols = [
        "event_key","year","gp_name","session","circuit_id",
        "session_start_utc","session_end_utc",
        "wx_min_utc","wx_max_utc",
        "nearest_wx_at_start_utc","delta_start_sec",
        "nearest_wx_at_end_utc","delta_end_sec",
        "has_overlap","ok_start<=tol","ok_end<=tol","passes_all_checks",
    ]
    audit = pd.DataFrame(rows)
    for c in audit_cols:
        if c not in audit.columns:
            audit[c] = pd.Series(dtype="object")

    sort_cols = [c for c in ["year", "gp_name", "session"] if c in audit.columns]
    if sort_cols:
        # Handle year with None gracefully
        if "year" in sort_cols:
            audit["year"] = pd.to_numeric(audit["year"], errors="coerce")
        audit = audit.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    # --- Save CSV report ---
    Path(REPORT_CSV).parent.mkdir(parents=True, exist_ok=True)
    audit[audit_cols].to_csv(REPORT_CSV, index=False)

    # --- Console summary ---
    n_rows = len(audit)
    n_ok = int(audit.get("passes_all_checks", pd.Series([], dtype=bool)).fillna(False).sum()) if n_rows else 0
    n_bad = n_rows - n_ok
    print(f"Sessions audited: {n_rows}  |  OK: {n_ok}  |  Issues: {n_bad}")
    print(f"Report: {REPORT_CSV}")

    if n_bad > 0:
        print("\n⚠️ Sessions failing checks (showing up to 10):")
        cols = [
            "event_key","circuit_id",
            "session_start_utc","nearest_wx_at_start_utc","delta_start_sec",
            "session_end_utc","nearest_wx_at_end_utc","delta_end_sec",
            "wx_min_utc","wx_max_utc","has_overlap"
        ]
        print(audit.loc[~audit["passes_all_checks"], cols].head(10).to_string(index=False))

    # Helpful diagnostics if nothing got audited
    if n_rows == 0:
        print("\nDiagnostics:")
        if "session" in laps.columns:
            print("• Unique raw session labels in laps:", sorted(set(laps["session"])))
        if "session_norm" in laps.columns:
            print("• Normalized FP labels found:", sorted(set(laps["session_norm"])))
        missing_map = sorted(set(laps["gp_name"]) - set(GP_TO_CIRCUIT.keys()))
        print("• GP names missing in GP_TO_CIRCUIT:", missing_map)
        print("• Any NaT in laps timestamp_utc:", laps["timestamp_utc"].isna().any())
        print("• Any NaT in weather timestamp_utc:", weather["timestamp_utc"].isna().any())

if __name__ == "__main__":
    main()
