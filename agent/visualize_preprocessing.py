import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG: set your filters here
# -----------------------------
YEARS = {2022, 2023}                 # your dataset years
SESSIONS = {"FP1","FP2", "FP3"}            # choose {"FP1","FP2","FP3"} or a subset
VALID_WINDOW = (30, 200)             # seconds
SHOW_OVERLAY_BY_SESSION = False      # set True to compare FP1/FP2/FP3

# -----------------------------
# Load & normalize
# -----------------------------
df = pd.read_csv("data/processed/laps.csv")

# Ensure numeric and clean types
df["lap_time_s"] = pd.to_numeric(df["lap_time_s"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

# Normalize session labels to a canonical set {FP1, FP2, FP3}
# (handles cases like 'Practice 1', 'P1', etc.)
session_map = {
    "practice 1": "FP1", "p1": "FP1", "fp1": "FP1",
    "practice 2": "FP2", "p2": "FP2", "fp2": "FP2",
    "practice 3": "FP3", "p3": "FP3", "fp3": "FP3",
}
df["session_norm"] = (
    df["session"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(session_map)
    .fillna(df["session"].astype(str).str.upper())  # fallback if already FP1/FP2/FP3
)

# Apply filters
df = df[df["year"].isin(YEARS)]
df = df[df["session_norm"].isin(SESSIONS)]

# Valid-only filter (if you truly have all valid laps, this will just keep everything)
lo, hi = VALID_WINDOW
valid = df["lap_time_s"].between(lo, hi, inclusive="both")
valid_laps = df.loc[valid, "lap_time_s"].dropna()

if valid_laps.empty:
    raise SystemExit("No valid laps found after filtering. Check YEARS/SESSIONS or VALID_WINDOW.")

# -----------------------------
# Build bins with 1-second resolution
# -----------------------------
min_time = int(valid_laps.min())
max_time = int(valid_laps.max())
# pad a bit for edges
bins = range(min_time - 1, max_time + 2, 1)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 6))

if SHOW_OVERLAY_BY_SESSION:
    # Overlay FP1/FP2/FP3 as outlines for comparison
    for ses in sorted(df["session_norm"].unique()):
        s = df.loc[valid & (df["session_norm"] == ses), "lap_time_s"]
        plt.hist(s, bins=bins, histtype="step", linewidth=1.8, label=ses)
    plt.legend(title="Session", frameon=True)
else:
    # Single series: valid laps only
    plt.hist(valid_laps, bins=bins, alpha=0.75, edgecolor="black", label="Valid Laps")
    plt.legend(loc="upper right", frameon=True)

title_years = " & ".join(str(y) for y in sorted(YEARS))
title_sessions = ", ".join(sorted(SESSIONS))
plt.title(f"Lap Time Distribution — Valid Laps Only ({title_sessions}, {title_years})", fontsize=14, weight="bold")
plt.xlabel("Lap Time (seconds)", fontsize=12)
plt.ylabel("Number of Laps", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.show()