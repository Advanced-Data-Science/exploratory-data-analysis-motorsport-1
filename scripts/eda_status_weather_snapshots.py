# scripts/eda_status_weather_snapshots.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN  = Path("data/processed/laps_weather_joined_clean.csv")
OUT = Path("reports/eda"); OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN, parse_dates=["timestamp_utc_lap"])

# --- 1) Laps by track status (count + share) ---
status_counts = (df.groupby("track_status_clean")
                   .size().rename("count")
                   .reset_index()
                   .sort_values("count", ascending=False))
status_counts["share_pct"] = 100 * status_counts["count"] / len(df)
status_counts.to_csv(OUT / "status_01_counts.csv", index=False)

plt.figure()
status_counts.set_index("track_status_clean")["count"].plot(kind="bar")
plt.title("Lap counts by track_status_clean")
plt.ylabel("Laps")
plt.tight_layout()
plt.savefig(OUT / "status_01_counts.png", dpi=150)
plt.close()

# Per-session share (useful context)
sess_status = (df.groupby(["year","gp_name","session","track_status_clean"])
                 .size().rename("count").reset_index())
sess_totals = (df.groupby(["year","gp_name","session"]).size()
                 .rename("tot").reset_index())
sess_status = sess_status.merge(sess_totals, on=["year","gp_name","session"], how="left")
sess_status["share_pct"] = 100 * sess_status["count"] / sess_status["tot"]
sess_status.to_csv(OUT / "status_02_by_session.csv", index=False)

# --- 2) Lap time by status (overall + within session/compound) ---
def agg_stats(g):
    return pd.Series({
        "n": g.size,
        "median": g.median(),
        "mean": g.mean(),
        "std": g.std(),
        "p10": g.quantile(0.10),
        "p90": g.quantile(0.90)
    })

pace_by_status = df.groupby("track_status_clean")["lap_time_s"].apply(agg_stats).reset_index()
pace_by_status.to_csv(OUT / "status_03_pace_overall.csv", index=False)

pace_sess_comp = (df.groupby(["session","compound_code","track_status_clean"])["lap_time_s"]
                    .apply(agg_stats).reset_index())
pace_sess_comp.to_csv(OUT / "status_04_pace_by_session_compound.csv", index=False)

# Quick boxplot (green vs not-green) for a visual sanity check
df["is_green"] = (df["track_status_clean"] == "Track clear")
plt.figure()
df.boxplot(column="lap_time_s", by="is_green")
plt.title("Lap time distribution: Track clear vs non-clear")
plt.suptitle("")
plt.xlabel("Is green status?")
plt.ylabel("Lap time (s)")
plt.tight_layout()
plt.savefig(OUT / "status_05_box_green_vs_non.png", dpi=150)
plt.close()

# --- 3) Weather main distribution & pace ---
wx_counts = (df["weather_main"].fillna("Unknown").value_counts()
               .rename_axis("weather_main").reset_index(name="count"))
wx_counts.to_csv(OUT / "weather_01_counts.csv", index=False)

wx_pace = (df.groupby("weather_main")["lap_time_s"]
             .apply(agg_stats).reset_index())
wx_pace.to_csv(OUT / "weather_02_pace_by_condition.csv", index=False)

# --- 4) PB under non-green conditions (sanity check) ---
pb_weird = df[(df["is_pb"] == True) & (df["track_status_clean"] != "Track clear")]
pb_weird.to_csv(OUT / "status_06_pb_non_green.csv", index=False)

print("EDA snapshots written to", OUT.resolve())
