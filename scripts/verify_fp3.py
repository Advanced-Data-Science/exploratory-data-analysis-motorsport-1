import fastf1, pandas as pd

# load your existing collected sessions
laps = pd.read_csv("C:/Users/Justin/Documents/jdulyanu-tire-deg/data/processed/laps.csv")
laps["session"] = laps["session"].str.upper().str.replace("PRACTICE ", "FP")
collected = set(zip(laps["year"], laps["gp_name"], laps["session"]))

wanted = {"FP1", "FP2", "FP3"}
missing = []

for year in [2022, 2023, 2024]:
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"⚠️ Failed to load schedule for {year}: {e}")
        continue

    session_cols = [c for c in schedule.columns if str(c).lower().startswith("session")]
    for _, row in schedule.iterrows():
        gp = str(row["EventName"])
        for c in session_cols:
            label = str(row[c]).lower()
            if "practice 1" in label:
                code = "FP1"
            elif "practice 2" in label:
                code = "FP2"
            elif "practice 3" in label:
                code = "FP3"
            else:
                continue

            if (year, gp, code) not in collected:
                missing.append((year, gp, code))

print(f"\nTotal missing FP sessions: {len(missing)}")
for y, e, s in missing:
    print(f"{y}: {e} - {s}")

df = pd.read_csv("data/processed/laps.csv")
print(df.query("year == 2022")["session"].value_counts())
