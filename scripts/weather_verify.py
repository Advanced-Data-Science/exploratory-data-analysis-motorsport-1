import pandas as pd

wx = pd.read_csv("data/processed/weather_track_conditions.csv")

# Basic shape
print("Rows:", len(wx))
print("Columns:", list(wx.columns))

# Years present
wx["year"] = pd.to_datetime(wx["timestamp_utc"], utc=True, errors="coerce").dt.year
print("Counts by year:\n", wx["year"].value_counts().sort_index())

# How many have circuit_id missing from the old schema block?
missing_cid = wx["circuit_id"].isna() | (wx["circuit_id"].astype(str).str.len() == 0)
print("Rows missing circuit_id:", missing_cid.sum())
print(wx.loc[missing_cid].head(3))
