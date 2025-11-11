import pandas as pd
from pathlib import Path

# === Paths ===
base_dir = Path(r"C:\Users\Justin\Documents\jdulyanu-tire-deg\data\processed")
input_path = base_dir / "weather_conditions_openmeteo.csv"
output_path = base_dir / "weather_conditions_openmeteo_v2.csv"

# === Load weather data ===
wx = pd.read_csv(input_path)

# === Drop unneeded columns ===
cols_to_drop = ["weather_main", "weather_desc", "source", "ingested_at"]
wx = wx.drop(columns=[c for c in cols_to_drop if c in wx.columns])

# === Optional: reorder columns for readability ===
col_order = [
    "circuit_id", "timestamp_utc", "local_time", "local_hour",
    "temp_c", "humidity_pct", "precip_mm", "wind_speed_ms"
]
wx = wx[[c for c in col_order if c in wx.columns]]

# === Save the cleaned file ===
wx.to_csv(output_path, index=False)
print(f"✅ Cleaned Open-Meteo weather file saved to:\n{output_path}")
print(f"Rows: {len(wx)}, Columns: {len(wx.columns)}")