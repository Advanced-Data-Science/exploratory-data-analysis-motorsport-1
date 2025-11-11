import pandas as pd
from pathlib import Path

p = Path(r"C:\Users\Justin\Documents\jdulyanu-tire-deg\data\processed\laps.csv")

# 1) Find suspicious lines (optional but helpful)
bad_lines = []
with p.open("r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f, start=1):
        # 14 columns expected; 27=> likely 2 lines jammed together
        if line.count(",") not in (13,):  # 14 fields -> 13 commas
            bad_lines.append((i, line.strip()[:200]))
print("Bad line candidates:", bad_lines[:10])

# 2) Load while skipping corrupted lines, then re-save cleanly
df = pd.read_csv(p, on_bad_lines="skip")  # pandas>=1.3
clean_path = p.with_name("laps_clean.csv")
df.to_csv(clean_path, index=False)
print("Wrote cleaned file:", clean_path)

# 3) (Optional) replace original AFTER you’re happy
# clean_path.replace(p)  # uncomment once inspected
