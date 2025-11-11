import fastf1
import pandas as pd

sched = fastf1.events.get_event_schedule(2024, include_testing=False)
pd.set_option('display.max_columns', None)

print("\n=== First 3 rows (transposed) ===")
print(sched.head(3).T)

print("\n=== Columns containing 'Session' ===")
print([c for c in sched.columns if 'Session' in c])
