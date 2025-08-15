import json
from pathlib import Path

sample = Path("data/raw/gw1_live.json")  # Change if needed
with open(sample) as f:
    gw_data = json.load(f)

# Check top-level keys
print("Top-level keys:", gw_data.keys())

# Check what each player entry looks like
print("Player keys:", gw_data["elements"][0].keys())
print("Stats keys:", gw_data["elements"][0]["stats"].keys())
