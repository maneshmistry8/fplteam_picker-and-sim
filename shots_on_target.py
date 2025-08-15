import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/2024_25")
merged_gw = pd.read_csv(RAW_DIR / "gws_merged_gw.csv", encoding='utf-8', quoting=csv.QUOTE_ALL)

# Get unique player IDs
player_ids = merged_gw["element"].unique()

# Fetch shots on target from FPL API
shots_data = []
for player_id in player_ids:
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        for gw in data["history"]:
            shots_data.append({
                "element": gw["element"],
                "round": gw["round"],
                "shots_on_target": gw.get("shots_on_target", 0)
            })
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data for player {player_id}: {e}")

# Convert to DataFrame and merge
shots_df = pd.DataFrame(shots_data)
merged_gw = merged_gw.merge(
    shots_df[["element", "round", "shots_on_target"]],
    left_on=["element", "GW"],
    right_on=["element", "round"],
    how="left"
)
merged_gw["shots_on_target"] = merged_gw["shots_on_target"].fillna(0)

# Save updated merged_gw
merged_gw.to_csv(RAW_DIR / "gws_merged_gw.csv", index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
print("Updated gws_merged_gw.csv with shots_on_target")
