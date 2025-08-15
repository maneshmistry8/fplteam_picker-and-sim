import requests
import pandas as pd
from pathlib import Path
import csv

DATA_DIR = Path("data/raw/2024_25")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/"

# Static files
static_files = [
    "players_raw.csv",
    "teams.csv",
    "fixtures.csv"
]

# Download static files
for file in static_files:
    url = BASE_URL + file
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        filepath = DATA_DIR / file.replace("/", "_")
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(resp.text)
        print(f"Saved {file} to {filepath}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file}: {e}")

# Expected columns for GW files (based on 2023-24 structure, plus shots_on_target)
expected_columns = [
    "name", "position", "team", "xP", "assists", "bonus", "bps", "clean_sheets",
    "creativity", "element", "fixture", "goals_conceded", "goals_scored", "ict_index",
    "influence", "minutes", "opponent_team", "own_goals", "penalties_missed",
    "penalties_saved", "red_cards", "round", "saves", "selected", "team_a_score",
    "team_h_score", "threat", "total_points", "transfers_balance", "transfers_in",
    "transfers_out", "value", "was_home", "yellow_cards", "shots_on_target"
]

# Download and combine GW files
gw_dfs = []
for gw in range(1, 39):
    file = f"gws/gw{gw}.csv"
    url = BASE_URL + file
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        filepath = DATA_DIR / f"gw{gw}.csv"
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(resp.text)
        # Read with robust parsing
        df = pd.read_csv(filepath, encoding='utf-8', quoting=csv.QUOTE_ALL)
        # Ensure consistent columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        for col in missing_cols:
            df[col] = 0
        df = df[expected_columns]
        df["GW"] = gw
        gw_dfs.append(df)
        print(f"Saved {file} to {filepath}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file}: {e}")
    except pd.errors.ParserError as e:
        print(f"Parsing error in {file}: {e}")

# Combine GW files into merged_gw.csv
if gw_dfs:
    merged_gw = pd.concat(gw_dfs, ignore_index=True)
    merged_filepath = DATA_DIR / "gws_merged_gw.csv"
    merged_gw.to_csv(merged_filepath, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    print(f"Saved combined GW data to {merged_filepath}")
else:
    print("No GW data downloaded, skipping gws_merged_gw.csv")
