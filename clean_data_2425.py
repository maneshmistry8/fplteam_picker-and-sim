import pandas as pd
from pathlib import Path
import csv

RAW_DIR = Path("data/raw/2024_25")
OUTPUT_FILE = Path("data/processed/fpl_2024_25_cleaned.csv")

# Load data
try:
    players_raw = pd.read_csv(RAW_DIR / "players_raw.csv", encoding='utf-8', quoting=csv.QUOTE_ALL)
    teams = pd.read_csv(RAW_DIR / "teams.csv", encoding='utf-8', quoting=csv.QUOTE_ALL)
    fixtures = pd.read_csv(RAW_DIR / "fixtures.csv", encoding='utf-8', quoting=csv.QUOTE_ALL)
    merged_gw = pd.read_csv(RAW_DIR / "gws_merged_gw.csv", encoding='utf-8', quoting=csv.QUOTE_ALL)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing file: {e}")

# Mapping dictionaries
id_to_name = players_raw.set_index("id")["web_name"].to_dict()
id_to_team = players_raw.set_index("id")["team"].to_dict()
team_id_to_name = teams.set_index("id")["name"].to_dict()
id_to_position = players_raw.set_index("id")["element_type"].to_dict()
position_id_to_name = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# FDR from fixtures
fdr_dict = {}
for _, fixture in fixtures.iterrows():
    gw = fixture["event"]
    if pd.isna(gw):
        continue
    gw = int(gw)
    home_team_id = fixture["team_h"]
    away_team_id = fixture["team_a"]
    fdr_dict[(gw, home_team_id)] = fixture["team_h_difficulty"]
    fdr_dict[(gw, away_team_id)] = fixture["team_a_difficulty"]

# Process merged_gw
merged_gw["team_id"] = merged_gw["team"].map({v: k for k, v in team_id_to_name.items()})
merged_gw["opponent_team_id"] = merged_gw["opponent_team"].map({v: k for k, v in team_id_to_name.items()})
merged_gw["position"] = merged_gw["element"].map(id_to_position).map(position_id_to_name)
merged_gw["name"] = merged_gw["element"].map(id_to_name)
merged_gw["fdr"] = merged_gw.apply(lambda row: fdr_dict.get((row["GW"], row["team_id"]), 3.0), axis=1)
merged_gw["value"] = merged_gw["value"] / 10  # Convert to millions
merged_gw["was_home"] = merged_gw["was_home"].astype(bool)
merged_gw["result"] = merged_gw.apply(
    lambda row: "W" if row["team_h_score"] > row["team_a_score"] else "D" if row["team_h_score"] == row["team_a_score"] else "L", axis=1
)
merged_gw["player_id"] = merged_gw["element"]
merged_gw["gameweek"] = merged_gw["GW"]
merged_gw["team"] = merged_gw["team_id"].map(team_id_to_name)
merged_gw["opponent"] = merged_gw["opponent_team_id"].map(team_id_to_name)

# Handle missing xG, xA, xCS, xGC, shots_on_target
merged_gw["xg"] = merged_gw.get("xG", 0.0)
merged_gw["xa"] = merged_gw.get("xA", 0.0)
merged_gw["xcs"] = merged_gw.get("xCS", 0.0)
merged_gw["xgc"] = merged_gw.get("xGC", 0.0)
merged_gw["shots_on_target"] = merged_gw.get("shots_on_target", 0)

# Select columns
cleaned_df = merged_gw[[
    "gameweek", "player_id", "name", "team", "team_id", "position", "opponent", "opponent_team_id",
    "was_home", "result", "team_h_score", "team_a_score", "minutes", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps", "influence", "creativity", "threat",
    "ict_index", "total_points", "value", "fdr", "xg", "xa", "xgc", "xcs", "shots_on_target"
]]

# Handle missing values
cleaned_df.fillna(
    {
        "fdr": 3.0, "xg": 0.0, "xa": 0.0, "xgc": 0.0, "xcs": 0.0, "shots_on_target": 0,
        "influence": 0.0, "creativity": 0.0, "threat": 0.0, "ict_index": 0.0,
        "minutes": 0, "goals_scored": 0, "assists": 0, "clean_sheets": 0,
        "goals_conceded": 0, "own_goals": 0, "penalties_saved": 0, "penalties_missed": 0,
        "yellow_cards": 0, "red_cards": 0, "saves": 0, "bonus": 0, "bps": 0,
        "total_points": 0
    },
    inplace=True
)

# Save cleaned dataset
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
cleaned_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved cleaned dataset to {OUTPUT_FILE}")
