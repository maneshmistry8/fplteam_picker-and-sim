import json
import pandas as pd
from pathlib import Path
import requests

RAW_DIR = Path("data/raw")
BOOTSTRAP_FILE = RAW_DIR / "bootstrap-static.json"
GW_FILES = sorted(RAW_DIR.glob("gw*_live.json"))
FIXTURES_FILE = RAW_DIR / "fixtures.json"

# Verify raw data files exist
if not BOOTSTRAP_FILE.exists():
    raise FileNotFoundError(f"Bootstrap file not found: {BOOTSTRAP_FILE}")
if not GW_FILES:
    raise FileNotFoundError("No gameweek files found in data/raw/")
if not FIXTURES_FILE.exists():
    raise FileNotFoundError(f"Fixtures file not found: {FIXTURES_FILE}")

# Load bootstrap metadata
with open(BOOTSTRAP_FILE) as f:
    bootstrap = json.load(f)

players = bootstrap["elements"]
teams = bootstrap["teams"]
positions = bootstrap["element_types"]

# Mapping dictionaries
id_to_name = {p["id"]: f"{p['first_name']} {p['second_name']}" for p in players}
id_to_team = {p["id"]: p["team"] for p in players}
team_id_to_name = {t["id"]: t["name"] for t in teams}
id_to_position = {p["id"]: p["element_type"] for p in players}
position_id_to_name = {p["id"]: p["singular_name_short"] for p in positions}

# Load fixtures for FDR
with open(FIXTURES_FILE) as f:
    fixtures = json.load(f)

# Create FDR mapping: dict of (gw, team_id) -> fdr
fdr_dict = {}
for fixture in fixtures:
    gw = fixture.get("event")
    if gw is None:
        continue
    home_team_id = fixture["team_h"]
    away_team_id = fixture["team_a"]
    home_fdr = fixture["team_h_difficulty"]
    away_fdr = fixture["team_a_difficulty"]
    fdr_dict[(gw, home_team_id)] = home_fdr
    fdr_dict[(gw, away_team_id)] = away_fdr

# Fetch Understat xG/xA/xCS data from vaastav's GitHub repo
understat_dfs = []
base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/"
for gw in range(1, 39):
    url = f"{base_url}gw{gw}.csv"
    try:
        gw_df = pd.read_csv(url, encoding='utf-8')
        gw_df["gameweek"] = gw
        understat_dfs.append(gw_df[["name", "gameweek", "expected_goals", "expected_assists", "expected_goals_conceded"]])
    except Exception as e:
        print(f"Failed to fetch GW {gw} Understat data: {e}")
understat_df = pd.concat(understat_dfs, ignore_index=True) if understat_dfs else pd.DataFrame()
understat_df.rename(columns={
    "expected_goals": "xg",
    "expected_assists": "xa",
    "expected_goals_conceded": "xgc"
}, inplace=True)
understat_df["xg"] = understat_df["xg"].astype(float, errors='ignore')
understat_df["xa"] = understat_df["xa"].astype(float, errors='ignore')
understat_df["xgc"] = understat_df["xgc"].astype(float, errors='ignore')
understat_df["xcs"] = 1 / (1 + understat_df["xgc"].fillna(0))  # Approximate clean sheet probability

# Load all player summaries for price and fixture metadata
player_prices = {}  # (player_id, gameweek) -> price in millions
player_gameweek_info = {}  # (player_id, gameweek) -> fixture context
for summary_file in RAW_DIR.glob("player_*_summary.json"):
    with open(summary_file) as f:
        pdata = json.load(f)
    pid = int(summary_file.stem.split("_")[1])
    for entry in pdata["history"]:
        gw = entry["round"]
        price = entry["value"] / 10
        player_prices[(pid, gw)] = price
        player_gameweek_info[(pid, gw)] = {
            "opponent_team": entry["opponent_team"],
            "was_home": entry["was_home"],
            "team_h_score": entry["team_h_score"],
            "team_a_score": entry["team_a_score"]
        }

# Collect cleaned rows
data_rows = []
for gw_file in GW_FILES:
    gw = int(gw_file.stem.replace("gw", "").replace("_live", ""))
    with open(gw_file) as f:
        gw_data = json.load(f)

    for player in gw_data["elements"]:
        pid = player["id"]
        stats = player["stats"]

        info = player_gameweek_info.get((pid, gw), {})
        opponent_team_id = info.get("opponent_team", None)
        was_home = info.get("was_home", True)
        team_h_score = info.get("team_h_score", 0)
        team_a_score = info.get("team_a_score", 0)
        result = "W" if team_h_score > team_a_score else "D" if team_h_score == team_a_score else "L"

        # Add FDR for the player's team
        team_id = id_to_team.get(pid, 0)
        fdr = fdr_dict.get((gw, team_id), None)

        data_rows.append({
            "gameweek": gw,
            "player_id": pid,
            "name": id_to_name.get(pid, "Unknown"),
            "team": team_id_to_name.get(id_to_team.get(pid, 0), "Unknown"),
            "team_id": id_to_team.get(pid, 0),
            "position": position_id_to_name.get(id_to_position.get(pid, 0), "Unknown"),
            "opponent": team_id_to_name.get(opponent_team_id, "Unknown"),
            "opponent_team_id": opponent_team_id,
            "was_home": was_home,
            "result": result,
            "team_h_score": team_h_score,
            "team_a_score": team_a_score,
            "minutes": stats["minutes"],
            "goals_scored": stats["goals_scored"],
            "assists": stats["assists"],
            "clean_sheets": stats["clean_sheets"],
            "goals_conceded": stats["goals_conceded"],
            "own_goals": stats["own_goals"],
            "penalties_saved": stats["penalties_saved"],
            "penalties_missed": stats["penalties_missed"],
            "yellow_cards": stats["yellow_cards"],
            "red_cards": stats["red_cards"],
            "saves": stats["saves"],
            "bonus": stats["bonus"],
            "bps": stats["bps"],
            "influence": float(stats["influence"]),
            "creativity": float(stats["creativity"]),
            "threat": float(stats["threat"]),
            "ict_index": float(stats["ict_index"]),
            "total_points": stats["total_points"],
            "value": player_prices.get((pid, gw), None),
            "fdr": fdr,
        })

# Create DataFrame
df = pd.DataFrame(data_rows)

# Verify required columns
required_columns = ["gameweek", "player_id", "team_id", "opponent_team_id", "team_h_score", "team_a_score", "was_home", "total_points"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

# Merge xG, xA, and xCS from Understat
df = df.merge(
    understat_df,
    on=["name", "gameweek"],
    how="left"
)
df["xg"] = df["xg"].fillna(0)
df["xa"] = df["xa"].fillna(0)
df["xcs"] = df["xcs"].fillna(0)

# Save cleaned dataset
out_path = Path("data/processed/fpl_2023_24_cleaned.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Saved cleaned dataset to {out_path}")
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
