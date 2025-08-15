import pandas as pd
import json
from pathlib import Path

INPUT_FILE = Path("data/processed/fpl_2023_24_cleaned.csv")
OUTPUT_FILE = Path("data/processed/fpl_2023_24_features.csv")
FIXTURES_FILE = Path("data/raw/fixtures.json")

# Load cleaned data
df = pd.read_csv(INPUT_FILE)
df.sort_values(by=["player_id", "gameweek"], inplace=True)

# Verify required input columns
required_columns = [
    "gameweek", "player_id", "team_id", "opponent_team_id", "team_h_score",
    "team_a_score", "was_home", "total_points", "value", "fdr", "xg", "xa", "xcs",
    "minutes", "goals_scored", "assists", "clean_sheets", "yellow_cards", "bonus",
    "bps", "influence", "creativity", "threat", "ict_index"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing input columns: {missing_columns}")

# Load fixtures for DGW
with open(FIXTURES_FILE) as f:
    fixtures = json.load(f)
# Count fixtures per team per GW
dgw_dict = {}
for fixture in fixtures:
    gw = fixture.get("event")
    if gw is None:
        continue
    home_team = fixture["team_h"]
    away_team = fixture["team_a"]
    dgw_dict[(gw, home_team)] = dgw_dict.get((gw, home_team), 0) + 1
    dgw_dict[(gw, away_team)] = dgw_dict.get((gw, away_team), 0) + 1
# Add DGW flag
df["is_dgw"] = df.apply(
    lambda row: 1 if dgw_dict.get((row["gameweek"], row["team_id"]), 0) > 1 else 0,
    axis=1
)

# Compute rolling features per player
rolling_features = [
    "minutes", "goals_scored", "assists", "clean_sheets",
    "yellow_cards", "bonus", "bps", "influence", "creativity",
    "threat", "ict_index", "total_points", "xg", "xa", "xcs"
]
for feature in rolling_features:
    df[f"{feature}_roll3"] = (
        df.groupby("player_id")[feature]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

# Add 3-week rolling std dev of points
df["total_points_std3"] = (
    df.groupby("player_id")["total_points"]
    .rolling(window=3, min_periods=1)
    .std()
    .reset_index(level=0, drop=True)
)

# Add 3-week rolling mean for FDR
df["fdr_roll3"] = (
    df.groupby("player_id")["fdr"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Add price delta
df["value_delta"] = df.groupby("player_id")["value"].diff()

# Add team form (3-GW rolling avg points for player's team)
team_points = df.groupby(["team_id", "gameweek"])["total_points"].sum().reset_index()
team_points["team_form_roll3"] = (
    team_points.groupby("team_id")["total_points"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
df = df.merge(
    team_points[["team_id", "gameweek", "team_form_roll3"]],
    on=["team_id", "gameweek"],
    how="left"
)
df["team_form_roll3"] = df["team_form_roll3"].fillna(0)

# Add opponent team form (3-GW rolling avg points)
opp_team_points = df.groupby(["opponent_team_id", "gameweek"])["total_points"].mean().reset_index()
opp_team_points["opp_team_form_roll3"] = (
    opp_team_points.groupby("opponent_team_id")["total_points"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
df = df.merge(
    opp_team_points[["opponent_team_id", "gameweek", "opp_team_form_roll3"]],
    on=["opponent_team_id", "gameweek"],
    how="left"
)
df["opp_team_form_roll3"] = df["opp_team_form_roll3"].fillna(0)

# Add player’s share of team points
team_points = df.groupby(["team_id", "gameweek"])["total_points"].sum().reset_index(name="team_total_points")
df = df.merge(team_points, on=["team_id", "gameweek"], how="left")
df["player_team_points_share"] = df["total_points"] / df["team_total_points"].replace(0, 1)

# Add home/away points differential
home_points = df[df["was_home"]][["player_id", "gameweek", "total_points"]]
home_points["home_points_roll3"] = (
    home_points.groupby("player_id")["total_points"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
away_points = df[~df["was_home"]][["player_id", "gameweek", "total_points"]]
away_points["away_points_roll3"] = (
    away_points.groupby("player_id")["total_points"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
df = df.merge(
    home_points[["player_id", "gameweek", "home_points_roll3"]],
    on=["player_id", "gameweek"],
    how="left"
)
df = df.merge(
    away_points[["player_id", "gameweek", "away_points_roll3"]],
    on=["player_id", "gameweek"],
    how="left"
)
df["home_points_roll3"] = df["home_points_roll3"].fillna(df["total_points_roll3"])
df["away_points_roll3"] = df["away_points_roll3"].fillna(df["total_points_roll3"])
df["home_points_diff_roll3"] = (df["home_points_roll3"] - df["away_points_roll3"]).fillna(0)

# Add cumulative minutes
df["cumulative_minutes"] = df.groupby("player_id")["minutes"].cumsum()

# Add team goal difference (3-GW rolling)
df["goal_diff"] = df.apply(
    lambda row: (row["team_h_score"] - row["team_a_score"]) if row["was_home"] else (row["team_a_score"] - row["team_h_score"]),
    axis=1
)
team_goal_diff = df.groupby(["team_id", "gameweek"])["goal_diff"].mean().reset_index()
team_goal_diff["team_goal_diff_roll3"] = (
    team_goal_diff.groupby("team_id")["goal_diff"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
df = df.merge(team_goal_diff[["team_id", "gameweek", "team_goal_diff_roll3"]], on=["team_id", "gameweek"], how="left")
df["team_goal_diff_roll3"] = df["team_goal_diff_roll3"].fillna(0)

# Add BPS share (3-GW rolling)
team_bps = df.groupby(["team_id", "gameweek"])["bps"].sum().reset_index(name="team_total_bps")
df = df.merge(team_bps, on=["team_id", "gameweek"], how="left")
df["bps_share"] = df["bps"] / df["team_total_bps"].replace(0, 1)
df["bps_share_roll3"] = (
    df.groupby("player_id")["bps_share"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Define features to lag
features_to_lag = [
    "minutes_roll3", "goals_scored_roll3", "assists_roll3", "clean_sheets_roll3",
    "yellow_cards_roll3", "bonus_roll3", "bps_roll3", "influence_roll3",
    "creativity_roll3", "threat_roll3", "ict_index_roll3", "total_points_roll3",
    "total_points_std3", "value_delta", "team_form_roll3", "fdr_roll3",
    "xcs_roll3", "opp_team_form_roll3", "player_team_points_share",
    "home_points_diff_roll3", "is_dgw", "cumulative_minutes", "team_goal_diff_roll3",
    "bps_share_roll3"
]

# Create lagged DataFrame
df_lagged = df.copy()
for feature in features_to_lag:
    if feature in df_lagged.columns:
        df_lagged[feature] = df_lagged.groupby("player_id")[feature].shift(1)
    else:
        raise ValueError(f"Feature {feature} not found in DataFrame before lagging")

# Verify output columns
output_columns = [
    "gameweek", "player_id", "name", "team_id", "position", "total_points",
    "total_points_roll3", "value", "value_delta", "team_form_roll3", "fdr", "fdr_roll3",
    "xg", "xg_roll3", "xa", "xa_roll3", "xcs", "xcs_roll3", "opp_team_form_roll3",
    "player_team_points_share", "home_points_diff_roll3", "is_dgw", "cumulative_minutes",
    "team_goal_diff_roll3", "bps_share_roll3"
]
missing_output = [col for col in output_columns if col not in df_lagged.columns]
if missing_output:
    raise ValueError(f"Missing output columns: {missing_output}")

# Save enhanced dataset
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_lagged.to_csv(OUTPUT_FILE, index=False)

# Show example output for one player
active_players = df_lagged.groupby("player_id")["minutes"].sum()
example_pid = active_players[active_players > 900].idxmax()
example_df = df_lagged[df_lagged["player_id"] == example_pid][output_columns].sort_values("gameweek")

print(f"Saved enhanced dataset with features to {OUTPUT_FILE}")
print(f"DataFrame shape: {df_lagged.shape}")
print(f"Columns: {list(df_lagged.columns)}")
print("\nExample player rolling stats:")
print(example_df.head(30).to_string(index=False))
