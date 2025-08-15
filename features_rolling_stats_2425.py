import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/fpl_2024_25_cleaned.csv")
OUTPUT_FILE = Path("data/processed/fpl_2024_25_features.csv")

# Load cleaned data
df = pd.read_csv(INPUT_FILE)
df.sort_values(by=["player_id", "gameweek"], inplace=True)

# Rolling features per player
rolling_features = [
    "minutes", "goals_scored", "assists", "clean_sheets",
    "yellow_cards", "bonus", "bps", "influence", "creativity",
    "threat", "ict_index", "total_points", "xg", "xa", "xcs"
]

# Conditionally add shots_on_target if available
if "shots_on_target" in df.columns:
    rolling_features.append("shots_on_target")
else:
    print("Warning: shots_on_target not found in input data. Setting shots_on_target_roll3 to 0.")
    df["shots_on_target_roll3"] = 0

# Add 3-week rolling mean per player
for feature in rolling_features:
    df[f"{feature}_roll3"] = (
        df.groupby("player_id")[feature]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

# For GW1, initialize rolling as 0 (no prior data)
df.loc[df["gameweek"] == 1, [f"{feature}_roll3" for feature in rolling_features]] = 0

# Add 3-week rolling std dev of points (volatility)
df["total_points_std3"] = (
    df.groupby("player_id")["total_points"]
    .rolling(window=3, min_periods=1)
    .std()
    .reset_index(level=0, drop=True)
)
df.loc[df["gameweek"] == 1, "total_points_std3"] = 0

# Add 3-week rolling mean for FDR
df["fdr_roll3"] = (
    df.groupby("player_id")["fdr"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
df.loc[df["gameweek"] == 1, "fdr_roll3"] = df["fdr"]

# Add team form (3-GW rolling mean of team points)
team_points = df.groupby(["team_id", "gameweek"])["total_points"].sum().reset_index(name="team_total_points")
team_points["team_form_roll3"] = (
    team_points.groupby("team_id")["team_total_points"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
df = df.merge(
    team_points[["team_id", "gameweek", "team_form_roll3"]],
    on=["team_id", "gameweek"],
    how="left"
)
df.loc[df["gameweek"] == 1, "team_form_roll3"] = 0  # No prior team form for GW1

# Add double gameweek indicator (is_dgw)
fixtures = pd.read_csv("data/raw/2024_25/fixtures.csv", encoding='utf-8')
fixtures["event"] = fixtures["event"].fillna(0).astype(int)
gw_team_counts = fixtures.groupby(["event", "team_h"]).size().reset_index(name="count_h")
gw_team_counts = gw_team_counts.merge(
    fixtures.groupby(["event", "team_a"]).size().reset_index(name="count_a"),
    left_on=["event", "team_h"],
    right_on=["event", "team_a"],
    how="outer"
)
gw_team_counts["team_id"] = gw_team_counts["team_h"].fillna(gw_team_counts["team_a"])
gw_team_counts["count"] = gw_team_counts["count_h"].fillna(0) + gw_team_counts["count_a"].fillna(0)
gw_team_counts["is_dgw"] = (gw_team_counts["count"] > 1).astype(int)
df = df.merge(
    gw_team_counts[["event", "team_id", "is_dgw"]],
    left_on=["gameweek", "team_id"],
    right_on=["event", "team_id"],
    how="left"
)
df["is_dgw"] = df["is_dgw"].fillna(0).astype(int)

# Add price delta (week-on-week change)
df["value_delta"] = df.groupby("player_id")["value"].diff()
df["value_delta"] = df["value_delta"].fillna(0)

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

# Save enhanced dataset
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved enhanced dataset with features to {OUTPUT_FILE}")
print("\nExample player rolling stats:")
print(df.head(30).to_string(index=False))
