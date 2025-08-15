import pandas as pd
import pulp
from pathlib import Path

# Load predictions
df = pd.read_csv("data/processed/gw1_2425_predictions.csv")

# Verify required columns
required_columns = ["player_id", "name", "position", "team_id", "predicted_points", "value"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in predictions: {missing_columns}")

# Filter for the latest gameweek
latest_gw = df["gameweek"].max()
df_gw = df[df["gameweek"] == latest_gw].copy()

# Ensure value is in millions
if df_gw["value"].max() > 100:  # Likely in tenths of millions
    df_gw["value"] = df_gw["value"] / 10
df_gw["value"] = df_gw["value"].astype(float)

# Check for valid data
if df_gw.empty:
    raise ValueError(f"No data for gameweek {latest_gw}")
if df_gw["predicted_points"].isna().any():
    raise ValueError("Missing predicted_points values")
if df_gw["value"].isna().any():
    raise ValueError("Missing value values")

# Prioritize high-scoring players (top 10% predicted_points per position)
for pos in ["GKP", "DEF", "MID", "FWD"]:
    pos_df = df_gw[df_gw["position"] == pos]
    if not pos_df.empty:
        threshold = pos_df["predicted_points"].quantile(0.90)  # Top 10%
        df_gw.loc[(df_gw["position"] == pos) & (df_gw["predicted_points"] < threshold), "predicted_points"] *= 0.7  # Strong discount

# Define position mapping
position_map = {
    "GKP": {"min_start": 1, "max_start": 1, "squad": 2},
    "DEF": {"min_start": 3, "max_start": 5, "squad": 5},
    "MID": {"min_start": 2, "max_start": 5, "squad": 5},
    "FWD": {"min_start": 1, "max_start": 3, "squad": 3}
}

# Position-specific cost weights
cost_weights = {
    "GKP": 0.01,  # Low weight for goalkeepers
    "DEF": 0.01,  # Low weight for defenders
    "MID": 0.15,  # Higher weight for midfielders
    "FWD": 0.15   # Higher weight for forwards
}

# Initialize PuLP problem
prob = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)

# Decision variables
squad = pulp.LpVariable.dicts("Squad", df_gw["player_id"], cat="Binary")  # In squad
start = pulp.LpVariable.dicts("Start", df_gw["player_id"], cat="Binary")  # In starting XI

# Objective: Maximize weighted predicted points + position-specific cost
points_weight = 10.0  # Heavy weight for predicted points
prob += pulp.lpSum([points_weight * df_gw.loc[i, "predicted_points"] * start[df_gw.loc[i, "player_id"]]
                    + cost_weights[df_gw.loc[i, "position"]] * df_gw.loc[i, "value"] * squad[df_gw.loc[i, "player_id"]]
                    for i in df_gw.index])

# Constraints
# 1. Budget constraint (£100m max)
prob += pulp.lpSum([df_gw.loc[i, "value"] * squad[df_gw.loc[i, "player_id"]]
                    for i in df_gw.index]) <= 100, "BudgetMax"

# 2. Squad size (15 players)
prob += pulp.lpSum([squad[pid] for pid in df_gw["player_id"]]) == 15, "SquadSize"

# 3. Position constraints
for pos, limits in position_map.items():
    prob += pulp.lpSum([squad[df_gw.loc[i, "player_id"]] for i in df_gw.index
                        if df_gw.loc[i, "position"] == pos]) == limits["squad"], f"Squad_{pos}"
    prob += pulp.lpSum([start[df_gw.loc[i, "player_id"]] for i in df_gw.index
                        if df_gw.loc[i, "position"] == pos]) >= limits["min_start"], f"MinStart_{pos}"
    prob += pulp.lpSum([start[df_gw.loc[i, "player_id"]] for i in df_gw.index
                        if df_gw.loc[i, "position"] == pos]) <= limits["max_start"], f"MaxStart_{pos}"

# 4. Starting XI size (11 players)
prob += pulp.lpSum([start[pid] for pid in df_gw["player_id"]]) == 11, "StartingXI"

# 5. Max 3 players per team
for team_id in df_gw["team_id"].unique():
    prob += pulp.lpSum([squad[df_gw.loc[i, "player_id"]] for i in df_gw.index
                        if df_gw.loc[i, "team_id"] == team_id]) <= 3, f"MaxTeam_{team_id}"

# 6. Starting players must be in squad
for pid in df_gw["player_id"]:
    prob += start[pid] <= squad[pid], f"StartInSquad_{pid}"

# 7. Ensure at least 4 premium players (value >= 8.0)
prob += pulp.lpSum([squad[df_gw.loc[i, "player_id"]] for i in df_gw.index
                    if df_gw.loc[i, "value"] >= 8.0]) >= 4, "MinPremiumPlayers"

# 8. Ensure bench players have predicted_points > 3.0
for i in df_gw.index:
    if df_gw.loc[i, "predicted_points"] <= 3.0:
        prob += squad[df_gw.loc[i, "player_id"]] <= start[df_gw.loc[i, "player_id"]], f"BenchPoints_{df_gw.loc[i, 'player_id']}"

# Solve
prob.solve()

# Check solution status
if pulp.LpStatus[prob.status] != "Optimal":
    raise ValueError(f"No optimal solution found: {pulp.LpStatus[prob.status]}")

# Extract squad
squad_players = []
starting_players = []
total_points = 0
total_cost = 0
premium_count = 0
for i in df_gw.index:
    pid = df_gw.loc[i, "player_id"]
    if squad[pid].value() > 0.5:
        is_premium = df_gw.loc[i, "value"] >= 8.0
        squad_players.append({
            "player_id": pid,
            "name": df_gw.loc[i, "name"],
            "position": df_gw.loc[i, "position"],
            "team_id": df_gw.loc[i, "team_id"],
            "predicted_points": df_gw.loc[i, "predicted_points"],
            "value": df_gw.loc[i, "value"],
            "starting": start[pid].value() > 0.5,
            "premium": is_premium
        })
        total_cost += df_gw.loc[i, "value"]
        if is_premium:
            premium_count += 1
        if start[pid].value() > 0.5:
            total_points += df_gw.loc[i, "predicted_points"]
            starting_players.append(squad_players[-1])

# Convert to DataFrame
squad_df = pd.DataFrame(squad_players)
starting_df = pd.DataFrame(starting_players)

# Save results
output_path = Path("data/processed/optimized_squad.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
squad_df.to_csv(output_path, index=False)

# Print results
print(f"\nOptimal Squad (Total Cost: £{total_cost:.1f}m, Predicted Points: {total_points:.2f}, Premium Players: {premium_count})")
print("\nStarting XI:")
print(starting_df[["name", "position", "team_id", "predicted_points", "value", "premium"]].to_string(index=False))
print("\nBench:")
print(squad_df[~squad_df["starting"]][["name", "position", "team_id", "predicted_points", "value", "premium"]].to_string(index=False))

# Verify squad constraints
for pos, limits in position_map.items():
    squad_count = len(squad_df[squad_df["position"] == pos])
    start_count = len(starting_df[starting_df["position"] == pos])
    print(f"\n{pos}: Squad={squad_count}/{limits['squad']}, Starting={start_count} ({limits['min_start']}–{limits['max_start']})")
team_counts = squad_df["team_id"].value_counts()
print("\nTeam Counts:")
print(team_counts)

# Verify budget allocation by position
for pos in position_map:
    pos_cost = squad_df[squad_df["position"] == pos]["value"].sum()
    print(f"\n{pos} Total Cost: £{pos_cost:.1f}m")
