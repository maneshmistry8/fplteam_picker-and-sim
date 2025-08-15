import pandas as pd
from pathlib import Path

# Load squad
squad_df = pd.read_csv("data/processed/optimized_squad.csv")
starting_11 = squad_df[squad_df["starting"]]["player_id"].tolist()

# Load 2024-25 cleaned data
df_2425 = pd.read_csv("data/processed/fpl_2024_25_cleaned.csv")

# Simulate points for each GW
gw_scores = []
total_score = 0
for gw in range(1, 39):
    gw_data = df_2425[df_2425["gameweek"] == gw]
    gw_points = gw_data[gw_data["player_id"].isin(starting_11)]["total_points"].sum()
    gw_scores.append({"gameweek": gw, "points": gw_points})
    total_score += gw_points
    print(f"GW{gw}: {gw_points} points")

# Save simulation results
simulation_df = pd.DataFrame(gw_scores)
simulation_df.to_csv("data/processed/2425_simulation.csv", index=False)
print(f"\nTotal Season Points: {total_score}")
print("Saved simulation to data/processed/2425_simulation.csv")
