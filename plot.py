import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the feature dataset
df = pd.read_csv("data/processed/fpl_2023_24_features.csv")

# Select active players: those with >900 minutes total
active_players = df.groupby("player_id")["minutes"].sum()
active_ids = active_players[active_players > 900].index
df = df[df["player_id"].isin(active_ids)]

# Select a handful of players for clarity
top_players = df["player_id"].value_counts().nlargest(5).index
df_top = df[df["player_id"].isin(top_players)]

# Pick pairs of actual vs rolling
feature_pairs = [
    ("total_points", "total_points_roll3"),
    ("minutes", "minutes_roll3"),
    ("bps", "bps_roll3"),
    ("ict_index", "ict_index_roll3")
]

# Plot each pair as subplots
num_plots = len(feature_pairs)
fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4 * num_plots), sharex=True)

for i, (actual, rolling) in enumerate(feature_pairs):
    ax = axes[i]
    sns.lineplot(data=df_top, x="gameweek", y=actual, hue="name", ax=ax, linestyle='--', legend=False)
    sns.lineplot(data=df_top, x="gameweek", y=rolling, hue="name", ax=ax, legend="brief" if i == 0 else False)
    ax.set_title(f"{actual} vs {rolling} (Top Active Players)")
    ax.set_ylabel("Value")
    ax.grid(True)

plt.xlabel("Gameweek")
plt.tight_layout()
plt.savefig("fpl_actual_vs_rolling_top_players.png", dpi=300)
print("Saved plot to fpl_actual_vs_rolling_top_players.png")
