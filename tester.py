import pandas as pd
import pulp
from pathlib import Path

df = pd.read_csv("data/processed/predictions_test.csv")
# Filter for Salah and Son
print(df[df["name"].str.contains("Salah|Son", case=False)][["gameweek", "name", "position", "team_id", "value", "predicted_points", "total_points"]])
# Check top predicted points
print(df[["name", "position", "team_id", "value", "predicted_points"]].sort_values("predicted_points", ascending=False).head(10))
