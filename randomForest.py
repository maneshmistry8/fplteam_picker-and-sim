import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Verify scikit-learn version
print(f"Using scikit-learn version: {sklearn.__version__}")

# Load features dataset
df = pd.read_csv("data/processed/fpl_2023_24_features.csv")

# Verify required columns
required_columns = [
    "gameweek", "player_id", "name", "position", "team_id", "value", "total_points",
    "minutes_roll3", "goals_scored_roll3", "assists_roll3", "clean_sheets_roll3",
    "yellow_cards_roll3", "bonus_roll3", "bps_roll3", "influence_roll3",
    "creativity_roll3", "threat_roll3", "ict_index_roll3", "total_points_roll3",
    "total_points_std3", "value_delta", "team_form_roll3", "fdr_roll3",
    "xcs_roll3", "opp_team_form_roll3", "player_team_points_share",
    "home_points_diff_roll3", "is_dgw", "cumulative_minutes", "team_goal_diff_roll3",
    "bps_share_roll3"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in input DataFrame: {missing_columns}")

# Handle missing values
df.fillna(0, inplace=True)

# Define position-specific features
position_features = {
    "FWD": [
        "minutes_roll3", "goals_scored_roll3", "assists_roll3", "bonus_roll3",
        "bps_roll3", "influence_roll3", "creativity_roll3", "threat_roll3",
        "ict_index_roll3", "total_points_roll3", "total_points_std3",
        "value_delta", "team_form_roll3", "fdr_roll3", "xg_roll3", "xa_roll3",
        "opp_team_form_roll3", "player_team_points_share", "home_points_diff_roll3",
        "is_dgw", "cumulative_minutes", "team_goal_diff_roll3", "bps_share_roll3"
    ],
    "MID": [
        "minutes_roll3", "goals_scored_roll3", "assists_roll3", "bonus_roll3",
        "bps_roll3", "influence_roll3", "creativity_roll3", "threat_roll3",
        "ict_index_roll3", "total_points_roll3", "total_points_std3",
        "value_delta", "team_form_roll3", "fdr_roll3", "xg_roll3", "xa_roll3",
        "opp_team_form_roll3", "player_team_points_share", "home_points_diff_roll3",
        "is_dgw", "cumulative_minutes", "team_goal_diff_roll3", "bps_share_roll3"
    ],
    "DEF": [
        "minutes_roll3", "clean_sheets_roll3", "yellow_cards_roll3", "bonus_roll3",
        "bps_roll3", "influence_roll3", "total_points_roll3", "total_points_std3",
        "value_delta", "team_form_roll3", "fdr_roll3", "xcs_roll3",
        "opp_team_form_roll3", "player_team_points_share", "home_points_diff_roll3",
        "is_dgw", "cumulative_minutes", "team_goal_diff_roll3", "bps_share_roll3"
    ],
    "GKP": [
        "minutes_roll3", "clean_sheets_roll3", "yellow_cards_roll3", "saves",
        "bonus_roll3", "bps_roll3", "influence_roll3", "total_points_roll3",
        "total_points_std3", "value_delta", "team_form_roll3", "fdr_roll3", "xcs_roll3",
        "opp_team_form_roll3", "player_team_points_share", "home_points_diff_roll3",
        "is_dgw", "cumulative_minutes", "team_goal_diff_roll3", "bps_share_roll3"
    ]
}

# Train and evaluate position-specific models
results = {}
for position in ["FWD", "MID", "DEF", "GKP"]:
    print(f"\nTraining model for {position}")
    df_pos = df[df["position"] == position]
    if df_pos.empty:
        print(f"No data for {position}, skipping")
        continue

    features = position_features[position]
    X = df_pos[features]
    y = df_pos["total_points"]

    # Split: Train on GW1–37, Test on GW38
    X_train = X[df_pos["gameweek"] < 38]
    y_train = y[df_pos["gameweek"] < 38]
    X_test = X[df_pos["gameweek"] == 38]
    y_test = y[df_pos["gameweek"] == 38]

    if X_test.empty:
        print(f"No GW38 data for {position}, skipping")
        continue

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE for {position}: {mae:.2f}")

    # Store results
    results[position] = {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "df_test": df_pos[df_pos["gameweek"] == 38][["gameweek", "player_id", "name", "position", "team_id", "value", "total_points"]]
    }

    # Feature importance
    importances = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    print(f"\nFeature Importances for {position}:")
    print(importances)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importances)
    plt.title(f"Feature Importances for {position}")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"feature_importances_{position}.png", dpi=300)
    plt.close()
    print(f"Saved feature importances plot to 'feature_importances_{position}.png'")

# Combine predictions
df_test = pd.concat([
    results[pos]["df_test"].assign(predicted_points=results[pos]["y_pred"])
    for pos in results
], ignore_index=True)
df_test.to_csv("data/processed/predictions_test.csv", index=False)
print("Saved test set predictions to 'data/processed/predictions_test.csv'")

# Overall MAE
overall_mae = mean_absolute_error(df_test["total_points"], df_test["predicted_points"])
print(f"\nOverall MAE across all positions: {overall_mae:.2f}")

# Plot error distribution
df_test["error"] = abs(df_test["total_points"] - df_test["predicted_points"])
plt.figure(figsize=(10, 6))
sns.histplot(df_test["error"], bins=30)
plt.title("Prediction Error Distribution")
plt.xlabel("Absolute Error (Points)")
plt.savefig("error_distribution.png", dpi=300)
plt.close()
print("Saved error distribution plot to 'error_distribution.png'")
