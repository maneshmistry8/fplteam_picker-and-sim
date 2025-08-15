import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path

# Load 23/24 features for training
df_2324 = pd.read_csv("data/processed/fpl_2023_24_features.csv")

# Load 24/25 GW1 features for prediction
df_2425 = pd.read_csv("data/processed/fpl_2024_25_features.csv")
df_gw1 = df_2425[df_2425["gameweek"] == 1].copy()  # Create a copy to avoid warnings

# Position-specific features
position_features = {
    "FWD": ["minutes_roll3", "goals_scored_roll3", "assists_roll3", "bonus_roll3",
            "bps_roll3", "influence_roll3", "creativity_roll3", "threat_roll3",
            "ict_index_roll3", "total_points_roll3", "total_points_std3",
            "value_delta", "team_form_roll3", "fdr_roll3", "xg_roll3", "xa_roll3",
            "opp_team_form_roll3", "player_team_points_share", "home_points_diff_roll3",
            "is_dgw", "cumulative_minutes", "team_goal_diff_roll3", "bps_share_roll3",
            "shots_on_target_roll3"],
    "MID": ["minutes_roll3", "goals_scored_roll3", "assists_roll3", "bonus_roll3",
            "bps_roll3", "influence_roll3", "creativity_roll3", "threat_roll3",
            "ict_index_roll3", "total_points_roll3", "total_points_std3",
            "value_delta", "team_form_roll3", "fdr_roll3", "xg_roll3", "xa_roll3",
            "opp_team_form_roll3", "player_team_points_share", "home_points_diff_roll3",
            "is_dgw", "cumulative_minutes", "team_goal_diff_roll3", "bps_share_roll3",
            "shots_on_target_roll3"],
    "DEF": ["minutes_roll3", "clean_sheets_roll3", "yellow_cards_roll3", "bonus_roll3",
            "bps_roll3", "influence_roll3", "total_points_roll3", "total_points_std3",
            "value_delta", "team_form_roll3", "fdr_roll3", "xcs_roll3",
            "opp_team_form_roll3", "player_team_points_share", "home_points_diff_roll3",
            "is_dgw", "cumulative_minutes", "team_goal_diff_roll3", "bps_share_roll3",
            "shots_on_target_roll3"],
    "GKP": ["minutes_roll3", "clean_sheets_roll3", "yellow_cards_roll3", "saves",
            "bonus_roll3", "bps_roll3", "influence_roll3", "total_points_roll3",
            "total_points_std3", "value_delta", "team_form_roll3", "fdr_roll3", "xcs_roll3",
            "opp_team_form_roll3", "player_team_points_share", "home_points_diff_roll3",
            "is_dgw", "cumulative_minutes", "team_goal_diff_roll3", "bps_share_roll3",
            "shots_on_target_roll3"]
}

# Train and predict for each position
predictions = []
for position in ["FWD", "MID", "DEF", "GKP"]:
    df_pos_2324 = df_2324[df_2324["position"] == position]
    if df_pos_2324.empty:
        continue

    features = position_features[position]
    # Verify all features exist in training data
    missing_features = [f for f in features if f not in df_pos_2324.columns]
    if missing_features:
        print(f"Warning: Features {missing_features} missing for {position} in 2023-24 data. Filling with 0.")
        for f in missing_features:
            df_pos_2324[f] = 0

    X_train = df_pos_2324[features]
    y_train = df_pos_2324["total_points"]

    # Tune Random Forest with GridSearchCV
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5]
    }
    model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"Best parameters for {position}: {model.best_params_}")

    df_pos_gw1 = df_gw1[df_gw1["position"] == position].copy()
    if df_pos_gw1.empty:
        continue

    # Verify all features exist in GW1 data
    missing_features = [f for f in features if f not in df_pos_gw1.columns]
    if missing_features:
        print(f"Warning: Features {missing_features} missing for {position} in GW1 2024-25 data. Filling with 0.")
        for f in missing_features:
            df_pos_gw1[f] = 0

    X_gw1 = df_pos_gw1[features]
    y_pred = model.best_estimator_.predict(X_gw1)

    # Assign predictions using .loc
    df_pos_gw1.loc[:, "predicted_points"] = y_pred
    predictions.append(df_pos_gw1[["gameweek", "player_id", "name", "position", "team_id", "value", "predicted_points"]])

# Combine GW1 predictions
gw1_predictions = pd.concat(predictions, ignore_index=True)
gw1_predictions.to_csv("data/processed/gw1_2425_predictions.csv", index=False)
print("Saved GW1 2024-25 predictions to data/processed/gw1_2425_predictions.csv")
