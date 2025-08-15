import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

def generate_features(cleaned_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Build GW1 2025/26 feature table using 24/25 GW36–38 rollups where available."""
    # Work only on 2025/26 rows; explicitly set GW1
    features_df = cleaned_df[cleaned_df["season"] == "2025-26"].copy()
    features_df["gameweek"] = 1

    # ---- Rolling features from GW36-38 of 2024/25 ----
    df_2425 = cleaned_df[cleaned_df["season"] == "2024-25"].copy()
    last_3_gws = df_2425[df_2425["gameweek"].isin([36, 37, 38])].copy()

    if last_3_gws.empty:
        print("Warning: No GW36-38 2024/25 data. Using default rolling features.")
        rolling_features = features_df[["player_id"]].copy()
        for col in [
            "minutes_roll3","goals_scored_roll3","assists_roll3","clean_sheets_roll3",
            "saves_roll3","bonus_roll3","bps_roll3","influence_roll3","creativity_roll3",
            "threat_roll3","ict_index_roll3","yellow_cards_roll3","goals_conceded_roll3",
            "total_points_roll3","total_points_std3"
        ]:
            rolling_features[col] = 0.0
    else:
        agg = last_3_gws.groupby("player_id").agg({
            "minutes": "mean",
            "goals_scored": "mean",
            "assists": "mean",
            "clean_sheets": "mean",
            "saves": "mean",
            "bonus": "mean",
            "bps": "mean",
            "influence": "mean",
            "creativity": "mean",
            "threat": "mean",
            "ict_index": "mean",
            "yellow_cards": "mean",
            "goals_conceded": "mean",
            "total_points": ["mean", "std"]
        }).reset_index()

        agg.columns = [
            "player_id",
            "minutes_roll3","goals_scored_roll3","assists_roll3","clean_sheets_roll3",
            "saves_roll3","bonus_roll3","bps_roll3","influence_roll3","creativity_roll3",
            "threat_roll3","ict_index_roll3","yellow_cards_roll3","goals_conceded_roll3",
            "total_points_roll3","total_points_std3"
        ]
        rolling_features = agg

    features_df = features_df.merge(rolling_features, on="player_id", how="left")

    # ---- GW1 2025/26 fixture context (FDR, home/away, opponent) ----
    gw1_fixtures = fixtures_df[(fixtures_df["season"] == "2025-26") & (fixtures_df["gameweek"] == 1)].copy()

    # Remove any existing fixture_id to avoid collisions
    features_df = features_df.drop(columns=[c for c in ["fixture_id","fixture_id_x","fixture_id_y"] if c in features_df.columns], errors="ignore")

    if gw1_fixtures.empty:
        print("Warning: No GW1 2025/26 fixtures found. Using default FDR/home/opp.")
        features_df["fdr_roll3"] = 3.0
        features_df["was_home"] = 0
        features_df["opponent_id"] = 0
    else:
        # home join
        home = features_df.merge(
            gw1_fixtures[["fixture_id","home_team_id","away_team_id","home_fdr","away_fdr"]],
            left_on="team_id", right_on="home_team_id", how="left"
        )
        home["fdr_roll3"] = home["home_fdr"]
        home["was_home"] = 1
        home["opponent_id"] = home["away_team_id"]

        # away join
        away = features_df.merge(
            gw1_fixtures[["fixture_id","home_team_id","away_team_id","home_fdr","away_fdr"]],
            left_on="team_id", right_on="away_team_id", how="left"
        )
        away["fdr_roll3"] = away["away_fdr"]
        away["was_home"] = 0
        away["opponent_id"] = away["home_team_id"]

        merged = pd.concat([home, away], ignore_index=True)
        merged["_match_found"] = merged["fdr_roll3"].notna() | merged["opponent_id"].notna()
        merged = merged.sort_values("_match_found", ascending=False)

        features_df = merged.drop_duplicates(subset="player_id", keep="first")
        features_df = features_df.drop(columns=[
            "home_team_id","away_team_id","home_fdr","away_fdr","fixture_id","_match_found"
        ], errors="ignore")

        features_df["fdr_roll3"] = features_df["fdr_roll3"].fillna(3.0)
        features_df["was_home"] = features_df["was_home"].fillna(0).astype(int)
        features_df["opponent_id"] = features_df["opponent_id"].fillna(0).astype(int)

    # ---- Additional team/player rollups (safe if last_3_gws empty) ----
    features_df["value_delta"] = 0.0  # one row per player in 25/26 → 0

    if last_3_gws.empty:
        features_df["team_form_roll3"] = 0.0
        features_df["opp_team_form_roll3"] = 0.0
        features_df["player_team_points_share"] = 0.0
        features_df["home_points_diff_roll3"] = 0.0
        features_df["is_dgw"] = 0
        features_df["cumulative_minutes"] = 0.0
        features_df["team_goal_diff_roll3"] = 0.0
        features_df["bps_share_roll3"] = 0.0
    else:
        team_pts_mean = last_3_gws.groupby("team_id")["total_points"].mean()
        team_pts_sum  = last_3_gws.groupby("team_id")["total_points"].sum()
        player_pts_sum = last_3_gws.groupby("player_id")["total_points"].sum()
        player_min_sum = last_3_gws.groupby("player_id")["minutes"].sum()
        team_bps_sum   = last_3_gws.groupby("team_id")["bps"].sum()
        player_bps_sum = last_3_gws.groupby("player_id")["bps"].sum()

        features_df["team_form_roll3"] = team_pts_mean.reindex(features_df["team_id"]).fillna(0).values
        features_df["opp_team_form_roll3"] = team_pts_mean.reindex(features_df["opponent_id"]).fillna(0).values

        denom_pts = team_pts_sum.reindex(features_df["team_id"]).fillna(1).values
        features_df["player_team_points_share"] = (player_pts_sum.reindex(features_df["player_id"]).fillna(0).values / denom_pts)

        home_mean = last_3_gws[last_3_gws["was_home"] == 1]["total_points"].mean()
        away_mean = last_3_gws[last_3_gws["was_home"] == 0]["total_points"].mean()
        features_df["home_points_diff_roll3"] = (home_mean - away_mean) if pd.notna(home_mean) and pd.notna(away_mean) else 0.0

        features_df["is_dgw"] = 0
        features_df["cumulative_minutes"] = player_min_sum.reindex(features_df["player_id"]).fillna(0).values

        goal_diff = (last_3_gws.groupby("team_id")["goals_scored"].sum()
                     - last_3_gws.groupby("team_id")["goals_conceded"].sum())
        features_df["team_goal_diff_roll3"] = goal_diff.reindex(features_df["team_id"]).fillna(0).values

        denom_bps = team_bps_sum.reindex(features_df["team_id"]).fillna(1).values
        features_df["bps_share_roll3"] = (player_bps_sum.reindex(features_df["player_id"]).fillna(0).values / denom_bps)

    # ---- Encodings (for convenience; for rigor, persist encoders) ----
    le_pos  = LabelEncoder()
    le_team = LabelEncoder()
    le_opp  = LabelEncoder()

    features_df["position_enc"]  = le_pos.fit_transform(features_df["position"])
    features_df["team_enc"]      = le_team.fit_transform(features_df["team_id"])
    features_df["opponent_enc"]  = le_opp.fit_transform(features_df["opponent_id"].fillna(0))

    features_df["was_home"] = features_df["was_home"].astype(int)

    return features_df.fillna(0)

def train_models(cleaned_df: pd.DataFrame):
    """Train RFs per position. If roll features are missing in cleaned_df, they’re created as zeros."""
    positions = ["FWD", "MID", "DEF", "GKP"]
    position_features = {
        "FWD": [
            "minutes_roll3","goals_scored_roll3","assists_roll3","bonus_roll3",
            "bps_roll3","influence_roll3","creativity_roll3","threat_roll3",
            "ict_index_roll3","total_points_roll3","total_points_std3",
            "value_delta","team_form_roll3","fdr_roll3","player_team_points_share",
            "home_points_diff_roll3","is_dgw","cumulative_minutes","team_goal_diff_roll3",
            "bps_share_roll3","position_enc","team_enc","opponent_enc","was_home"
        ],
        "MID": [
            "minutes_roll3","goals_scored_roll3","assists_roll3","bonus_roll3",
            "bps_roll3","influence_roll3","creativity_roll3","threat_roll3",
            "ict_index_roll3","total_points_roll3","total_points_std3",
            "value_delta","team_form_roll3","fdr_roll3","player_team_points_share",
            "home_points_diff_roll3","is_dgw","cumulative_minutes","team_goal_diff_roll3",
            "bps_share_roll3","position_enc","team_enc","opponent_enc","was_home"
        ],
        "DEF": [
            "minutes_roll3","clean_sheets_roll3","yellow_cards_roll3","bonus_roll3",
            "bps_roll3","influence_roll3","total_points_roll3","total_points_std3",
            "value_delta","team_form_roll3","fdr_roll3","player_team_points_share",
            "home_points_diff_roll3","is_dgw","cumulative_minutes","team_goal_diff_roll3",
            "bps_share_roll3","position_enc","team_enc","opponent_enc","was_home"
        ],
        "GKP": [
            "minutes_roll3","clean_sheets_roll3","saves_roll3","bonus_roll3",
            "bps_roll3","influence_roll3","total_points_roll3","total_points_std3",
            "value_delta","team_form_roll3","fdr_roll3","player_team_points_share",
            "home_points_diff_roll3","is_dgw","cumulative_minutes","team_goal_diff_roll3",
            "bps_share_roll3","position_enc","team_enc","opponent_enc","was_home"
        ]
    }

    # Ensure required cols exist (fill zeros)
    needed = set(col for cols in position_features.values() for col in cols)
    if "position" not in cleaned_df.columns:
        raise ValueError("cleaned_df must include a 'position' column.")

    for col in needed:
        if col not in cleaned_df.columns:
            cleaned_df[col] = 0

    # Simple encodings for training
    cleaned_df["position_enc"] = LabelEncoder().fit_transform(cleaned_df["position"])
    cleaned_df["team_enc"]     = LabelEncoder().fit_transform(cleaned_df["team_id"])
    cleaned_df["opponent_enc"] = LabelEncoder().fit_transform(cleaned_df.get("opponent_id", pd.Series(0, index=cleaned_df.index)))
    if "was_home" not in cleaned_df.columns:
        cleaned_df["was_home"] = 0
    cleaned_df["was_home"] = cleaned_df["was_home"].astype(int)

    Path("data/processed/multi_season").mkdir(parents=True, exist_ok=True)
    models = {}

    for pos in positions:
        df_pos = cleaned_df[cleaned_df["position"] == pos].copy()
        if df_pos.empty:
            print(f"Warning: No data for {pos}. Skipping.")
            continue

        X = df_pos[position_features[pos]].fillna(0)
        y = df_pos["total_points"].fillna(0)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        models[pos] = model
        joblib.dump(model, f"data/processed/multi_season/fpl_rf_model_{pos}.joblib")
        print(f"Trained and saved model for {pos}")

    return models

def main():
    # Inputs
    try:
        cleaned_df = pd.read_csv("data/processed/fpl_2024-25_cleaned.csv")
    except FileNotFoundError:
        print("Error: data/processed/fpl_2024-25_cleaned.csv not found. Run clean_data.py first.")
        return
    try:
        fixtures_df = pd.read_csv("data/raw/fpl_fixtures.csv")
    except FileNotFoundError:
        print("Error: data/raw/fpl_fixtures.csv not found. Run datadownload.py first.")
        return

    # Build 25/26 GW1 features
    features_df = generate_features(cleaned_df, fixtures_df)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    features_df.to_csv("data/processed/fpl_2025-26_features.csv", index=False)
    print("Saved features to data/processed/fpl_2025-26_features.csv")

    # Train RFs on 24/25 (with caveat that many roll cols may be zeros)
    train_models(cleaned_df)
    print("All models trained and saved")

if __name__ == "__main__":
    main()
