import pandas as pd
from pathlib import Path

def clean_data(players_df, teams_df, fixtures_df, gw_data_df=None):
    """Clean raw FPL data for processing."""
    # Filter available players for 2025-26
    players_df = players_df[(players_df["season"] != "2025-26") | (players_df["status"] == "a")]

    # Merge players with team data
    cleaned_df = players_df.merge(
        teams_df[["team_id", "team_name", "season"]],
        on=["team_id", "season"],
        how="left"
    )

    # Merge with gameweek data if available (for historical seasons)
    if gw_data_df is not None and not gw_data_df.empty:
        cleaned_df = cleaned_df.merge(
            gw_data_df,
            on=["player_id", "season"],
            how="left",
            suffixes=("", "_gw")
        )
    else:
        # Add default columns for gameweek data
        for col in [
            "gameweek", "minutes", "goals_scored", "assists", "clean_sheets",
            "saves", "bonus", "bps", "influence", "creativity", "threat",
            "ict_index", "total_points", "goals_conceded", "yellow_cards"
        ]:
            cleaned_df[col] = 0

    # Merge fixtures for BOTH home and away
    fx = fixtures_df[[
        "fixture_id", "gameweek", "home_team_id", "away_team_id", "home_fdr", "away_fdr", "season"
    ]]

    # left join as home
    home = cleaned_df.merge(
        fx, left_on=["team_id", "gameweek", "season"],
        right_on=["home_team_id", "gameweek", "season"], how="left"
    )
    home["was_home"] = 1
    home["fdr"] = home["home_fdr"]

    # left join as away
    away = cleaned_df.merge(
        fx, left_on=["team_id", "gameweek", "season"],
        right_on=["away_team_id", "gameweek", "season"], how="left"
    )
    away["was_home"] = 0
    away["fdr"] = away["away_fdr"]

    # Combine: prefer whichever side matched
    merged = home.copy()
    for col in ["fixture_id", "home_team_id", "away_team_id", "home_fdr", "away_fdr", "fdr"]:
        merged[col] = merged[col].fillna(away[col])

    merged["was_home"] = merged["was_home"].where(merged["fixture_id"].notna(), away["was_home"])

    cleaned_df = merged

    # Handle missing values
    cleaned_df = cleaned_df.fillna({
        "minutes": 0, "goals_scored": 0, "assists": 0,
        "clean_sheets": 0, "saves": 0, "bonus": 0,
        "bps": 0, "influence": 0, "creativity": 0,
        "threat": 0, "ict_index": 0, "total_points": 0,
        "goals_conceded": 0, "yellow_cards": 0,
        "fdr": 3, "team_name": "Unknown", "was_home": 0
    })

    # Drop unnecessary columns
    cleaned_df = cleaned_df.drop(
        columns=["position_id", "home_team_id", "away_team_id", "home_fdr", "away_fdr"],
        errors="ignore"
    )

    return cleaned_df

def main():
    # Ensure data/raw directory exists
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    # Load raw data
    try:
        players_df = pd.read_csv("data/raw/fpl_players.csv")
    except FileNotFoundError:
        print("Error: data/raw/fpl_players.csv not found. Run datadownload.py first.")
        return
    try:
        teams_df = pd.read_csv("data/raw/fpl_teams.csv")
    except FileNotFoundError:
        print("Error: data/raw/fpl_teams.csv not found. Run datadownload.py first.")
        return
    try:
        fixtures_df = pd.read_csv("data/raw/fpl_fixtures.csv")
    except FileNotFoundError:
        print("Error: data/raw/fpl_fixtures.csv not found. Run datadownload.py first.")
        return
    try:
        gw_data_df = pd.read_csv("data/raw/fpl_gameweek_data.csv")
    except FileNotFoundError:
        print("Warning: data/raw/fpl_gameweek_data.csv not found. Proceeding without gameweek data.")
        gw_data_df = pd.DataFrame()

    # Clean data
    cleaned_df = clean_data(players_df, teams_df, fixtures_df, gw_data_df)

    # Save cleaned data
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    cleaned_df[cleaned_df["season"] != "2025-26"].to_csv("data/processed/fpl_2024-25_cleaned.csv", index=False)
    cleaned_df[cleaned_df["season"] == "2025-26"].to_csv("data/processed/fpl_2025-26_cleaned.csv", index=False)
    print("Saved cleaned data to data/processed/fpl_2024-25_cleaned.csv and fpl_2025-26_cleaned.csv")

if __name__ == "__main__":
    main()
