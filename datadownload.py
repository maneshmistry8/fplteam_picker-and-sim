import requests
import pandas as pd
from pathlib import Path
import io

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
BASE_URL = "https://fantasy.premierleague.com/api/"
VAASTAV_RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

Path("data/raw").mkdir(parents=True, exist_ok=True)

def fetch_csv(url):
    r = requests.get(url)
    if r.status_code != 200:
        print(f"⚠️ Failed to fetch {url}")
        return None
    return pd.read_csv(io.StringIO(r.text))

def download_player_team_fixture_archive(season):
    """Download players, teams, and fixtures from vaastav archive."""
    season_path = season.replace("-", "%2F")
    players_url = f"{VAASTAV_RAW}/{season}/players_raw.csv"
    teams_url = f"{VAASTAV_RAW}/{season}/teams.csv"
    fixtures_url = f"{VAASTAV_RAW}/{season}/fixtures.csv"

    players_df = fetch_csv(players_url)
    teams_df = fetch_csv(teams_url)
    fixtures_df = fetch_csv(fixtures_url)

    if players_df is not None:
        players_df = players_df.rename(columns={
            "id": "player_id",
            "web_name": "name",
            "element_type": "position_id",
            "team": "team_id",
            "now_cost": "value"
        })
        players_df["value"] = players_df["value"] / 10
        pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        players_df["position"] = players_df["position_id"].map(pos_map)
        players_df["season"] = season
    if teams_df is not None:
        teams_df = teams_df.rename(columns={
            "id": "team_id",
            "name": "team_name",
            "short_name": "team_short_name",
            "strength": "strength"
        })
        teams_df["season"] = season
    if fixtures_df is not None:
        fixtures_df = fixtures_df.rename(columns={
            "id": "fixture_id",
            "event": "gameweek",
            "team_h": "home_team_id",
            "team_a": "away_team_id",
            "team_h_difficulty": "home_fdr",
            "team_a_difficulty": "away_fdr"
        })
        fixtures_df["season"] = season

    return players_df, teams_df, fixtures_df

def download_gameweeks_archive(season):
    """Download all GW CSVs from vaastav archive for a season."""
    gw_dfs = []
    for gw in range(1, 39):
        gw_url = f"{VAASTAV_RAW}/{season}/gws/gw{gw}.csv"
        gw_df = fetch_csv(gw_url)
        if gw_df is None:
            continue
        gw_out = pd.DataFrame({
            "player_id": gw_df["element"],
            "gameweek": gw,
            "minutes": gw_df["minutes"],
            "goals_scored": gw_df["goals_scored"],
            "assists": gw_df["assists"],
            "clean_sheets": gw_df["clean_sheets"],
            "saves": gw_df["saves"],
            "bonus": gw_df["bonus"],
            "bps": gw_df["bps"],
            "influence": gw_df["influence"],
            "creativity": gw_df["creativity"],
            "threat": gw_df["threat"],
            "ict_index": gw_df["ict_index"],
            "total_points": gw_df["total_points"],
            "goals_conceded": gw_df["goals_conceded"],
            "yellow_cards": gw_df["yellow_cards"],
            "season": season
        })
        gw_dfs.append(gw_out)
    return pd.concat(gw_dfs, ignore_index=True) if gw_dfs else pd.DataFrame()

def download_current_season():
    """Download players, teams, fixtures, and GWs for current season from API."""
    players_url = f"{BASE_URL}bootstrap-static/"
    r = requests.get(players_url)
    r.raise_for_status()
    data = r.json()

    teams_df = pd.DataFrame(data["teams"])[["id", "name", "short_name", "strength"]]
    teams_df = teams_df.rename(columns={
        "id": "team_id",
        "name": "team_name",
        "short_name": "team_short_name"
    })

    players_df = pd.DataFrame(data["elements"])[[
        "id", "web_name", "element_type", "team", "now_cost", "status",
        "total_points", "minutes", "goals_scored", "assists",
        "clean_sheets", "saves", "bonus", "bps", "influence",
        "creativity", "threat", "ict_index"
    ]]
    players_df = players_df.rename(columns={
        "id": "player_id",
        "web_name": "name",
        "element_type": "position_id",
        "team": "team_id",
        "now_cost": "value"
    })
    players_df["value"] = players_df["value"] / 10
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    players_df["position"] = players_df["position_id"].map(pos_map)

    fixtures_url = f"{BASE_URL}fixtures/"
    r = requests.get(fixtures_url)
    r.raise_for_status()
    fixtures_df = pd.DataFrame(r.json())[[
        "id", "event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"
    ]]
    fixtures_df = fixtures_df.rename(columns={
        "id": "fixture_id",
        "event": "gameweek",
        "team_h": "home_team_id",
        "team_a": "away_team_id",
        "team_h_difficulty": "home_fdr",
        "team_a_difficulty": "away_fdr"
    })

    # Current season doesn't have past GW data for all weeks
    gw_df = pd.DataFrame()

    return players_df, teams_df, fixtures_df, gw_df

def main():
    all_players, all_teams, all_fixtures, all_gws = [], [], [], []

    for season in SEASONS:
        print(f"📥 Downloading {season}...")
        if season != SEASONS[-1]:  # Past seasons
            p_df, t_df, f_df = download_player_team_fixture_archive(season)
            gw_df = download_gameweeks_archive(season)
        else:  # Current season
            p_df, t_df, f_df, gw_df = download_current_season()
            p_df["season"] = season
            t_df["season"] = season
            f_df["season"] = season

        if p_df is not None: all_players.append(p_df)
        if t_df is not None: all_teams.append(t_df)
        if f_df is not None: all_fixtures.append(f_df)
        if gw_df is not None and not gw_df.empty: all_gws.append(gw_df)

    pd.concat(all_players, ignore_index=True).to_csv("data/raw/fpl_players.csv", index=False)
    pd.concat(all_teams, ignore_index=True).to_csv("data/raw/fpl_teams.csv", index=False)
    pd.concat(all_fixtures, ignore_index=True).to_csv("data/raw/fpl_fixtures.csv", index=False)
    pd.concat(all_gws, ignore_index=True).to_csv("data/raw/fpl_gameweek_data.csv", index=False)
    print("✅ All raw data saved in data/raw/")

if __name__ == "__main__":
    main()
