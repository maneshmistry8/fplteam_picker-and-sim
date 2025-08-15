import os
import json
import requests
from pathlib import Path
from time import sleep

BASE_URL = "https://fantasy.premierleague.com/api/"
DATA_DIR = Path("data/raw")
ENDPOINTS = {
    "bootstrap": "bootstrap-static/",
    "fixtures": "fixtures/",
    "event_live": lambda gw: f"event/{gw}/live/",
    "player_summary": lambda pid: f"element-summary/{pid}/"
}

def fetch_and_save(endpoint_name, param=None):
    if endpoint_name not in ENDPOINTS:
        raise ValueError("Invalid endpoint")

    endpoint = ENDPOINTS[endpoint_name](param) if callable(ENDPOINTS[endpoint_name]) else ENDPOINTS[endpoint_name]
    url = BASE_URL + endpoint
    print(f"Fetching: {url}")
    response = requests.get(url)
    response.raise_for_status()

    filename = f"{endpoint_name}_{param}.json" if param else f"{endpoint_name}.json"
    filepath = DATA_DIR / filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w") as f:
        json.dump(response.json(), f, indent=2)

    print(f"Saved {filename}")
    sleep(0.5)  # Avoid API rate-limiting

if __name__ == "__main__":
    # Fetch main data and fixtures
    fetch_and_save("bootstrap")
    fetch_and_save("fixtures")

    # Fetch all gameweek live data (GW1-38)
    for gw in range(1, 39):
        fetch_and_save("event_live", gw)

    # Fetch player summaries for all players in bootstrap
    with open(DATA_DIR / "bootstrap-static.json") as f:
        bootstrap = json.load(f)
    for player in bootstrap["elements"]:
        pid = player["id"]
        fetch_and_save("player_summary", pid)
