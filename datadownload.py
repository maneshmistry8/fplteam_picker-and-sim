import os
import json
import requests
from pathlib import Path

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
    response = requests.get(url)
    response.raise_for_status()

    filename = f"{endpoint_name}_{param}.json" if param else f"{endpoint_name}.json"
    filepath = DATA_DIR / filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w") as f:
        json.dump(response.json(), f, indent=2)

    print(f"Saved {filename}")

if __name__ == "__main__":
    # Fetch main data and fixtures
    fetch_and_save("bootstrap")
    fetch_and_save("fixtures")

    # Example: Fetch GW1 live data and one player summary
    fetch_and_save("event_live", 1)
    fetch_and_save("player_summary", 1)
