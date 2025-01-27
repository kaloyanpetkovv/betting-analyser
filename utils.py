import time
import requests
import logging
import pandas as pd
import numpy as np
from functools import wraps
from constants import API_BASE_URL, API_HEADERS, LEAGUE_IDS, SEASON, REQUESTS_LIMIT, TIME_INTERVAL

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def rate_limiter(func):
    """Decorator to limit the rate of API requests."""
    last_call = [time.time()]
    call_count = [0]

    @wraps(func)
    def wrapper(*args, **kwargs):
        current_time = time.time()
        elapsed_time = current_time - last_call[0]

        if elapsed_time >= TIME_INTERVAL:
            call_count[0] = 0
            last_call[0] = current_time

        if call_count[0] >= REQUESTS_LIMIT:
            sleep_time = TIME_INTERVAL - elapsed_time
            logging.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)

        call_count[0] += 1
        return func(*args, **kwargs)

    return wrapper

@rate_limiter
def make_api_request(endpoint, params=None):
    """Makes an API request to the API-Football endpoint."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.get(url, headers=API_HEADERS, params=params)
        response.raise_for_status()

        # Log rate limit info
        remaining = response.headers.get("x-ratelimit-remaining", "Unknown")
        reset = response.headers.get("x-ratelimit-reset", "Unknown")
        logging.info(f"API Request: {endpoint} - Remaining: {remaining}, Reset: {reset}")

        data = response.json()
        if data.get("errors"):
            logging.error(f"API Error: {data['errors']}")
            return {"error": data['errors']}
            
        return data

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return {"error": str(e)}

def validate_api_response(response, key="response"):
    """Validate the API response structure."""
    if not response or "error" in response:
        logging.error("Invalid API response or error encountered.")
        return None
    if key not in response:
        logging.error(f"Expected key '{key}' not found in response.")
        return None
    return response.get(key, [])

def fetch_team_recent_statistics(team_id, league_id):
    """Fetch team statistics including corners."""
    try:
        response = make_api_request("teams/statistics", {
            "team": team_id,
            "league": league_id,
            "season": SEASON
        })
        
        stats = validate_api_response(response)
        if not stats:
            return {}

        # Extract corner statistics
        corners = stats.get("fixtures", {}).get("corners", {})
        corners_for_total = sum(int(c) for c in corners.get("for", {}).values())
        corners_against_total = sum(int(c) for c in corners.get("against", {}).values())
        total_matches = stats.get("fixtures", {}).get("played", {}).get("total", 1)

        return {
            "average_goals_scored": float(stats.get("goals", {}).get("for", {}).get("average", {}).get("total", 0)),
            "average_goals_conceded": float(stats.get("goals", {}).get("against", {}).get("average", {}).get("total", 0)),
            "average_corners_for": corners_for_total / total_matches if total_matches > 0 else 0,
            "average_corners_against": corners_against_total / total_matches if total_matches > 0 else 0,
            "win_draw_percentage": (stats.get("fixtures", {}).get("wins", {}).get("total", 0) + 
                                  stats.get("fixtures", {}).get("draws", {}).get("total", 0)) / total_matches if total_matches > 0 else 0,
        }
    except Exception as e:
        logging.error(f"Error fetching statistics for team {team_id}: {e}")
        return {}

def validate_training_data(data: pd.DataFrame) -> bool:
    """Validate that all required features are present in training data."""
    required_columns = [
        # Features
        "Home Team", "Away Team", "League ID",
        "Home Avg Goals", "Away Avg Goals",
        "Home Avg Conceded", "Away Avg Conceded",
        "Home Avg Corners", "Away Avg Corners",
        "Combined Avg Corners",
        
        # Targets
        "Result", "Total Goals",
        "Over/Under 0.5", "Over/Under 1.5", 
        "Over/Under 2.5", "Over/Under 3.5",
        "X1", "X2", 
        "Over/Under 9.5 Corners",
        "Both Teams to Score"
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logging.error(f"Missing required columns in training data: {missing_columns}")
        return False
        
    # Check for null values
    null_columns = data.columns[data.isnull().any()].tolist()
    if null_columns:
        logging.error(f"Columns containing null values: {null_columns}")
        return False
        
    return True
