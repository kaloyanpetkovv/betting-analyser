import logging
import time
from typing import Dict, List
from utils import make_api_request, validate_api_response
from constants import LEAGUE_IDS

def fetch_today_fixtures() -> List[Dict]:
    """Fetch today's fixtures from the API."""
    try:
        today = time.strftime("%Y-%m-%d")
        response = make_api_request("fixtures", {
            "date": today,
            "status": "NS"  # Not Started matches
        })
        
        fixtures = validate_api_response(response)
        if not fixtures:
            logging.info("No fixtures found for today")
            return []
            
        # Filter fixtures for supported leagues
        supported_fixtures = [
            fixture for fixture in fixtures 
            if fixture.get("league", {}).get("id") in LEAGUE_IDS.values()
        ]
        
        logging.info(f"Found {len(supported_fixtures)} fixtures for today")
        return supported_fixtures
        
    except Exception as e:
        logging.error(f"Error fetching fixtures: {e}")
        return []
