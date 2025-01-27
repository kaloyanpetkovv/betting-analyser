import logging
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from utils import make_api_request, validate_api_response, fetch_team_recent_statistics
from model import load_model
from constants import API_BASE_URL, API_HEADERS, LEAGUE_IDS, SEASON
from api_client import fetch_today_fixtures

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

__all__ = ['fetch_today_fixtures', 'enrich_fixture_with_statistics', 'enrich_fixture_with_predictions']

def get_api_date():
    """
    Get the current date in UTC for querying the API.
    Returns:
        str: Current date in 'YYYY-MM-DD' format.
    """
    utc_now = datetime.now(pytz.UTC)
    return utc_now.strftime("%Y-%m-%d")

def enrich_fixture_with_statistics(fixture_id: int, home_team_id: int, away_team_id: int, league_id: int) -> Dict[str, Any]:
    """Enrich fixture data with team statistics."""
    try:
        # Get team statistics
        home_stats = fetch_team_recent_statistics(home_team_id, league_id)
        away_stats = fetch_team_recent_statistics(away_team_id, league_id)

        if not home_stats or not away_stats:
            logging.warning(f"Missing statistics for fixture {fixture_id}")
            return {}

        # Calculate combined metrics
        combined_avg_goals = (home_stats.get("average_goals_scored", 0) + away_stats.get("average_goals_scored", 0)) / 2

        return {
            "Home Avg Goals": home_stats.get("average_goals_scored", 0),
            "Away Avg Goals": away_stats.get("average_goals_scored", 0),
            "Home Avg Conceded": home_stats.get("average_goals_conceded", 0),
            "Away Avg Conceded": away_stats.get("average_goals_conceded", 0),
            "Home Avg Corners": home_stats.get("average_corners_for", 0),
            "Away Avg Corners": away_stats.get("average_corners_for", 0),
            "Combined Avg Goals": combined_avg_goals,
            "Home Win Draw %": home_stats.get("win_draw_percentage", 0),
            "Away Win Draw %": away_stats.get("win_draw_percentage", 0)
        }

    except Exception as e:
        logging.error(f"Error enriching fixture {fixture_id} with statistics: {e}")
        return {}

def enrich_fixture_with_predictions(fixture_id: int) -> Dict[str, Any]:
    """Fix the predictions structure."""
    try:
        response = make_api_request("predictions", {"fixture": fixture_id})
        predictions = validate_api_response(response)
        
        if not predictions or len(predictions) == 0:
            return {}
            
        pred_data = predictions[0]
        return {
            "Result": pred_data.get("predictions", {}).get("winner", {}).get("name", "Unknown"),
            "Over/Under 2.5": "Over" if pred_data.get("predictions", {}).get("goals", {}).get("total", 0) > 2.5 else "Under",
            "Over/Under 1.5": "Over" if pred_data.get("predictions", {}).get("goals", {}).get("total", 0) > 1.5 else "Under",
            "Over/Under 0.5": "Over" if pred_data.get("predictions", {}).get("goals", {}).get("total", 0) > 0.5 else "Under",
            "Over/Under 3.5": "Over" if pred_data.get("predictions", {}).get("goals", {}).get("total", 0) > 3.5 else "Under",
            "Both Teams to Score": pred_data.get("predictions", {}).get("btts", "No"),
            "Total Goals": float(pred_data.get("predictions", {}).get("goals", {}).get("total", 0))
        }
        
    except Exception as e:
        logging.error(f"Error enriching fixture {fixture_id} with predictions: {e}")
        return {}

def fetch_and_enrich_fixtures() -> pd.DataFrame:
    """Fetch and enrich fixtures with statistics and predictions."""
    try:
        fixtures = fetch_today_fixtures()
        if not fixtures:
            logging.warning("No fixtures found to enrich")
            return pd.DataFrame()
            
        enriched_data = []
        for fixture in fixtures:
            fixture_id = fixture["fixture"]["id"]
            home_team_id = fixture["teams"]["home"]["id"]
            away_team_id = fixture["teams"]["away"]["id"]
            league_id = fixture["league"]["id"]
            
            # Get enrichment data
            stats = enrich_fixture_with_statistics(fixture_id, home_team_id, away_team_id, league_id)
            preds = enrich_fixture_with_predictions(fixture_id)
            
            # Combine fixture data with enrichments
            enriched_data.append({
                "Fixture ID": fixture_id,
                "Home Team": fixture["teams"]["home"]["name"],
                "Away Team": fixture["teams"]["away"]["name"],
                "League ID": league_id,
                "Date": fixture["fixture"]["date"],
                **stats,
                **preds
            })
            
        return pd.DataFrame(enriched_data)
        
    except Exception as e:
        logging.error(f"Error in fetch_and_enrich_fixtures: {e}")
        return pd.DataFrame()

def get_fixture_features(fixture_id: int):
    """
    Extract feature vector for a specific fixture.
    Args:
        fixture_id (int): The ID of the fixture to get features for
    Returns:
        np.array: Feature vector for model prediction
    """
    try:
        # Get raw fixture data
        response = make_api_request(f"fixtures?id={fixture_id}")
        fixture_data = validate_api_response(response)[0]
        
        # Debugging: Print raw fixture data
        print("Raw Fixture Data:", fixture_data)

        # Transform API response into feature vector
        # Ensure this matches the features used during training
        feature_vector = np.array([
            fixture_data["teams"]["home"]["id"],  # Home Team ID
            fixture_data["teams"]["away"]["id"],  # Away Team ID
            fixture_data["league"]["id"],         # League ID
        ]).reshape(1, -1)

        # Debugging: Print feature vector
        print("Feature Vector:", feature_vector)

        return feature_vector
        
    except Exception as e:
        logging.error(f"Error getting features for fixture {fixture_id}: {e}")
        raise

def process_match_for_betting(fixture_id):
    """Process a single match for betting."""
    try:
        # 1. Get match details
        match_details = make_api_request(f"fixtures/id/{fixture_id}")
        if not match_details:
            return None
            
        # 2. Get live odds
        odds = fetch_live_odds(fixture_id)
        
        # 3. Get predictions
        predictions = enrich_fixture_with_predictions(fixture_id)
        
        # 4. Get statistics
        home_team_id = match_details["teams"]["home"]["id"]
        away_team_id = match_details["teams"]["away"]["id"]
        league_id = match_details["league"]["id"]
        stats = enrich_fixture_with_statistics(fixture_id, home_team_id, away_team_id, league_id)
        
        # 5. Combine all data
        betting_data = {
            "fixture_id": fixture_id,
            "home_team": match_details["teams"]["home"]["name"],
            "away_team": match_details["teams"]["away"]["name"],
            "kickoff": match_details["fixture"]["date"],
            "league": match_details["league"]["name"],
            **predictions,
            **odds,
            **stats
        }
        
        return betting_data
        
    except Exception as e:
        logging.error(f"Error processing match {fixture_id}: {e}")
        return None

def get_todays_betting_slips():
    """Get betting slips for today's matches."""
    try:
        # 1. Get today's fixtures
        fixtures = fetch_today_fixtures()
        if not fixtures:
            return []
            
        # 2. Process each fixture
        betting_slips = []
        for fixture in fixtures:
            fixture_id = fixture["fixture"]["id"]
            bet_data = process_match_for_betting(fixture_id)
            if bet_data:
                betting_slips.append(bet_data)
                
        return betting_slips
        
    except Exception as e:
        logging.error(f"Error generating betting slips: {e}")
        return []


