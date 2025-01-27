import logging
import pandas as pd
from datetime import datetime, timedelta
from utils import make_api_request, validate_api_response
from constants import LEAGUE_IDS, SEASON
from data_processing import (
    fetch_and_enrich_fixtures,
    enrich_fixture_with_statistics,
    enrich_fixture_with_predictions
)
from api_client import fetch_today_fixtures
from model import load_model  # Correct import
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def collect_historical_matches(days_back=90):  # Increased from 30 to 90 days
    """Collect historical match data for the past N days."""
    try:
        all_matches = []
        today = datetime.now()
        progress_bar = tqdm(range(days_back), desc="Collecting matches")
        
        for i in progress_bar:
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            progress_bar.set_description(f"Collecting matches for {date}")
            
            for league_id in LEAGUE_IDS.values():
                response = make_api_request("fixtures", {
                    "league": league_id,
                    "season": SEASON,
                    "date": date,
                    "status": "FT"  # Only completed matches
                })
                
                matches = validate_api_response(response)
                if matches:
                    # Enrich each match with statistics and predictions
                    for match in matches:
                        try:
                            fixture_id = match["fixture"]["id"]
                            home_team_id = match["teams"]["home"]["id"]
                            away_team_id = match["teams"]["away"]["id"]
                            
                            # Get match details
                            stats = enrich_fixture_with_statistics(fixture_id, home_team_id, away_team_id, league_id)
                            match.update(stats)
                            
                            # Add actual result data
                            match["actual_result"] = get_match_result(match)
                            match["actual_goals"] = get_match_goals(match)
                            
                            all_matches.append(match)
                            
                        except Exception as e:
                            logging.error(f"Error processing match {fixture_id}: {e}")
                            continue
                            
                # Respect API rate limits
                time.sleep(0.5)  # Small delay between requests
            
        logging.info(f"Collected {len(all_matches)} historical matches")
        return all_matches
        
    except Exception as e:
        logging.error(f"Error collecting historical matches: {e}")
        return []

def collect_season_matches():
    """Collect all completed matches from the current season."""
    try:
        all_matches = []
        total_matches = 0
        progress_bar = tqdm(LEAGUE_IDS.items(), desc="Collecting season matches")
        
        for league_name, league_id in progress_bar:
            progress_bar.set_description(f"Collecting {league_name} matches")
            
            # First get all matches for the league
            response = make_api_request("fixtures", {
                "season": SEASON,
                "league": league_id,
                "status": "FT",  # Only completed matches
                "from": "2023-07-01",  # Start of season
                "to": datetime.now().strftime("%Y-%m-%d")  # Today
            })
            
            matches = validate_api_response(response)
            if matches:
                for match in matches:
                    try:
                        home_team = match["teams"]["home"]
                        away_team = match["teams"]["away"]
                        
                        # Get detailed team statistics
                        home_stats = get_team_statistics(home_team["id"], league_id)
                        away_stats = get_team_statistics(away_team["id"], league_id)
                        
                        if home_stats and away_stats:
                            match_data = {
                                "fixture": match["fixture"],
                                "league": match["league"],
                                "teams": match["teams"],
                                "goals": match["goals"],
                                "score": match.get("score", {}),
                                "league_name": league_name,
                                "league_id": league_id,
                                "statistics": {
                                    "home_avg_goals": home_stats["goals"]["avg_scored"],
                                    "away_avg_goals": away_stats["goals"]["avg_scored"],
                                    "home_avg_conceded": home_stats["goals"]["avg_conceded"],
                                    "away_avg_conceded": away_stats["goals"]["avg_conceded"],
                                    "home_avg_corners": home_stats["corners"]["avg_per_game"],
                                    "away_avg_corners": away_stats["corners"]["avg_per_game"],
                                    "home_clean_sheets": home_stats["clean_sheets"],
                                    "away_clean_sheets": away_stats["clean_sheets"],
                                    "home_win_rate": home_stats["win_rate"],
                                    "away_win_rate": away_stats["win_rate"]
                                },
                                "actual_result": get_match_result(match),
                                "actual_goals": match["goals"]["home"] + match["goals"]["away"],
                                "actual_btts": "Yes" if match["goals"]["home"] > 0 and match["goals"]["away"] > 0 else "No",
                                "actual_over_2_5": "Over" if (match["goals"]["home"] + match["goals"]["away"]) > 2.5 else "Under"
                            }
                            
                            all_matches.append(match_data)
                            total_matches += 1
                            
                            if total_matches % 10 == 0:
                                logging.info(f"Collected {total_matches} total matches")
                            
                    except Exception as e:
                        logging.error(f"Error processing match: {e}")
                        continue
                        
                time.sleep(1)  # Respect API rate limits
                
        logging.info(f"\nCollection Summary:")
        logging.info(f"Total matches collected: {total_matches}")
        return all_matches
        
    except Exception as e:
        logging.error(f"Error collecting season matches: {e}")
        return []

def get_team_statistics(team_id, league_id):
    """Get detailed team statistics from the API."""
    try:
        response = make_api_request("teams/statistics", {
            "team": team_id,
            "league": league_id,
            "season": SEASON
        })
        
        stats = validate_api_response(response)
        if not stats:
            return None
            
        return {
            "goals": {
                "scored": stats.get("goals", {}).get("for", {}).get("total", {}).get("total", 0),
                "conceded": stats.get("goals", {}).get("against", {}).get("total", {}).get("total", 0),
                "avg_scored": stats.get("goals", {}).get("for", {}).get("average", {}).get("total", 0),
                "avg_conceded": stats.get("goals", {}).get("against", {}).get("average", {}).get("total", 0)
            },
            "corners": {
                "total": stats.get("corners", {}).get("total", {}).get("total", 0),
                "avg_per_game": stats.get("corners", {}).get("average", {}).get("total", 0)
            },
            "form": stats.get("form", ""),
            "clean_sheets": stats.get("clean_sheets", {}).get("total", 0),
            "failed_to_score": stats.get("failed_to_score", {}).get("total", 0),
            "win_rate": stats.get("fixtures", {}).get("wins", {}).get("total", 0) / 
                       stats.get("fixtures", {}).get("played", {}).get("total", 1)
        }
    except Exception as e:
        logging.error(f"Error fetching team statistics: {e}")
        return None

def get_match_result(match):
    """Extract actual match result."""
    home_goals = match["goals"]["home"]
    away_goals = match["goals"]["away"]
    
    if home_goals > away_goals:
        return "Home Win"
    elif away_goals > home_goals:
        return "Away Win"
    return "Draw"

def get_match_goals(match):
    """Extract actual match goals."""
    return match["goals"]["home"] + match["goals"]["away"]

def process_and_save_matches(matches):
    """Process matches into training data format."""
    try:
        df = pd.DataFrame(matches)
        
        # Add derived features
        df["Total Goals"] = df.apply(lambda x: x["goals"]["home"] + x["goals"]["away"], axis=1)
        df["Over/Under 2.5"] = df["Total Goals"].apply(lambda x: "Over" if x > 2.5 else "Under")
        df["Over/Under 1.5"] = df["Total Goals"].apply(lambda x: "Over" if x > 1.5 else "Under")
        df["Both Teams to Score"] = df.apply(lambda x: "Yes" if x["goals"]["home"] > 0 and x["goals"]["away"] > 0 else "No", axis=1)
        
        # Save full dataset
        df.to_csv("historical_matches.csv", index=False)
        logging.info(f"Saved {len(df)} matches to historical_matches.csv")
        
        # Save training subset
        training_columns = [
            "Fixture ID", "Home Team", "Away Team", "League ID",
            "Result", "Total Goals", "Over/Under 2.5", "Over/Under 1.5",
            "Both Teams to Score",
            "home_avg_goals", "away_avg_goals",
            "home_avg_conceded", "away_avg_conceded",
            "home_avg_corners", "away_avg_corners",
            "home_clean_sheets", "away_clean_sheets",
            "home_win_rate", "away_win_rate"
        ]
        training_df = df[training_columns]
        training_df.to_csv("training_data.csv", index=False)
        logging.info(f"Saved training data with {len(training_df)} matches")
        
        return df
        
    except Exception as e:
        logging.error(f"Error processing matches: {e}")
        return pd.DataFrame()

def validate_match_data(match_data):
    """Validate individual match data quality."""
    try:
        # Check for required fields
        required_fields = {
            "fixture": ["id", "date"],
            "teams": ["home", "away"],
            "goals": ["home", "away"],
            "league_id": None,
            "league_name": None,
            "statistics": None
        }

        for field, subfields in required_fields.items():
            if field not in match_data:
                return False
            if subfields:
                if not all(sf in match_data[field] for sf in subfields):
                    return False

        # Validate numeric values
        if not isinstance(match_data["actual_goals"], (int, float)):
            return False
        if match_data["actual_goals"] < 0 or match_data["actual_goals"] > 15:
            return False

        return True
    except Exception:
        return False

def prepare_training_data(matches):
    """Process matches into training format with validation."""
    try:
        if not matches:
            raise ValueError("No matches provided")
            
        # Filter valid matches
        valid_matches = [m for m in matches if validate_match_data(m)]
        logging.info(f"Valid matches: {len(valid_matches)} out of {len(matches)}")
        
        if len(valid_matches) < 100:
            raise ValueError(f"Insufficient valid matches: {len(valid_matches)} < 100 required")
            
        df = pd.DataFrame(valid_matches)
        
        # Add stats columns with proper defaults
        stats_columns = {
            "Home Avg Goals": "home_avg_goals",
            "Away Avg Goals": "away_avg_goals",
            "Home Avg Conceded": "home_avg_conceded",
            "Away Avg Conceded": "away_avg_conceded",
            "Home Avg Corners": "home_avg_corners",
            "Away Avg Corners": "away_avg_corners",
            "Home Clean Sheets": "home_clean_sheets",
            "Away Clean Sheets": "away_clean_sheets",
            "Home Win Rate": "home_win_rate",
            "Away Win Rate": "away_win_rate"
        }
        
        for col, stat_key in stats_columns.items():
            df[col] = df["statistics"].apply(lambda x: float(x.get(stat_key, 0)))

        # Create training dataset
        training_data = pd.DataFrame({
            "Fixture ID": df["fixture"].apply(lambda x: x["id"]),
            "Date": df["fixture"].apply(lambda x: x["date"]),
            "League ID": df["league_id"],
            "League Name": df["league_name"],
            "Home Team": df["teams"].apply(lambda x: x["home"]["name"]),
            "Away Team": df["teams"].apply(lambda x: x["away"]["name"]),
            "Result": df["actual_result"],
            "Total Goals": df["actual_goals"],
            "Over/Under 2.5": df["actual_over_2_5"],
            "Over/Under 1.5": df["actual_goals"].apply(lambda x: "Over" if x > 1.5 else "Under"),
            "Over/Under 0.5": df["actual_goals"].apply(lambda x: "Over" if x > 0.5 else "Under"),
            "Over/Under 3.5": df["actual_goals"].apply(lambda x: "Over" if x > 3.5 else "Under"),
            "Both Teams to Score": df["actual_btts"],
            "Home Avg Goals": df["Home Avg Goals"],
            "Away Avg Goals": df["Away Avg Goals"],
            "Home Avg Conceded": df["Home Avg Conceded"],
            "Away Avg Conceded": df["Away Avg Conceded"],
            "Home Avg Corners": df["Home Avg Corners"],
            "Away Avg Corners": df["Away Avg Corners"],
            "Home Clean Sheets": df["Home Clean Sheets"],
            "Away Clean Sheets": df["Away Clean Sheets"],
            "Home Win Rate": df["Home Win Rate"],
            "Away Win Rate": df["Away Win Rate"]
        })

        # Validate distributions
        result_dist = training_data["Result"].value_counts(normalize=True)
        if max(result_dist) > 0.5:  # No result should be >50% of data
            logging.warning(f"Unbalanced results: {result_dist.to_dict()}")
            
        goals_mean = training_data["Total Goals"].mean()
        if not 2.0 <= goals_mean <= 3.5:  # Expected range for average goals
            logging.warning(f"Unusual average goals: {goals_mean:.2f}")
            
        # Log detailed statistics
        logging.info("\nDetailed Data Summary:")
        logging.info(f"\nResults by League:")
        for league in training_data["League Name"].unique():
            league_data = training_data[training_data["League Name"] == league]
            logging.info(f"\n{league}:")
            logging.info(f"Matches: {len(league_data)}")
            logging.info(f"Avg Goals: {league_data['Total Goals'].mean():.2f}")
            logging.info(f"Results: {league_data['Result'].value_counts().to_dict()}")

        # Save dataset
        training_data.to_csv("season_training_data.csv", index=False)
        logging.info(f"Saved {len(training_data)} matches to training dataset")
        
        return training_data
        
    except Exception as e:
        logging.error(f"Error preparing training data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    matches = collect_season_matches()
    if matches:
        training_data = prepare_training_data(matches)
        print(f"Successfully collected {len(training_data)} matches for training")
