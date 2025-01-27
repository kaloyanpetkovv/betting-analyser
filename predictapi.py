import logging
import numpy as np
import pandas as pd
from utils import make_api_request, validate_api_response
from data_processing import fetch_today_fixtures, fetch_and_enrich_fixtures
from model import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_match_predictions(fixture_id):
    """Fetch predictions from the API for a specific fixture."""
    try:
        response = make_api_request(f"predictions", {"fixture": fixture_id})
        predictions = validate_api_response(response)
        
        if not predictions or len(predictions) == 0:
            return {}
            
        pred_data = predictions[0]
        predictions_data = pred_data.get("predictions", {})
        match_data = pred_data.get("match", {})
        
        return {
            "winner": predictions_data.get("winner", {}).get("name"),
            "winner_comment": predictions_data.get("winner", {}).get("comment"),
            "win_or_draw": predictions_data.get("win_or_draw"),
            "goals": {
                "home": predictions_data.get("goals", {}).get("home"),
                "away": predictions_data.get("goals", {}).get("away"),
                "total": predictions_data.get("goals", {}).get("total")
            },
            "over_2_5": predictions_data.get("goals", {}).get("over", False),
            "under_2_5": predictions_data.get("goals", {}).get("under", False),
            "btts": predictions_data.get("btts", False),
            "score": predictions_data.get("correct_score"),
            "winning_percent": predictions_data.get("winning_percent"),
            "form": {
                "home": match_data.get("teams", {}).get("home", {}).get("last_5", {}).get("form"),
                "away": match_data.get("teams", {}).get("away", {}).get("last_5", {}).get("form")
            }
        }
        
    except Exception as e:
        logging.error(f"Error fetching predictions for fixture {fixture_id}: {e}")
        return {}

def get_match_statistics(fixture_id):
    """Fetch match statistics from the API."""
    try:
        response = make_api_request(f"fixtures/statistics", {"fixture": fixture_id})
        stats = validate_api_response(response)
        
        if not stats:
            return {}
            
        home_stats = next((s for s in stats if s.get("team", {}).get("location") == "home"), {})
        away_stats = next((s for s in stats if s.get("team", {}).get("location") == "away"), {})
        
        return {
            "home": {
                "shots_on_goal": extract_stat_value(home_stats.get("statistics", []), "Shots on Goal"),
                "shots_off_goal": extract_stat_value(home_stats.get("statistics", []), "Shots off Goal"),
                "total_shots": extract_stat_value(home_stats.get("statistics", []), "Total Shots"),
                "corners": extract_stat_value(home_stats.get("statistics", []), "Corner Kicks"),
                "possession": extract_stat_value(home_stats.get("statistics", []), "Ball Possession")
            },
            "away": {
                "shots_on_goal": extract_stat_value(away_stats.get("statistics", []), "Shots on Goal"),
                "shots_off_goal": extract_stat_value(away_stats.get("statistics", []), "Shots off Goal"),
                "total_shots": extract_stat_value(away_stats.get("statistics", []), "Total Shots"),
                "corners": extract_stat_value(away_stats.get("statistics", []), "Corner Kicks"),
                "possession": extract_stat_value(away_stats.get("statistics", []), "Ball Possession")
            }
        }
        
    except Exception as e:
        logging.error(f"Error fetching statistics for fixture {fixture_id}: {e}")
        return {}

def extract_stat_value(statistics, stat_name):
    """Extract value from statistics array."""
    try:
        stat = next((s for s in statistics if s.get("type") == stat_name), {})
        value = stat.get("value")
        if isinstance(value, str) and value.endswith('%'):
            return float(value.rstrip('%'))
        return float(value) if value is not None else 0
    except Exception:
        return 0

def predict_outcomes(model_bundle, raw_data):
    """Predict outcomes using model bundle components."""
    try:
        classifier = model_bundle['classifier']
        regressor = model_bundle['regressor']
        encoders = model_bundle['encoders']
        feature_columns = model_bundle['feature_columns']
        
        # Preprocess new data
        processed_data = raw_data.copy()
        
        # Convert all numeric columns
        numeric_columns = [col for col in feature_columns if col not in ["Home Team", "Away Team"]]
        for col in numeric_columns:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0)
        
        # Encode categorical columns
        for col in ["Home Team", "Away Team"]:
            if col in encoders and col in processed_data:
                processed_data[col] = encoders[col].transform(processed_data[col].astype(str))

        # Ensure all feature columns exist
        missing_features = set(feature_columns) - set(processed_data.columns)
        for col in missing_features:
            processed_data[col] = 0

        # Select features in correct order
        X = processed_data[feature_columns]

        # Debugging logs
        logging.info(f"Processed data for prediction: \n{processed_data.head()}")
        logging.info(f"Features used for prediction: {feature_columns}")
        logging.info(f"Data shape: {X.shape}")

        # Get predictions with probabilities for each class
        classification_predictions = classifier.predict(X)
        probabilities = np.array([est.predict_proba(X) for est in classifier.estimators_])
        regression_predictions = regressor.predict(X)

        # Get confidence scores from probabilities
        confidences = np.max(probabilities, axis=2)  # Get max probability for each prediction
        
        predictions_dict = {
            "Fixture ID": raw_data["Fixture ID"].values
        }
        
        # Add predictions with confidence scores
        for i, target in enumerate(model_bundle['class_targets']):
            predictions_dict[target] = classification_predictions[:, i]
            predictions_dict[f"{target}_confidence"] = confidences[i]
        
        for i, target in enumerate(model_bundle['reg_targets']):
            predictions_dict[target] = regression_predictions[:, i]

        logging.info(f"Generated predictions for {len(raw_data)} fixtures")
        logging.info(f"Predictions: {predictions_dict}")
        return pd.DataFrame(predictions_dict)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        logging.error(f"Available features: {feature_columns}")
        logging.error(f"Data columns: {raw_data.columns}")
        raise ValueError(f"Error during prediction: {e}")

def predict_outcomes(model_bundle, raw_data):
    """Predict outcomes using model bundle components."""
    try:
        classifier = model_bundle['classifier']
        regressor = model_bundle['regressor']
        encoders = model_bundle['encoders']
        feature_columns = model_bundle['feature_columns']

        # Prepare data for prediction
        processed_data = raw_data.copy()

        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in processed_data.columns:
                processed_data[col] = 0

        # Encode categorical columns
        for col in ["Home Team", "Away Team"]:
            if col in encoders and col in processed_data.columns:
                processed_data[col] = encoders[col].transform(processed_data[col].astype(str))

        # Convert to numeric
        for col in feature_columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0)

        # Align data to feature columns
        X = processed_data[feature_columns]

        # Predictions
        classification_preds = classifier.predict(X)
        regression_preds = regressor.predict(X)

        # Format predictions for Over/Under and BTTS
        over_under_preds = (regression_preds[:, 0] > 2.5).astype(int)  # Over/Under 2.5
        btts_preds = (regression_preds[:, 1] > 0.5).astype(int)  # Both Teams to Score

        # Build output DataFrame
        predictions_dict = {
            "Fixture ID": raw_data["Fixture ID"].values,
            "Result": classification_preds[:, 0],  # Assuming 1st target is the match result
            "Over/Under 2.5": over_under_preds,
            "Both Teams to Score": btts_preds,
            "Total Goals": regression_preds[:, 0]  # Assuming 1st regression target is total goals
        }
        return pd.DataFrame(predictions_dict)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise


def get_todays_betting_slips():
    """Fetch today's betting slips."""
    try:
        fixtures = fetch_today_fixtures()
        if not fixtures:
            logging.warning("No fixtures available for today.")
            return []

        enriched_fixtures = fetch_and_enrich_fixtures()
        if enriched_fixtures.empty:
            logging.warning("No enriched fixtures available for today.")
            return []

        model_bundle = load_model("trained_model.pkl")
        predictions = predict_outcomes(model_bundle, enriched_fixtures)
        formatted_predictions = format_predictions(predictions)

        return formatted_predictions

    except Exception as e:
        logging.error(f"Error fetching today's betting slips: {e}")
        return []
