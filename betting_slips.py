import logging
import pandas as pd
import numpy as np
from predictapi import predict_outcomes, format_predictions

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_betting_slips(model_bundle, raw_data):
    """Generate betting slips from predictions."""
    try:
        # Get predictions
        predictions = predict_outcomes(model_bundle, raw_data)
        formatted = format_predictions(predictions)
        
        betting_slips = []
        for _, row in formatted.iterrows():
            try:
                fixture_id = row["Fixture ID"]
                fixture_data = raw_data[raw_data["Fixture ID"] == fixture_id].iloc[0]
                
                slip = {
                    "fixture_id": fixture_id,
                    "home_team": fixture_data["Home Team"],
                    "away_team": fixture_data["Away Team"],
                    "date": fixture_data.get("Date", "N/A"),
                    "predicted_winner": str(row.get("Result", "Unknown")),
                    "predicted_total_goals": round(float(row.get("Total Goals", 0)), 2),
                    "predicted_over_under": str(row.get("Over/Under 2.5", "Unknown")),
                    "predicted_btts": str(row.get("Both Teams to Score", "Unknown")),
                    "winner_confidence": round(float(row.get("winner_confidence", 0)) * 100, 1),
                    "recommended_bet": calculate_recommended_bet(row)
                }
                betting_slips.append(slip)
                logging.info(f"Created slip for {slip['home_team']} vs {slip['away_team']}")
                logging.info(f"Predictions: {dict(row)}")
                
            except Exception as e:
                logging.error(f"Error creating slip: {e}")
                continue

        return betting_slips

    except Exception as e:
        logging.error(f"Error in betting slips: {e}")
        raise ValueError(f"Betting slip generation failed: {e}")


def calculate_recommended_bet(prediction_row):
    """Calculate recommended bet based on prediction values."""
    try:
        recommendations = []
        
        # Match Result
        result = str(prediction_row.get("Result", "Unknown"))
        if result not in ["Unknown", "Draw"]:
            recommendations.append(result)
        
        # Goals recommendations
        total_goals = float(prediction_row.get("Total Goals", 0))
        if total_goals > 0:  # Only add if we have a meaningful prediction
            if total_goals > 2.5:
                recommendations.append("Over 2.5 Goals")
            elif total_goals < 2.5:
                recommendations.append("Under 2.5 Goals")
            
        # BTTS recommendation
        btts = str(prediction_row.get("Both Teams to Score", "Unknown"))
        if btts == "Yes":
            recommendations.append("Both Teams to Score")
            
        if not recommendations:
            return "No clear betting opportunity"
            
        return " & ".join(recommendations)
        
    except Exception as e:
        logging.error(f"Error calculating bet: {e}")
        logging.error(f"Prediction row data: {prediction_row}")
        return "No clear recommendation"
