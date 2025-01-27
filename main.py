import logging
import sys
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional
from api_client import fetch_today_fixtures
from data_processing import (
    fetch_and_enrich_fixtures,
    enrich_fixture_with_statistics,
    enrich_fixture_with_predictions
)
from model import train_model, save_model, load_model  # Correct import
from evaluation import evaluate_predictions
from betting_slips import generate_betting_slips
from collect_training_data import collect_season_matches, prepare_training_data

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(asctime)s - %(message)s")

def display_fixtures(fixtures: list) -> None:
    """
    Display today's fixtures using Streamlit.
    
    Args:
        fixtures (list): List of fixtures.
        
    Raises:
        ValueError: If fixtures is not a list or is empty
    """
    if not isinstance(fixtures, list):
        raise ValueError("Fixtures must be a list")
    if fixtures:
        st.write("### Today's Fixtures")
        fixtures_df = pd.DataFrame([{
            "Fixture ID": fixture["fixture"]["id"],
            "Home Team": fixture["teams"]["home"]["name"],
            "Away Team": fixture["teams"]["away"]["name"],
            "League": fixture["league"]["name"],
            "Date": fixture["fixture"]["date"]
        } for fixture in fixtures])
        st.dataframe(fixtures_df)
    else:
        st.write("No fixtures available for today.")

def train_and_evaluate(data):
    """
    Train and evaluate the prediction model with the provided data.
    Args:
        data (pd.DataFrame): Enriched match data.
    Returns:
        dict: Models and evaluation metrics.
    """
    st.write("### Training/Loading Prediction Model")
    try:
        model_path = "trained_model.pkl"
        
        # Try to load existing model first
        try:
            model_bundle = load_model(model_path)
            st.success("Loaded existing model successfully!")
            logging.info("Using pre-trained model")
            return model_bundle, model_bundle.get("metrics")
        except Exception as e:
            logging.info(f"Could not load existing model ({e}), training new one...")
            
        # Train new model if loading failed
        required_columns = ["Result", "Total Goals", "Over/Under 2.5", "Both Teams to Score"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        # Log dataset details
        logging.info(f"Dataset size: {data.shape}")
        logging.info(f"Dataset columns: {list(data.columns)}")

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Train the models and get results
        result = train_model(data)
        metrics = result["metrics"]

        # Save the trained model
        save_model(result, model_path)
        st.success("Model trained and saved successfully!")

        return result, metrics
    except Exception as e:
        st.error(f"Error during model training/loading: {e}")
        logging.error(f"Model training/loading error: {e}")
        return None, None

def load_or_collect_training_data():
    """Load existing training data or collect new data if needed."""
    try:
        # Try to load existing training data
        try:
            training_data = pd.read_csv("season_training_data.csv")
            logging.info(f"Loaded existing training data with {len(training_data)} matches")
            return training_data
        except FileNotFoundError:
            logging.info("No existing training data found, collecting new data...")

        # Collect new training data
        matches = collect_season_matches()
        if matches:
            training_data = prepare_training_data(matches)
            logging.info(f"Collected new training data with {len(training_data)} matches")
            return training_data
        else:
            raise ValueError("Failed to collect training data")

    except Exception as e:
        logging.error(f"Error loading/collecting training data: {e}")
        return None

def main() -> None:
    """
    Main Streamlit app function.
    
    Handles the main application flow and state management.
    """
    st.title("Football Match Predictions")
    
    # Add data collection controls
    with st.sidebar:
        st.header("Training Data")
        if st.button("Collect New Training Data"):
            with st.spinner("Collecting season data..."):
                training_data = load_or_collect_training_data()
                if training_data is not None:
                    st.success(f"Collected {len(training_data)} matches!")
                else:
                    st.error("Failed to collect training data")
    
    # Load training data
    training_data = None
    try:
        training_data = pd.read_csv("season_training_data.csv")
        st.sidebar.info(f"Using {len(training_data)} matches for training")
    except FileNotFoundError:
        if st.sidebar.button("Initialize Training Data"):
            with st.spinner("Collecting initial training data..."):
                training_data = load_or_collect_training_data()

    if training_data is None:
        st.warning("No training data available. Please collect training data first.")
        return

    # Fetch today's fixtures
    st.write("### Fetching Today's Fixtures...")
    fixtures = fetch_today_fixtures()
    display_fixtures(fixtures)

    if fixtures:
        st.write("### Enriching and Preparing Data...")
        enriched_data = fetch_and_enrich_fixtures()

        if not enriched_data.empty:
            # Train model using season data
            models, metrics = train_and_evaluate(training_data)
            if models and metrics:
                st.write("### Model Metrics")
                st.write("#### Classification Metrics")
                st.json(metrics["classification_metrics"])
                st.write("#### Regression Metrics")
                st.json(metrics["regression_metrics"])
                
                st.write("### Generating Betting Slips...")
                # Load the full model bundle
                model_bundle = models
                
                # Generate predictions using the proper API
                predictions = generate_betting_slips(
                    model_bundle,
                    enriched_data  # Pass raw data to be preprocessed by predict_outcomes
                )
                for slip in predictions:
                    st.markdown(f"---")
                    st.subheader(f"📊 {slip['home_team']} vs {slip['away_team']}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**🎯 Match Winner**  \n{slip.get('predicted_winner', 'N/A')}")
                        st.markdown(f"**🔢 Confidence**  \n{slip.get('winner_confidence', 'N/A')}%")
                    
                    with col2:
                        st.markdown(f"**⚽ Total Goals**  \n{slip.get('predicted_total_goals', 'N/A')}")
                        st.markdown(f"**📈 Over/Under 2.5**  \n{slip.get('predicted_over_under', 'N/A')}")
                    
                    with col3:
                        st.markdown(f"**✅ Both Teams Score**  \n{slip.get('predicted_btts', 'N/A')}")
                        st.markdown(f"**🏆 Recommended Bet**  \n{slip.get('recommended_bet', 'N/A')}")
                    
                    st.markdown(f"**📅 Fixture ID:** {slip['fixture_id']} | **📆 Date:** {slip.get('date', 'N/A')}")
        else:
            st.warning("No enriched data available for analysis.")
    else:
        st.warning("No fixtures available for today.")

if __name__ == "__main__":
    main()