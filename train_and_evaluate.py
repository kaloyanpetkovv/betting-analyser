import logging
import pandas as pd
from model import train_model, save_model, load_model

def train_and_evaluate(data):
    """Train and evaluate the model with season data."""
    try:
        model_path = "trained_model.pkl"
        
        # Add data quality checks
        if len(data) < 100:
            logging.warning(f"Small training dataset: only {len(data)} matches")
        
        logging.info("\nTraining Data Summary:")
        logging.info(f"Total matches: {len(data)}")
        if "League Name" in data.columns:
            logging.info("\nMatches per league:")
            logging.info(data["League Name"].value_counts())
        if "Result" in data.columns:
            logging.info("\nResult distribution:")
            logging.info(data["Result"].value_counts(normalize=True))

        # Train model
        model_bundle = train_model(data)
        
        # Save model
        save_model(model_bundle, model_path)
        logging.info(f"Model saved to {model_path}")
        
        return model_bundle, model_bundle["metrics"]
        
    except Exception as e:
        logging.error(f"Error in train_and_evaluate: {e}")
        raise

if __name__ == "__main__":
    # Test training with season data
    try:
        training_data = pd.read_csv("season_training_data.csv")
        model_bundle, metrics = train_and_evaluate(training_data)
        print("Training successful!")
        print("\nMetrics:", metrics)
    except Exception as e:
        print(f"Training failed: {e}")
