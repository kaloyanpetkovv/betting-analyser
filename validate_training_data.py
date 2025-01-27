import logging
import pandas as pd
import numpy as np
from utils import validate_training_data
from constants import FEATURE_COLUMNS, OUTCOME_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def validate_data_quality(data: pd.DataFrame) -> bool:
    """Validate data quality and completeness."""
    try:
        # Check for minimum required rows
        if len(data) < 100:
            logging.error(f"Insufficient data: only {len(data)} rows found. Minimum 100 required.")
            return False

        # Check data types
        expected_types = {
            "Home Team": str,
            "Away Team": str,
            "League ID": (int, np.int64),
            "Total Goals": (float, np.float64),
            "Total Corners": (float, np.float64)
        }

        for col, expected_type in expected_types.items():
            if col in data.columns:
                if not all(isinstance(x, expected_type) for x in data[col].dropna()):
                    logging.error(f"Invalid data type in column {col}")
                    return False

        # Check value ranges
        validations = {
            "Total Goals": (0, 20),
            "Total Corners": (0, 30),
            "League ID": (1, 1000)
        }

        for col, (min_val, max_val) in validations.items():
            if col in data.columns:
                if not all(data[col].between(min_val, max_val)):
                    logging.error(f"Values out of range in column {col}")
                    return False

        # Validate categorical values
        categorical_validations = {
            "Result": ["Home Win", "Draw", "Away Win"],
            "Over/Under 2.5": ["Over", "Under"],
            "Over/Under 1.5": ["Over", "Under"],
            "Over/Under 0.5": ["Over", "Under"],
            "Over/Under 3.5": ["Over", "Under"],
            "Over/Under 9.5 Corners": ["Over", "Under"],
            "Both Teams to Score": ["Yes", "No"],
            "X1": ["Yes", "No"],
            "X2": ["Yes", "No"]
        }

        for col, valid_values in categorical_validations.items():
            if col in data.columns:
                invalid_values = set(data[col].dropna().unique()) - set(valid_values)
                if invalid_values:
                    logging.error(f"Invalid values in {col}: {invalid_values}")
                    return False

        return True

    except Exception as e:
        logging.error(f"Error during data quality validation: {e}")
        return False

def main():
    """Main validation function."""
    try:
        # Load your training data
        training_data = pd.read_csv("training_data.csv")  # Adjust path as needed
        
        logging.info("Starting training data validation...")
        logging.info(f"Data shape: {training_data.shape}")
        
        # Basic structure validation
        if not validate_training_data(training_data):
            logging.error("Basic structure validation failed!")
            return False
            
        # Data quality validation
        if not validate_data_quality(training_data):
            logging.error("Data quality validation failed!")
            return False
            
        # Summary statistics
        logging.info("\nData Summary:")
        logging.info(f"Total samples: {len(training_data)}")
        logging.info("\nMissing values:")
        logging.info(training_data.isnull().sum())
        
        logging.info("\nValue distributions:")
        for col in OUTCOME_COLUMNS:
            if col in training_data.columns:
                logging.info(f"\n{col}:")
                logging.info(training_data[col].value_counts())
        
        logging.info("\nValidation completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Error during validation: {e}")
        return False

if __name__ == "__main__":
    main()
