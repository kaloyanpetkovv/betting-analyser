import pickle
import logging
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from evaluation import evaluate_predictions  # Add this import

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_api_data(api_data):
    """Process raw API data for prediction."""
    try:
        processed_data = pd.DataFrame(api_data)
        
        # Encode categorical features
        encoders = {}
        categorical_cols = ["Home Team", "Away Team"]
        for col in categorical_cols:
            le = LabelEncoder()
            processed_data[col] = le.fit_transform(processed_data[col].astype(str))
            encoders[col] = le
            
        return processed_data, encoders
        
    except Exception as e:
        logging.error(f"Error processing API data: {e}")
        raise

def preprocess_data(data):
    """Preprocess data for model training."""
    try:
        processed_data = data.copy()

        # Convert numeric columns
        numeric_columns = {
            "League ID": int,
            "Total Goals": float,
            "Total Corners": float
        }
        
        for col, dtype in numeric_columns.items():
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                
        # Encode categorical features
        encoders = {}
        categorical_cols = ["Home Team", "Away Team"]
        
        for col in categorical_cols:
            le = LabelEncoder()
            processed_data[col] = le.fit_transform(processed_data[col].astype(str))
            encoders[col] = le
            
        # Convert result to numeric
        result_map = {"Home Win": 0, "Draw": 1, "Away Win": 2, "Unknown": 1}
        if "Result" in processed_data.columns:
            processed_data["Result"] = processed_data["Result"].map(result_map).fillna(1)

        # Convert binary columns
        binary_columns = [
            "Over/Under 2.5", "Over/Under 1.5", "Over/Under 0.5",
            "Over/Under 3.5", "X1", "X2", "Over/Under 9.5 Corners",
            "Both Teams to Score"
        ]
        
        binary_map = {
            "Over": 1, "Under": 0,
            "Yes": 1, "No": 0,
            True: 1, False: 0
        }
        
        for col in binary_columns:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].map(binary_map).fillna(0)

        # Select features
        feature_columns = ["Home Team", "Away Team", "League ID"]
        X = processed_data[feature_columns]
        
        # Select classification targets
        class_targets = ["Result"]
        class_targets.extend([col for col in binary_columns if col in processed_data.columns])
        y_class = processed_data[class_targets]
        
        # Select regression targets
        reg_targets = ["Total Goals"]
        if "Total Corners" in processed_data.columns:
            reg_targets.append("Total Corners")
        y_reg = processed_data[reg_targets]
        
        logging.info(f"Features shape: {X.shape}")
        logging.info(f"Classification targets shape: {y_class.shape}")
        logging.info(f"Regression targets shape: {y_reg.shape}")
        
        return X, y_class, y_reg, encoders
        
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        logging.error(f"Data columns: {data.columns}")
        logging.error(f"Data types: {data.dtypes}")
        raise

def train_model(data):
    """Train the prediction models."""
    try:
        # Validate basic requirements
        required_columns = ["Home Team", "Away Team", "League ID", "Result", "Total Goals"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Process the data
        X, y_class, y_reg, encoders = preprocess_data(data)
        
        # Define and store feature columns
        feature_columns = [
            "Home Team", "Away Team", "League ID",
            "Home Avg Goals", "Away Avg Goals",
            "Home Avg Conceded", "Away Avg Conceded",
            "Home Avg Corners", "Away Avg Corners",
            "Home Win Draw %", "Away Win Draw %",
            "Combined Avg Goals"
        ]
        
        X = data[feature_columns].copy()
        
        # Convert all numeric columns properly
        for col in feature_columns:
            if col not in ["Home Team", "Away Team"]:
                X[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # Encode categorical columns
        for col in ["Home Team", "Away Team"]:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        # Train/test split
        X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
        _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

        # Initialize models
        clf = MultiOutputClassifier(RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ))
        
        reg = MultiOutputRegressor(RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ))

        # Train models
        clf.fit(X_train, y_class_train)
        reg.fit(X_train, y_reg_train)

        # Evaluate
        class_preds = clf.predict(X_test)
        reg_preds = reg.predict(X_test)

        class_preds_df = pd.DataFrame(class_preds, columns=y_class.columns)
        reg_preds_df = pd.DataFrame(reg_preds, columns=y_reg.columns)

        metrics = evaluate_predictions(
            y_class_test, class_preds_df,
            y_reg_test, reg_preds_df,
            class_targets=y_class.columns.tolist(),
            reg_targets=y_reg.columns.tolist()
        )

        # Create model bundle with all required keys
        model_bundle = {
            "classifier": clf,
            "regressor": reg,
            "encoders": encoders,
            "metrics": metrics,
            "feature_columns": feature_columns,
            "class_targets": y_class.columns.tolist(),
            "reg_targets": y_reg.columns.tolist(),
            "sklearn_version": sklearn.__version__
        }

        # Save the model immediately after training
        save_model(model_bundle, "trained_model.pkl")

        return model_bundle

    except Exception as e:
        logging.error(f"Training error: {e}")
        raise ValueError(f"Model training failed: {e}")

def predict_match_outcomes(processed_data):
    """Generate predictions using processed API data."""
    try:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Make predictions
        class_predictions = clf.predict_proba(processed_data)
        reg_predictions = reg.predict(processed_data)
        
        return class_predictions, reg_predictions
        
    except Exception as e:
        logging.error(f"Error predicting outcomes: {e}")
        raise

def save_model(model_bundle: dict, filename: str) -> None:
    """Save the model bundle with validation."""
    try:
        # Create a complete bundle with all required information
        complete_bundle = {
            "classifier": model_bundle["classifier"],
            "regressor": model_bundle["regressor"],
            "encoders": model_bundle["encoders"],
            "metrics": model_bundle["metrics"],
            "feature_columns": model_bundle["feature_columns"],
            "class_targets": model_bundle["class_targets"],
            "reg_targets": model_bundle["reg_targets"],
            "sklearn_version": sklearn.__version__,
            "model_timestamp": pd.Timestamp.now().isoformat(),
            "model_params": {
                "classifier_params": model_bundle["classifier"].get_params(),
                "regressor_params": model_bundle["regressor"].get_params()
            }
        }

        # Verify all keys exist before saving
        for key in complete_bundle:
            if complete_bundle[key] is None:
                raise ValueError(f"Missing value for key: {key}")

        with open(filename, "wb") as file:
            pickle.dump(complete_bundle, file)
        logging.info(f"Model saved successfully to {filename}")
        
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def load_model(filename: str) -> dict:
    """Load and validate model bundle with version handling."""
    try:
        with open(filename, "rb") as file:
            model_bundle = pickle.load(file)

        # Check sklearn version compatibility
        saved_version = model_bundle.get("sklearn_version", "0.0.0")
        current_version = sklearn.__version__
        
        if saved_version != current_version:
            logging.warning(f"Version mismatch - saved: {saved_version}, current: {current_version}")
            # Retrain model if versions are incompatible
            raise ValueError("Sklearn version mismatch - model needs retraining")

        # Verify all required components
        required_keys = [
            "classifier", "regressor", "encoders", "metrics",
            "feature_columns", "class_targets", "reg_targets"
        ]
        
        missing_keys = [key for key in required_keys if key not in model_bundle]
        if missing_keys:
            raise ValueError(f"Invalid model bundle: missing {missing_keys}")

        # Verify model components are valid
        if not isinstance(model_bundle["classifier"], MultiOutputClassifier):
            raise ValueError("Invalid classifier type")
        if not isinstance(model_bundle["regressor"], MultiOutputRegressor):
            raise ValueError("Invalid regressor type")

        logging.info(f"Model loaded successfully (saved on: {model_bundle.get('model_timestamp', 'unknown')})")
        return model_bundle

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
