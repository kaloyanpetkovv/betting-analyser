import logging
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score
import pandas as pd
from typing import List, Dict, Any  # For type hinting

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def evaluate_predictions(
    y_class_true: pd.DataFrame,
    y_class_pred: pd.DataFrame,
    y_reg_true: pd.DataFrame,
    y_reg_pred: pd.DataFrame,
    class_targets: List[str],
    reg_targets: List[str]
) -> Dict[str, Any]:
    """
    Evaluate predictions for classification and regression targets, and display metrics.
    Args:
        y_class_true (pd.DataFrame): True values for classification targets.
        y_class_pred (pd.DataFrame): Predicted values for classification targets.
        y_reg_true (pd.DataFrame): True values for regression targets.
        y_reg_pred (pd.DataFrame): Predicted values for regression targets.
        class_targets (list): List of classification target names.
        reg_targets (list): List of regression target names.
    Returns:
        dict: Overall and per-target metrics for classification and regression.
    """
    try:
        # Ensure input data is a DataFrame
        if not isinstance(y_class_true, pd.DataFrame) or not isinstance(y_class_pred, pd.DataFrame):
            raise ValueError("Classification inputs (y_class_true, y_class_pred) must be pandas DataFrames.")
        if not isinstance(y_reg_true, pd.DataFrame) or not isinstance(y_reg_pred, pd.DataFrame):
            raise ValueError("Regression inputs (y_reg_true, y_reg_pred) must be pandas DataFrames.")

        # Validate data dimensions
        if y_class_true.shape != y_class_pred.shape:
            raise ValueError(f"Classification shape mismatch - true: {y_class_true.shape}, pred: {y_class_pred.shape}")
        if y_reg_true.shape != y_reg_pred.shape:
            raise ValueError(f"Regression shape mismatch - true: {y_reg_true.shape}, pred: {y_reg_pred.shape}")
            
        # Validate target names
        if len(class_targets) != y_class_true.shape[1] or len(class_targets) != y_class_pred.shape[1]:
            raise ValueError(f"Class targets mismatch - targets: {len(class_targets)}, true: {y_class_true.shape[1]}, pred: {y_class_pred.shape[1]}")
        if len(reg_targets) != y_reg_true.shape[1] or len(reg_targets) != y_reg_pred.shape[1]:
            raise ValueError(f"Reg targets mismatch - targets: {len(reg_targets)}, true: {y_reg_true.shape[1]}, pred: {y_reg_pred.shape[1]}")

        # Evaluate Classification
        classification_metrics = {}
        for i, target in enumerate(class_targets):
            accuracy = accuracy_score(y_class_true.iloc[:, i], y_class_pred.iloc[:, i])
            f1 = f1_score(y_class_true.iloc[:, i], y_class_pred.iloc[:, i], average="weighted")
            precision = precision_score(y_class_true.iloc[:, i], y_class_pred.iloc[:, i], average="weighted", zero_division=0)
            recall = recall_score(y_class_true.iloc[:, i], y_class_pred.iloc[:, i], average="weighted", zero_division=0)
            classification_metrics[target] = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
            }

        # Evaluate Regression
        regression_metrics = {}
        for i, target in enumerate(reg_targets):
            mse = mean_squared_error(y_reg_true.iloc[:, i], y_reg_pred.iloc[:, i])
            regression_metrics[target] = {
                "mean_squared_error": mse,
            }

        # Calculate Overall Metrics
        overall_metrics = {
            "classification_accuracy": sum(metric["accuracy"] for metric in classification_metrics.values()) / len(classification_metrics),
            "classification_f1_score": sum(metric["f1_score"] for metric in classification_metrics.values()) / len(classification_metrics),
            "regression_mse": sum(metric["mean_squared_error"] for metric in regression_metrics.values()) / len(regression_metrics),
        }

        # Add validation for suspiciously perfect metrics
        if overall_metrics["classification_accuracy"] > 0.99:
            logging.warning("Warning: Classification accuracy suspiciously high - possible overfitting")
        if overall_metrics["regression_mse"] < 0.01:
            logging.warning("Warning: Regression MSE suspiciously low - possible overfitting")

        # Display Metrics
        logging.info("### Classification Metrics ###")
        for target, metrics in classification_metrics.items():
            logging.info(f"- {target}: {metrics}")

        logging.info("### Regression Metrics ###")
        for target, metrics in regression_metrics.items():
            logging.info(f"- {target}: {metrics}")

        logging.info("### Overall Metrics ###")
        for metric, value in overall_metrics.items():
            logging.info(f"- {metric}: {value:.4f}")

        return {
            "classification_metrics": classification_metrics,
            "regression_metrics": regression_metrics,
            "overall_metrics": overall_metrics,
        }

    except Exception as e:
        logging.error(f"Error during predictions evaluation: {e}")
        raise