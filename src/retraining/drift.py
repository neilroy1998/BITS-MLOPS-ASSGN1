# src/retraining/drift.py
import pandas as pd
from scipy.stats import ks_2samp


def detect_drift(
    reference_data_path: str, new_data: pd.DataFrame, p_value_threshold: float = 0.05
):
    """
    Detects data drift between reference data and new data using the K-S test.

    Args:
        reference_data_path (str): Path to the reference dataset (e.g., original training data).
        new_data (pd.DataFrame): DataFrame containing the new data to check for drift.
        p_value_threshold (float): The significance level for the K-S test.

    Returns:
        dict: A report of drifted features and overall drift status.
    """
    try:
        reference_df = pd.read_parquet(reference_data_path)
        # We only need the features for comparison, not the target variable
        reference_features = reference_df.drop(columns=["MedHouseVal"], errors="ignore")
    except Exception as e:
        return {"error": f"Failed to load reference data: {e}", "drift_detected": True}

    drift_report = {}
    drift_detected = False

    for col in reference_features.columns:
        # Ensure the column exists in the new data before testing
        if col not in new_data.columns:
            drift_report[col] = "Column not found in new data"
            drift_detected = True
            continue

        # The ks_2samp test compares two samples and returns the test statistic and p-value
        ks_stat, p_value = ks_2samp(reference_features[col], new_data[col])

        # A p-value below our threshold suggests the distributions are different
        if p_value <= p_value_threshold:
            drift_detected = True
            drift_report[col] = {"p_value": round(p_value, 4), "drifted": True}
        else:
            drift_report[col] = {"p_value": round(p_value, 4), "drifted": False}

    return {"drift_detected": drift_detected, "details": drift_report}
