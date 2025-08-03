import subprocess
import sys
from pathlib import Path  # <-- New import

import pandas as pd
from drift import detect_drift

# --- CONFIGURATION ---
# Define the project root to create robust, absolute paths
ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DATA_PATH = ROOT / "data/processed/train.parquet"
RAW_DATA_PATH = ROOT / "data/raw/california_housing.csv"
NEW_DATA_SAMPLE_SIZE = 5000


def run_retraining_pipeline():
    """
    Orchestrates the drift detection and retraining process.
    """
    print("🚀 Starting retraining pipeline...")

    # --- 1. SIMULATE NEW DATA ---
    # In a real system, you would load this from a prediction log database.
    # Here, we simulate it by taking a slice of the raw data and altering it.
    print(
        f"\nSTEP 1: Simulating collection of {NEW_DATA_SAMPLE_SIZE} new data points..."
    )
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH)
        # We'll pretend the last N rows are "new" data
        new_data_sample = raw_df.tail(NEW_DATA_SAMPLE_SIZE).copy()

        # To GUARANTEE drift for this test, let's drastically alter a key feature.
        # This ensures the drift detection logic will fire.
        print("INFO: Artificially altering 'MedInc' to trigger drift detection.")
        new_data_sample["MedInc"] = new_data_sample["MedInc"] * 2.5
    except Exception as e:
        print(f"❌ Failed to simulate new data: {e}")
        sys.exit(1)

    # --- 2. DETECT DRIFT ---
    print("\nSTEP 2: Detecting data drift...")
    drift_results = detect_drift(REFERENCE_DATA_PATH, new_data_sample)

    if drift_results.get("error"):
        print(f"❌ Error during drift detection: {drift_results['error']}")
        sys.exit(1)

    if not drift_results["drift_detected"]:
        print("✅ No significant data drift detected. Pipeline ends here.")
        sys.exit(0)

    print("🚨 DRIFT DETECTED! Proceeding with retraining.")
    print("Drift details:", drift_results["details"]["MedInc"])

    # --- 3. PREPARE DATA & RETRAIN ---
    # We append the new, drifted data to the original raw data to simulate
    # an updated dataset. DVC will use this updated file for retraining.
    print("\nSTEP 3: Appending new data and triggering DVC pipeline...")
    try:
        # NOTE: In a production scenario, you might want versioning or backups
        # before overwriting your raw data source.
        updated_df = pd.concat([raw_df, new_data_sample], ignore_index=True)
        updated_df.to_csv(RAW_DATA_PATH, index=False)
        print(f"Updated '{RAW_DATA_PATH}' with {len(new_data_sample)} new rows.")

        # Run the full DVC pipeline to retrain and register the new model
        print("Executing 'dvc repro'...")
        subprocess.run(["dvc", "repro"], check=True)
        print(
            "\n✅ DVC pipeline executed successfully. Model has been retrained and registered."
        )

    except FileNotFoundError:
        print("❌ Error: DVC command not found. Is DVC installed and in your PATH?")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ DVC pipeline failed with exit code {e.returncode}.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred during the retraining step: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_retraining_pipeline()
