import os
import pandas as pd
from metrics import compute_all_metrics  # Make sure metrics.py is in PYTHONPATH or same folder

def load_tradebook(file_path: str) -> pd.DataFrame:
    """
    Load a tradebook DataFrame from a CSV file and parse datetime columns.

    Args:
        file_path (str): Path to the .csv file.

    Returns:
        pd.DataFrame: Cleaned and parsed tradebook DataFrame.
    """
    try:
        df = pd.read_csv(file_path)

        # Drop redundant index column if present
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Parse datetime fields
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["system_timestamp"] = pd.to_datetime(df["system_timestamp"])
        df["date"] = pd.to_datetime(df["date"]).dt.date  # Keep as date, not full datetime

        return df
    except Exception as e:
        print(f"Error loading tradebook: {e}")
        return pd.DataFrame()

def analyze_tradebook(file_path: str, verbose: bool = True) -> dict:
    """
    Analyze a tradebook using compute_all_metrics.

    Args:
        file_path (str): Path to the tradebook CSV file.
        verbose (bool): Whether to print the results to console.

    Returns:
        dict: Dictionary of computed metrics.
    """
    df = load_tradebook(file_path)
    if df.empty:
        print("Tradebook is empty or failed to load.")
        return {}

    if verbose:
        print(f"Loaded tradebook with {len(df)} trades from {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    metrics = compute_all_metrics(df)

    if verbose:
        for k, v in metrics.items():
            print(f"\n--- {k.upper().replace('_', ' ')} ---")
            print(v)

    return metrics

def main():
    # Path setup
    tradebook_dir = "storage/tradebooks/tbs"
    uid = "tbs_0_x0_0_1_NIFTY_P_20_10_0.25_0.5_True_True"
    tradebook_file = os.path.join(tradebook_dir, f"{uid}.csv")

    # Run analysis
    analyze_tradebook(tradebook_file)

if __name__ == "__main__":
    main()
