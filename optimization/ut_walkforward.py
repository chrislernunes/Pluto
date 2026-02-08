import os
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from metrics.metric_utils import compute_metrics

# === CONFIGURATION ===

CSV_PATH = "reports/optimization/ut_optimization_results.csv"
TRADEBOOK_DIR = "storage/tradebooks/ut/"
OUTPUT_DIR = "reports/walkforward/ut/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Walkforward Parameters ===

TOP_N = 5                         # Select top-N UIDs
LOOKBACK_MONTHS = 3              # Look back over K months
TEST_MONTH = "2024-08"           # Format: YYYY-MM
SELECTION_METRIC = "calmar_ratio"  # Metric to rank by

# === Step 1: Get UIDs with Good Past Performance ===

def select_top_uids(csv_path, metric, test_month, lookback_months, top_n):
    df = pd.read_csv(csv_path)
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')  # assumes a 'date' column exists in the results
    test_month_dt = pd.Period(test_month)

    # Filter to lookback window
    lookback_start = test_month_dt - lookback_months
    df_lb = df[(df['month'] >= lookback_start) & (df['month'] < test_month_dt)]

    if df_lb.empty:
        print("❌ No training data found in lookback window.")
        return []

    # Aggregate metric per UID across lookback period
    grouped = df_lb.groupby('uid')[metric].mean().reset_index()
    top_uids = grouped.sort_values(metric, ascending=False)['uid'].head(top_n).tolist()

    print(f"\n✅ Selected top {top_n} UIDs based on mean {metric} from {lookback_start} to {test_month_dt - 1}")
    return top_uids

# === Step 2: Load Tradebooks for Selected UIDs and Filter by Test Month ===

def load_test_month_tradebooks(uids, tradebook_dir, test_month):
    start = pd.to_datetime(test_month + "-01")
    end = (start + relativedelta(months=1)) - timedelta(days=1)

    tradebooks = []
    for uid in uids:
        path = os.path.join(tradebook_dir, f"{uid}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['date'])
            df['uid'] = uid
            df = df[(df['date'] >= start) & (df['date'] <= end)]
            tradebooks.append(df)
        else:
            print(f"⚠️ Missing tradebook: {path}")
    return pd.concat(tradebooks, ignore_index=True) if tradebooks else pd.DataFrame()

# === Step 3: Run Walkforward Evaluation ===

def run_ut_walkforward():
    top_uids = select_top_uids(
        csv_path=CSV_PATH,
        metric=SELECTION_METRIC,
        test_month=TEST_MONTH,
        lookback_months=LOOKBACK_MONTHS,
        top_n=TOP_N
    )

    if not top_uids:
        return

    test_tb = load_test_month_tradebooks(top_uids, TRADEBOOK_DIR, TEST_MONTH)
    if test_tb.empty:
        print("❌ No trades found in test period.")
        return

    metrics = compute_metrics(test_tb)

    print("\n===== WALKFORWARD METRICS for Test Month:", TEST_MONTH, "=====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Save results
    test_tb.to_csv(os.path.join(OUTPUT_DIR, f"{TEST_MONTH}_walkforward_tradebook.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, f"{TEST_MONTH}_walkforward_metrics.csv"), index=False)

# === Run It ===
if __name__ == "__main__":
    run_ut_walkforward()
