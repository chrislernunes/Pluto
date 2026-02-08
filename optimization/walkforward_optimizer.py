import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from metrics.metrics import (
    apply_slippage,
    compute_daily_pnl,
    compute_summary_metrics_daily,
    monthly_pnl
)

# === CONFIGURATION ===
STRATEGY = "minpmisl"
TRADEBOOK_DIR = "storage/tradebooks/" + STRATEGY + "/"
OUTPUT_DIR = "reports/walkforward/" + STRATEGY + "/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_N = 50
LOOKBACK_MONTHS = 6
START_TRAINING_MONTH = "2024-01"
END_TEST_MONTH = "2025-05"
SELECTION_METRIC = "calmar_ratio"
DEFAULT_MARGIN = 100000

# === Step 1: Collect Daily PNLs for All UIDs ===

def get_all_uid_daily_pnls(tradebook_dir):
    uid_to_daily_pnl = {}
    for fname in os.listdir(tradebook_dir):
        if not fname.endswith(".csv"):
            continue
        uid = fname.replace(".csv", "")
        fpath = os.path.join(tradebook_dir, fname)

        df = pd.read_csv(fpath, parse_dates=["date"])
        if "turnover" not in df.columns or "value" not in df.columns:
            continue
        df = apply_slippage(df)
        daily_pnl = compute_daily_pnl(df)
        uid_to_daily_pnl[uid] = daily_pnl
    return uid_to_daily_pnl

# === Step 2: Compute Metrics in Lookback Window ===

def compute_uid_metrics(uid_to_daily_pnl, start_date, end_date):
    records = []
    for uid, daily_pnl in uid_to_daily_pnl.items():
        daily_pnl = daily_pnl[(daily_pnl.index >= start_date) & (daily_pnl.index <= end_date)]
        if daily_pnl.empty:
            continue
        metrics = compute_summary_metrics_daily(daily_pnl, DEFAULT_MARGIN)
        metrics["uid"] = uid
        records.append(metrics)
    return pd.DataFrame(records)

# === Step 3: Load Test Month Trades for Top UIDs ===

def load_test_trades(uids, tradebook_dir, test_start, test_end):
    tradebooks = []
    for uid in uids:
        path = os.path.join(tradebook_dir, f"{uid}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            df = apply_slippage(df)
            df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
            df["uid"] = uid
            tradebooks.append(df)
    return pd.concat(tradebooks, ignore_index=True) if tradebooks else pd.DataFrame()

# === Walkforward for a Single Month ===

def walkforward_for_month(uid_daily_pnls, test_month_str):
    test_start = pd.to_datetime(f"{test_month_str}-01")
    test_end = (test_start + relativedelta(months=1)) - timedelta(days=1)
    lookback_start = test_start - relativedelta(months=LOOKBACK_MONTHS)
    lookback_end = test_start - timedelta(days=1)

    print(f"\n📅 Walkforward for {test_month_str} | Lookback: {lookback_start.date()} to {lookback_end.date()}")

    metric_df = compute_uid_metrics(uid_daily_pnls, lookback_start, lookback_end)
    if metric_df.empty:
        print("❌ No valid UIDs found in lookback window.")
        return
    print(metric_df.sort_values(SELECTION_METRIC, ascending=False).head(TOP_N))
    top_uids = metric_df.sort_values(SELECTION_METRIC, ascending=False).head(TOP_N)["uid"].tolist()
    print(f"✅ Top-{TOP_N} UIDs selected: {top_uids}")

    test_trades = load_test_trades(top_uids, TRADEBOOK_DIR, test_start, test_end)
    if test_trades.empty:
        print("❌ No trades found in test period.")
        return

    test_daily_pnl = compute_daily_pnl(test_trades)
    metrics = compute_summary_metrics_daily(test_daily_pnl, DEFAULT_MARGIN)

    # Output
    test_trades.to_csv(os.path.join(OUTPUT_DIR, f"{test_month_str}_walkforward_tradebook.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, f"{test_month_str}_walkforward_metrics.csv"), index=False)
    pd.DataFrame(top_uids, columns=["uid"]).to_csv(os.path.join(OUTPUT_DIR, f"{test_month_str}_top_uids.csv"), index=False)

    print(f"📊 Summary Metrics for {test_month_str}:")
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
    return test_trades  
# === Main Execution ===

def run_ut_walkforward_loop():
    print("🔁 Running rolling walkforward for " + STRATEGY + " strategy...")
    uid_daily_pnls = get_all_uid_daily_pnls(TRADEBOOK_DIR)

    current = pd.to_datetime(START_TRAINING_MONTH + "-01") + relativedelta(months=LOOKBACK_MONTHS)
    end = pd.to_datetime(END_TEST_MONTH + "-01")

    all_test_trades = []  # ✅ Accumulator for merged test tradebook

    while current <= end:
        test_month_str = current.strftime("%Y-%m")
        test_start = pd.to_datetime(f"{test_month_str}-01")
        test_end = (test_start + relativedelta(months=1)) - timedelta(days=1)

        test_trades = walkforward_for_month(uid_daily_pnls, test_month_str)

        if test_trades is not None and not test_trades.empty:
            all_test_trades.append(test_trades)

        current += relativedelta(months=1)
# === Compute Monthly PNL Across Entire Walkforward ===
    if all_test_trades:
        combined_test_trades = pd.concat(all_test_trades, ignore_index=True)
        combined_test_trades = apply_slippage(combined_test_trades)
        daily_pnl_series = compute_daily_pnl(combined_test_trades)
        monthly_results = monthly_pnl(daily_pnl_series)
        print(f"\n✅ Final walkforward monthly")
        print(monthly_results)
    else:
        print("\n⚠️ No valid test trades found across walkforward period.")


# === Run ===
if __name__ == "__main__":
    run_ut_walkforward_loop()
