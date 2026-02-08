import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
import os
from metrics.metrics import apply_slippage, compute_all_metrics, compute_summary_metrics, compute_daily_pnl

# === CONFIGURATION ===

CSV_PATH = "reports/optimization/spikesell_optimization_results.csv"
OUTPUT_DIR = "reports/optimization/spikesell/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODE = 'Viz'
#MODE = 'Finalize'

# All parameter names from UID
PARAM_NAMES = [
    'underlying', 'active_dte', 'session', 'delay', 'haste', 'timeframe', 'offset',
    'selector', 'selector_val', 'hedge_shift', 'lookback', 'spike_thresh', 'retrace_thresh',
    'sl_type', 'sl_val', 'trail_on', 'tgt_pct', 'lock_profit',
    'max_loss', 'lock_profit_trigger', 'lock_profit_to', 'subsequent_profit_step', 'subsequent_profit_amt'
]

# arameters to exclude from analysis
IGNORE_PARAMS = [
    'lock_profit', 'lock_profit_trigger', 'lock_profit_to', 'subsequent_profit_step', 'subsequent_profit_amt'
]

# Final params for report generation
REPORT_PARAMS = [p for p in PARAM_NAMES if p not in IGNORE_PARAMS]

AGG_METRICS = ['total_pnl', 'return_on_margin', 'max_drawdown', 'calmar_ratio']

TRADEBOOK_DIR = "storage/tradebooks/spikesell"

FILTER_PARAMS = {
    'underlying': "NIFTY",
    'session': 's1',
    'spike_thresh': [0.5],
    #'selector_val':[0,1],
    #'sl_val':[0.4,0.5]
}

def apply_filter(df, filter_dict):
    for key, val in filter_dict.items():
        if key not in df.columns:
            raise KeyError(f"Filter key '{key}' is not a valid column.")
        if isinstance(val, list):
            df = df[df[key].isin(val)]
        else:
            df = df[df[key] == val]
    return df

def generate_reports_by_param(df_subset, label):
        print(f"\n===== {label.upper()} PARAMETER-WISE METRIC SUMMARY =====\n")
        output_path = os.path.join(OUTPUT_DIR, label.upper())
        os.makedirs(output_path, exist_ok=True)

        for param in REPORT_PARAMS:
            grouped = df_subset.groupby(param)[AGG_METRICS].mean().reset_index()
            file_name = os.path.join(output_path, f"{param}_summary.csv")
            grouped.to_csv(file_name, index=False)
            print(f"\n--- {label.upper()} - Parameter: {param} ---")
            print(grouped.to_string(index=False))

def load_and_merge_tradebooks(uids, tradebook_dir):
    tradebooks = []
    for uid in uids:
        path = os.path.join(tradebook_dir, f"{uid}.csv")
        if os.path.exists(path):
            tb = pd.read_csv(path)
            tb['uid'] = uid  # optional: tag with UID for traceability
            tradebooks.append(tb)
        else:
            print(f"Tradebook not found: {path}")
    return pd.concat(tradebooks, ignore_index=True) if tradebooks else pd.DataFrame()

def read_and_process_df():
    df = pd.read_csv(CSV_PATH)
    # Parse UID
    df[PARAM_NAMES] = (
        df['uid'].str.replace('spikesell_', '', regex=False)
        .str.split('_', expand=True)
    )

    # Convert types
    df['active_dte'] = df['active_dte'].astype(int)
    df['delay'] = df['delay'].astype(int)
    df['haste'] = df['haste'].astype(int)
    df['timeframe'] = df['timeframe'].astype(int)
    df['offset'] = df['offset'].astype(int)
    df['selector_val'] = df['selector_val'].astype(float)
    df['hedge_shift'] = df['hedge_shift'].astype(int)
    df['lookback'] = df['lookback'].astype(int)
    df['spike_thresh'] = df['spike_thresh'].astype(float)
    df['retrace_thresh'] = df['retrace_thresh'].astype(float)
    df['sl_val'] = df['sl_val'].astype(float)
    df['trail_on'] = df['trail_on'].astype(str) == 'True'
    df['tgt_pct'] = df['tgt_pct'].astype(float)
    df['lock_profit'] = df['lock_profit'].astype(str) == 'True'
    df['max_loss'] = df['max_loss'].astype(int)
    df['lock_profit_trigger'] = df['lock_profit_trigger'].astype(int)
    df['lock_profit_to'] = df['lock_profit_to'].astype(int)
    df['subsequent_profit_step'] = df['subsequent_profit_step'].astype(int)
    df['subsequent_profit_amt'] = df['subsequent_profit_amt'].astype(int)

    return df

if MODE == 'Viz':
    # === STEP 1: Load and Preprocess Data ===
    df = read_and_process_df()

    # === STEP 2: Apply Optional Filtering ===
    if FILTER_PARAMS:
        df = apply_filter(df, FILTER_PARAMS)
    print("Number of Filtered UIDs: " + str(len(df['uid'])))

    # === STEP 3: Separate Reports for NIFTY and SENSEX ===

    for underlying in df['underlying'].unique():
        df_underlying = df[df['underlying'] == underlying]
        generate_reports_by_param(df_underlying, underlying)

elif MODE == 'Finalize':
    FINAL_PARAMS = [
        {
            'underlying': "NIFTY",
            'session': 's2',
            'selector_val':[0,1],
        },
        {
            'underlying': "SENSEX",
            'session': 's2',
            'selector_val':[0,1],
            'sl_val':[0.4,0.5]
        }
    ]

    FINAL_UIDS = []
    for params in FINAL_PARAMS:
        # === STEP 1: Load and Preprocess Data ===
        df = read_and_process_df()
        
        # === STEP 2: Apply Optional Filters ===
        if params:
            df = apply_filter(df, params)

        for uid in df['uid']:
            FINAL_UIDS.append(uid)
    print(FINAL_UIDS)
    margin = 100000*len(FINAL_UIDS)
    print(margin)

    uid_config = {
    "uid_configs_spikesell_nifty": [{"uid": uid, "weight": 1} for uid in FINAL_UIDS if "NIFTY" in uid],
    "uid_configs_spikesell_sensex": [{"uid": uid, "weight": 1} for uid in FINAL_UIDS if "SENSEX" in uid],
    }
    with open("basket/configs/spikesell_basket_config_new.json", "w") as f:
        json.dump(uid_config, f, indent=2)

    merged_tradebook = load_and_merge_tradebooks(FINAL_UIDS, TRADEBOOK_DIR)
    if not merged_tradebook.empty:
        merged_tradebook = apply_slippage(merged_tradebook)
        metrics = compute_all_metrics(merged_tradebook,margin)

    print("\n===== FINALIZED BASKET METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    os.makedirs("reports/optimization/spikesell/final/", exist_ok=True)
    compute_daily_pnl(merged_tradebook).to_csv("reports/optimization/spikesell/final/daily.csv")
    metrics["monthly_pnl"].to_csv("reports/optimization/spikesell/final/monthly.csv")

else:
    print("Invalid MODE")
