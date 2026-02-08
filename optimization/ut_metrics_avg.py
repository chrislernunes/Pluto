import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import os
from metrics.metrics import apply_slippage, compute_all_metrics, compute_summary_metrics, compute_daily_pnl
from generate_reports import *
from utils.definitions import *
import json

# MODE = 'Viz'
MODE = 'Finalize'

# === CONFIGURATION ===

CSV_PATH = "reports/optimization/ut_optimization_results.csv"  # Path to UT strategy results
PARAM_NAMES = [
    'active_weekday', 'session', 'delay', 'timeframe', 'underlying',
    'selector', 'selector_val', 'hedge_shift', 'sl_pct', 'tgt_pct',
    'max_reset', 'trail_on', 'trigger_pct', 'trail_pct'
]
AGG_METRICS = ['total_pnl','return_on_margin', 'max_drawdown', 'calmar_ratio']
TRADEBOOK_DIR = "storage/tradebooks/ut"
OUTPUT_DIR = "reports/optimization/ut/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
REGENERATE_METRICS = False

if REGENERATE_METRICS:
    generate_metrics_from_tradebooks(
        path=TRADEBOOK_DIR,
        metrics_file=CSV_PATH,
        margin=MARGIN,
        slippage=SLIPPAGE,
        param_names=PARAM_NAMES,
        prefix="ut_"
    )

# Optional filter: Set to restrict to specific parameter settings
FILTER_PARAMS = {
    'underlying': 'SENSEX',
    'session': ['x0'],  # example usage
    #'trigger_pct': [0.2,0.25],
    'sl_pct': [0.3,0.4],
    'max_reset':[1,2],
    'selector_val': [-1,0]
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

        for param in PARAM_NAMES:
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
    # Extract parameter values from UID
    df[PARAM_NAMES] = (
        df['uid'].str.replace('ut_', '', regex=False)
                .str.split('_', expand=True)
    )
    # Type conversions
    df['active_weekday'] = df['active_weekday'].astype(int)
    df['delay'] = df['delay'].astype(int)
    df['timeframe'] = df['timeframe'].astype(int)
    df['selector_val'] = df['selector_val'].astype(int)
    df['hedge_shift'] = df['hedge_shift'].astype(int)
    df['sl_pct'] = df['sl_pct'].astype(float)
    df['tgt_pct'] = df['tgt_pct'].astype(float)
    df['max_reset'] = df['max_reset'].astype(int)
    df['trail_on'] = df['trail_on'].astype(str) == 'True'
    df['trigger_pct'] = df['trigger_pct'].astype(float)
    df['trail_pct'] = df['trail_pct'].astype(float)
    return df

if MODE == 'Viz':
    # === STEP 1: Load and Preprocess Data ===
    df = read_and_process_df()
    
    # === STEP 2: Apply Optional Filters ===
    if FILTER_PARAMS:
        df = apply_filter(df, FILTER_PARAMS)

    print("Number of Filtered UIDs: " + str(len(df['uid'])))

    # === STEP 3: Run Reports for Each Underlying Separately ===

    for underlying in df['underlying'].unique():
        df_underlying = df[df['underlying'] == underlying]
        generate_reports_by_param(df_underlying, underlying)

elif MODE == 'Finalize':
    FINAL_PARAMS = [{
        'underlying': 'NIFTY',
        'session': ['x0'],  # example usage
        'sl_pct': [0.4,0.5],
        'selector_val': [-1]
        }, 
        {
        'underlying': 'NIFTY',
        'session': ['s2'],  # example usage
        'sl_pct': [0.3,0.4],
        'max_reset':[1,2]
    #'selector_val': [-1]
        },
        {
        'underlying': 'SENSEX',
        'session': ['s1'],  # example usage
        'trigger_pct': [0.2,0.25]
        },
        {
        'underlying': 'SENSEX',
        'session': ['x0'],  # example usage
        'trigger_pct': [0.2,0.25],
        'sl_pct': [0.3,0.4],
        #'max_reset':[1,2]
        'selector_val': [-1,0]
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
    print(len(FINAL_UIDS))
    margin = 100000*len(FINAL_UIDS)
    print(margin)

    uid_config = {
    "uid_configs_ut_nifty": [{"uid": uid, "weight": 1} for uid in FINAL_UIDS if "NIFTY" in uid],
    "uid_configs_ut_sensex": [{"uid": uid, "weight": 1} for uid in FINAL_UIDS if "SENSEX" in uid],
    }
    with open("basket/configs/ut_basket_config_new.json", "w") as f:
        json.dump(uid_config, f, indent=2)

    merged_tradebook = load_and_merge_tradebooks(FINAL_UIDS, TRADEBOOK_DIR)
    if not merged_tradebook.empty:
        merged_tradebook = apply_slippage(merged_tradebook)
        metrics = compute_all_metrics(merged_tradebook,margin)

    print("\n===== FINALIZED BASKET METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    os.makedirs("reports/optimization/ut/final/", exist_ok=True)
    compute_daily_pnl(merged_tradebook).to_csv("reports/optimization/ut/final/daily.csv")
    metrics["monthly_pnl"].to_csv("reports/optimization/ut/final/monthly.csv")
else:
    print("Invalid MODE")

    
    

