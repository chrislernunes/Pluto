import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import os
import json
from metrics.metrics import apply_slippage, compute_all_metrics, compute_summary_metrics, compute_daily_pnl
from generate_reports import *
from utils.definitions import *
from collections import defaultdict
# === CONFIGURATION ===

CSV_PATH = "reports/optimization/minpmisl_optimization_results.csv"  # Path to RESULTS CSV
PARAM_NAMES = [
    'active_weekday', 'session', 'delay', 'timeframe', 'underlying',
    'selector', 'selector_val', 'hedge_shift', 'sl_pct', 'tgt_pct',
    'max_reset', 'trail_on', 'Target', 'ps'
]
AGG_METRICS = ['total_pnl','return_on_margin', 'max_drawdown', 'calmar_ratio']
TRADEBOOK_DIR = "storage/tradebooks/minpmisl"
OUTPUT_DIR = "reports/optimization/minpmisl/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

generate_metrics_from_tradebooks(
    path=TRADEBOOK_DIR,
    metrics_file=CSV_PATH,
    margin=MARGIN,
    slippage=SLIPPAGE,
    param_names=PARAM_NAMES,
    prefix="minpmisl_"
)

# Optional filter: Set as a dictionary like {'session': 'x0', 'sl_pct': 0.5}
# Leave as empty dict {} to skip filtering
FILTER_PARAMS = {
    'underlying': 'SENSEX',
    'session': 's0',  
    #'ps': [0.005, 0.007, 0.010],
    'delay': [0,15,30],
    #'sl_pct': [0.5,0.6,0.8]   
}

# MODE = 'Viz'
MODE = 'Finalize'


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
    df[PARAM_NAMES] = (
        df['uid'].str.replace('minpmisl_', '', regex=False)
        .str.split('_', expand=True)
    )
    # Convert appropriate columns
    df['delay'] = df['delay'].astype(int)
    df['sl_pct'] = df['sl_pct'].astype(float)
    df['tgt_pct'] = df['tgt_pct'].astype(float)
    df['ps'] = df['ps'].astype(float)
    df['max_reset'] = df['max_reset'].astype(int)
    df['trail_on'] = df['trail_on'].astype(str) == 'True'
    df['Target'] = df['Target'].astype(str) == 'True'
    return df

def get_margin():
    session_count = defaultdict(lambda: defaultdict(int))  # session_count['NIFTY']['x0'] = count

    for uid in FINAL_UIDS:
        parts = uid.replace("minpmisl_", "").split("_")
        if len(parts) < 5:
            print(f"Skipping malformed UID: {uid}")
            continue
        underlying = parts[4]
        session = parts[1]
        session_count[underlying][session] += 1

    # Function to compute effective UID count for margin
    def effective_count(sessions):
        return sessions.get("x0", 0) + max(sessions.get("s0", 0), sessions.get("s2", 0))

    nifty_sessions = session_count["NIFTY"]
    sensex_sessions = session_count["SENSEX"]

    nifty_effective = effective_count(nifty_sessions)
    sensex_effective = effective_count(sensex_sessions)

    final_count = max(nifty_effective, sensex_effective)
    final_margin = final_count * MARGIN

    # === Print Summary ===
    print("\n===== MARGIN CALCULATION BREAKDOWN =====")
    print(f"NIFTY session counts: {dict(nifty_sessions)}")
    print(f"SENSEX session counts: {dict(sensex_sessions)}")
    print(f"NIFTY effective UID count: {nifty_effective}")
    print(f"SENSEX effective UID count: {sensex_effective}")
    print(f"Final UID count used for margin: {final_count}")
    print(f"Total margin required: ₹{final_margin:,}")
    return final_margin

if MODE == 'Viz':
    # === STEP 1: Load and Preprocess Data ===
    df = read_and_process_df()
    
    # === STEP 2: Apply Optional Filters ===
    if FILTER_PARAMS:
        df = apply_filter(df, FILTER_PARAMS)

    print(df['uid'])

    # === STEP 3: Run Reports for Each Underlying Separately ===

    for underlying in df['underlying'].unique():
        df_underlying = df[df['underlying'] == underlying]
        generate_reports_by_param(df_underlying, underlying)

elif MODE == 'Finalize':
    FINAL_PARAMS = [
    {'underlying': 'NIFTY','session': 'x0',  'delay': [0,15,30,60]},
    {'underlying': 'NIFTY','session': 's0','delay': [0,60]},
    {'underlying': 'NIFTY','session': 's2','ps': [0.005, 0.007, 0.010],'sl_pct': [0.5,0.6,0.8]},
    {'underlying': 'SENSEX','session': 'x0','delay': [0,15,30],'sl_pct': [0.3,0.4,0.5,0.6]},
    {'underlying': 'SENSEX','session': 's2','sl_pct': [0.5,0.6,0.8]},
    {'underlying': 'SENSEX','session': 's0','delay': [0,15,30]}
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
    margin = get_margin()
    # margin = 100000*len(FINAL_UIDS)
    print(margin)

    uid_config = {
    "uid_configs_minpmisl_nifty": [{"uid": uid, "weight": 1} for uid in FINAL_UIDS if "NIFTY" in uid],
    "uid_configs_minpmisl_sensex": [{"uid": uid, "weight": 1} for uid in FINAL_UIDS if "SENSEX" in uid],
    }
    with open("basket/configs/minpmisl_basket_config_new.json", "w") as f:
        json.dump(uid_config, f, indent=2)

    merged_tradebook = load_and_merge_tradebooks(FINAL_UIDS, TRADEBOOK_DIR)
    if not merged_tradebook.empty:
        merged_tradebook = apply_slippage(merged_tradebook)
        metrics = compute_all_metrics(merged_tradebook,margin)

    print("\n===== FINALIZED BASKET METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    os.makedirs("reports/optimization/minpmisl/final/", exist_ok=True)
    compute_daily_pnl(merged_tradebook).to_csv("reports/optimization/minpmisl/final/daily.csv")
    metrics["monthly_pnl"].to_csv("reports/optimization/minpmisl/final/monthly.csv")
else:
    print("Invalid MODE")
