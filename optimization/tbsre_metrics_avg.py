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

CSV_PATH = "reports/optimization/tbsre_optimization_results.csv"
TRADEBOOK_DIR = "storage/tradebooks/tbsre"
OUTPUT_DIR = "reports/optimization/tbsre/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
REGENERATE_METRICS = False

PARAM_NAMES = [
    'underlying', 'active_dte', 'session', 'delay', 'haste', 'timeframe', 'offset',
    'selector', 'selector_val', 'hedge_shift', 'sl_type', 'sl_val', 'trail_on',
    'reentry_type', 'reentry_limit', 'reentry_val', 'tgt_pct',
    'reenter_on_tgt', 'lock_profit', 'max_loss', 'lock_profit_trigger',
    'lock_profit_to', 'subsequent_profit_step', 'subsequent_profit_amt'
]

AGG_METRICS = ['total_pnl','return_on_margin', 'max_drawdown', 'calmar_ratio']

if REGENERATE_METRICS:
    generate_metrics_from_tradebooks(
        path=TRADEBOOK_DIR,
        metrics_file=CSV_PATH,
        margin=MARGIN,
        slippage=SLIPPAGE,
        param_names=PARAM_NAMES,
        prefix="tbsre_"
    )

# Optional filter
FILTER_PARAMS = {
    'underlying': 'NIFTY',
    'active_dte': '0',
    'session': 'x0',
    #'delay': [0, 15, 30],
    #'sl_val': [0.3, 0.4, 0.5, 0.6],
    #'haste': 15,
    #'selector_val': [20,30,50],
    #'reentry_limit': [4]
    #'trail_on': False
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
            tb['uid'] = uid
            tradebooks.append(tb)
        else:
            print(f"Tradebook not found: {path}")
    return pd.concat(tradebooks, ignore_index=True) if tradebooks else pd.DataFrame()

def read_and_process_df():
    df = pd.read_csv(CSV_PATH)
    df[PARAM_NAMES] = (
        df['uid'].str.replace('tbsre_', '', regex=False)
        .str.split('_', expand=True)
    )

    # Cast types
    df['delay'] = df['delay'].astype(int)
    df['haste'] = df['haste'].astype(int)
    df['timeframe'] = df['timeframe'].astype(int)
    df['offset'] = df['offset'].astype(int)
    df['selector_val'] = df['selector_val'].astype(int)
    df['hedge_shift'] = df['hedge_shift'].astype(int)
    df['sl_val'] = df['sl_val'].astype(float)
    df['tgt_pct'] = df['tgt_pct'].astype(float)
    df['trail_on'] = df['trail_on'].astype(str) == 'True'
    df['reentry_type'] = df['reentry_type'].astype(int)
    df['reentry_limit'] = df['reentry_limit'].astype(int)
    df['reentry_val'] = df['reentry_val'].astype(float)
    df['reenter_on_tgt'] = df['reenter_on_tgt'].astype(str) == 'True'
    df['lock_profit'] = df['lock_profit'].astype(str) == 'True'
    df['max_loss'] = df['max_loss'].astype(float)
    df['lock_profit_trigger'] = df['lock_profit_trigger'].astype(float)
    df['lock_profit_to'] = df['lock_profit_to'].astype(float)
    df['subsequent_profit_step'] = df['subsequent_profit_step'].astype(float)
    df['subsequent_profit_amt'] = df['subsequent_profit_amt'].astype(float)
    return df

def get_margin():
    session_count = defaultdict(lambda: defaultdict(int))  # session_count['NIFTY']['x0'] = count

    for uid in FINAL_UIDS:
        parts = uid.replace("tbsre_", "").split("_")
        if len(parts) < 3:
            print(f"Skipping malformed UID: {uid}")
            continue
        underlying = parts[0]
        session = parts[2]
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
    df = read_and_process_df()
    if FILTER_PARAMS:
        df = apply_filter(df, FILTER_PARAMS)

    print(df['uid'])
    print("Number of UIDS: " + str(len(df['uid'])))

    for underlying in df['underlying'].unique():
        df_underlying = df[df['underlying'] == underlying]
        generate_reports_by_param(df_underlying, underlying)

elif MODE == 'Finalize':
    FINAL_PARAMS = [
    {'underlying': 'NIFTY','session': 's2','selector_val': [20,30,50],'delay': [30,60],'trail_on': False},
    {'underlying': 'NIFTY','active_dte': '0','session': 'x0','delay': [0, 15, 30]}, #,'sl_val': [0.3, 0.4, 0.5, 0.6],'selector_val': [20,30,50,70],'reentry_limit': [1,2]},
    {'underlying': 'SENSEX','active_dte': '0','session': 'x0','sl_val': [0.3, 0.4, 0.5, 0.6],'selector_val': [20,30,50, 70],'reentry_limit': [1,2]},
    {'underlying': 'SENSEX','active_dte': '0','session': 's2','sl_val': [0.5, 0.6, 0.8]},
    {'underlying': 'NIFTY','active_dte': '0','session': 's0','haste': 15,'selector_val': [20,30,50]},
    {'underlying': 'SENSEX','active_dte': '0','session': 's0','delay': [0, 15, 30],'sl_val': [0.3, 0.4, 0.5, 0.6],'reentry_limit': [1,2]}
    ]

    FINAL_UIDS = []
    for params in FINAL_PARAMS:
        df = read_and_process_df()
        df = apply_filter(df, params)
        FINAL_UIDS.extend(df['uid'].tolist())

    print(FINAL_UIDS)
    print(f"Total UIDs: {len(FINAL_UIDS)}")
    # margin = MARGIN * len(FINAL_UIDS)
    margin = get_margin()
    print(f"Total Margin: {margin}")

    uid_config = {
        "uid_configs_tbsre_nifty": [{"uid": uid, "weight": 1} for uid in FINAL_UIDS if "NIFTY" in uid],
        "uid_configs_tbsre_sensex": [{"uid": uid, "weight": 1} for uid in FINAL_UIDS if "SENSEX" in uid],
    }

    with open("basket/configs/tbsre_basket_config.json", "w") as f:
        json.dump(uid_config, f, indent=2)

    merged_tradebook = load_and_merge_tradebooks(FINAL_UIDS, TRADEBOOK_DIR)
    if not merged_tradebook.empty:
        merged_tradebook = apply_slippage(merged_tradebook)
        metrics = compute_all_metrics(merged_tradebook, margin)
        print("\n===== FINALIZED BASKET METRICS =====")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        os.makedirs("reports/optimization/tbsre/final/", exist_ok=True)
        compute_daily_pnl(merged_tradebook).to_csv("reports/optimization/tbsre/final/daily.csv")
        metrics["monthly_pnl"].to_csv("reports/optimization/tbsre/final/monthly.csv")
else:
    print("Invalid MODE")
