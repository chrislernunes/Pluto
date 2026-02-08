import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import glob
from metrics.metrics import apply_slippage, compute_all_metrics, compute_summary_metrics

# path = "storage/tradebooks/spikesell/"  # Replace with your directory path
# metrics_file = "reports/optimization/spikesell_optimization_results.csv"

def generate_metrics_from_tradebooks(path, metrics_file, margin, slippage, param_names, prefix="minpmisl_"):
    import pandas as pd
    import os
    import glob

    csv_files = glob.glob(os.path.join(path, "*.csv"))
    metrics_list = []

    for f in csv_files:
        tb = pd.read_csv(f)
        if not tb.empty:
            tb['value'] = tb['value'] - slippage * tb['turnover'].abs()
            metrics = compute_summary_metrics(tb, margin)
            uid = os.path.splitext(os.path.basename(f))[0]
            metrics['uid'] = uid

            # Strip prefix and split
            if uid.startswith(prefix):
                uid_body = uid[len(prefix):]
            else:
                uid_body = uid
            parts = uid_body.split("_")

            if len(parts) != len(param_names):
                print(f"Warning: UID {uid} has {len(parts)} parts, expected {len(param_names)}. Skipping.")
                continue

            for name, val in zip(param_names, parts):
                metrics[name] = val

            metrics_list.append(metrics)

    all_results = pd.DataFrame(metrics_list)
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    all_results.to_csv(metrics_file, index=False)

# EXAMPLE CALL
# PARAM_NAMES = [
#    'active_weekday', 'session', 'delay', 'timeframe', 'underlying',
#    'selector', 'selector_val', 'hedge_shift', 'sl_pct', 'tgt_pct',
#    'max_reset', 'trail_on', 'Target', 'ps'
#]

#generate_metrics_from_tradebooks(
#    path="storage/tradebooks/minpmisl/",
#    metrics_file="reports/optimization/minpmisl_optimization_results.csv",
#    margin=125000,
#    slippage=0.01,
#    param_names=PARAM_NAMES,
#    prefix="minpmisl_"
#)
