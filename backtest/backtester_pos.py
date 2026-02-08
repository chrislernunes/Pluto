"""
JSON-based Strategy Backtester (Positional)
--------------------------------------------

Reads a single strategy config from JSON, generates the UID, and runs the backtest for positional strategies.
Includes slippage-adjusted performance metrics and clean printing.

Usage:
$ python backtester_positional.py --config_file configs/btstdir_config.json --start_date 2025-01-01 --end_date 2025-05-31
"""

import os
import sys
import argparse
import datetime
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest.sim_new import sim_pos_for_period
from strategies import strats_dict
import direct_redis
from metrics.metrics import apply_slippage, apply_brokerage,compute_all_metrics, compute_perf_metrics

r = direct_redis.DirectRedis(host='localhost', port=6379, db=0)


def generate_uid_from_dict(config: dict) -> str:
    strategy = config["strategy"]
    del config["strategy"]
    uid = f"{strategy}_{'_'.join(str(config[k]) for k in config)}"
    print("Generated UID:", uid)
    return uid


def run_simulation(uid: str, start_date: datetime.date, end_date: datetime.date,
                   output_dir: str, margin: float = 125000):
    strategy_name = uid.split('_')[0]
    strat = strats_dict[strategy_name]()
    strat.set_params_from_uid(uid)

    print(f"\nRunning positional simulation for UID: {uid}")

    trades = sim_pos_for_period(strat, start_date, end_date)

    if not trades:
        print(f"[SKIPPED] No trades for UID: {uid}")
        return

    tradebook = pd.DataFrame(trades)
    tradebook['date'] = pd.to_datetime(tradebook['ts']).dt.date
    tradebook = tradebook.sort_values('ts', kind='mergesort').reset_index(drop=True)

    tradebook = apply_slippage(tradebook)
    tradebook = apply_brokerage(tradebook=tradebook, brokerage=0)
    metrics = compute_all_metrics(tradebook, margin=margin)

    os.makedirs(output_dir, exist_ok=True)
    tradebook.to_csv(os.path.join(output_dir, f"{uid}.csv"))

    print("\n===== Performance Metrics =====")
    for k, v in metrics.items():
        if k in ["monthly_weekday_pnl", "yearly_weekday_pnl"]:
            print(f"\n{k}:")
            for period, weekday_data in v.items():
                print(f"  {period}:")
                for weekday, pnl in weekday_data.items():
                    print(f"    {weekday:<9}: {pnl:,.2f}")
        else:
            print(f"{k}: {v}")

    metrics_df = compute_perf_metrics(tradebook, margin=margin)        
    return metrics_df

def main(config_file: str, start_date: str, end_date: str):
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    with open(config_file, 'r') as f:
        config = json.load(f)

    uid = generate_uid_from_dict(config)
    strategy_name = uid.split('_')[0]
    output_dir = os.path.join("storage", "tradebooks", "positional", strategy_name)

    run_simulation(uid, start_dt, end_dt, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Positional Strategy Backtester (JSON Config)")
    parser.add_argument('--config_file', type=str, required=True, help='Path to JSON config file')
    parser.add_argument('--start_date', type=str, required=True, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='Backtest end date (YYYY-MM-DD)')

    args = parser.parse_args()
    main(args.config_file, args.start_date, args.end_date)
