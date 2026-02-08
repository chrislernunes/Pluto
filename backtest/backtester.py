import os
import sys
import argparse
import datetime
import json
import pandas as pd
from tqdm import tqdm
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backtest.sim_new import sim_for_strat
from strategies import strats_dict
from metrics.metrics import compute_all_metrics, apply_slippage, apply_brokerage

MARGIN = 100000


def generate_uid_from_dict(config: dict) -> str:
    """
    Generate UID string from config dict, assuming the order of values is the correct UID structure.

    Parameters:
    -----------
    config : dict

    Returns:
    --------
    uid : str
    """
    strategy = config["strategy"]
    del config["strategy"]
    uid = f"{strategy}_{'_'.join(str(config[k]) for k in config)}"
    print(f"Generated UID: {uid}")
    return uid


def run_simulation(
    uid: str,
    start_date: datetime.date,
    end_date: datetime.date,
    output_dir: str,
    meta_dir: str,
    max_threads: int = 15,
):
    """
    Run strategy simulation and save tradebook, metadata, and print metrics.

    Parameters:
    -----------
    uid : str
    start_date : datetime.date
    end_date : datetime.date
    output_dir : str
    meta_dir : str
    max_threads : int
    """
    strategy_name = uid.split("_")[0]
    strat_class = strats_dict[strategy_name]
    strat_uid = uid

    print(f"Running simulation for UID: {uid}")
    print(f"Initializing {strategy_name}...")

    total_days = (end_date - start_date).days + 1
    print(f"Running backtest ({total_days} days)")
    
    tb, meta_data = sim_for_strat(
        strat_class, strat_uid, start_date, end_date, strat_uid, max_threads
    )

    if tb.empty:
        print(f"[SKIPPED] No trades for UID: {uid}")
        return (pd.DataFrame(), pd.DataFrame())

    print("Processing trade data...")
    tb["date"] = tb["ts"].dt.date
    tb.rename(columns={"ts": "timestamp"}, inplace=True)
    tb = tb.sort_values("timestamp").reset_index(drop=True)

    print("Applying slippage & brokerage...")
    tb = apply_slippage(tb)
    tb = apply_brokerage(tradebook=tb, brokerage=0)

    print("Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    tb.to_csv(os.path.join(output_dir, f"{uid}.csv"), index=False)
    meta_data.to_csv(os.path.join(meta_dir, f"{uid}.csv"), index=False)
    
    print("Simulation completed")

    print(f"[DONE] UID: {uid} -> Results saved.")
    return (tb, meta_data)


def sim_metrics(tb):
    """Calculate and display metrics with progress tracking."""
    if tb.empty:
        print("\n===== PERFORMANCE METRICS =====")
        print("No trades executed - no metrics to display")
        return {}

    print("\nCalculating performance metrics...")

    with tqdm(total=2, desc="Applying adjustments", unit="step") as pbar:

        pbar.update(1)
        pbar.set_description("Computing metrics")
        metrics = compute_all_metrics(tb, MARGIN)
        pbar.update(1)

    print("\n===== PERFORMANCE METRICS =====")

    metric_items = list(metrics.items())
    with tqdm(
        metric_items, desc="Displaying metrics", unit="metric", leave=False
    ) as pbar:
        for k, v in pbar:
            pbar.set_description(f"Displaying {k}")
            if k in ["monthly_weekday_pnl", "yearly_weekday_pnl"]:
                print(f"\n{k}:")
                for month, weekday_data in v.items():
                    print(f"  {month}:")
                    for weekday, pnl in weekday_data.items():
                        print(f"    {weekday:<9}: {pnl:,.2f}")
            else:
                print(f"{k}: {v}")
            time.sleep(0.01)

    return metrics


def main(config_file: str, start_date: str, end_date: str, max_threads: int):
    """
    Load config, generate UID, and execute simulation with progress tracking.

    Parameters:
    -----------
    config_file : str
    start_date : str
    end_date : str
    max_threads : int
    """
    print("=" * 60)
    print("STRATEGY BACKTESTER STARTING")
    print("=" * 60)

    with tqdm(total=4, desc="Initializing", unit="step") as pbar:
        pbar.set_description("Parsing dates")
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        pbar.update(1)

        pbar.set_description("Loading configuration")
        with open(config_file, "r") as f:
            config = json.load(f)
        pbar.update(1)

        pbar.set_description("Generating UID")
        uid = generate_uid_from_dict(config)
        strategy_name = uid.split("_")[0]
        pbar.update(1)

        pbar.set_description("Setting up directories")
        output_dir = os.path.join("storage", "tradebooks", strategy_name)
        meta_dir = os.path.join("storage", "metadata", strategy_name)
        pbar.update(1)

    print(f"\n BACKTEST CONFIGURATION")
    print(f"   Strategy: {strategy_name}")
    print(f"   UID: {uid}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Duration: {(end_dt - start_dt).days + 1} days")
    print(f"   Threads: {max_threads}")
    print(f"   Output: {output_dir}")
    print()

    (tb, meta_data) = run_simulation(
        uid, start_dt, end_dt, output_dir, meta_dir, max_threads
    )
    sim_metrics(tb)

    print("\n" + "=" * 60)
    print("BACKTESTING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single UID JSON Strategy Backtester")
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to JSON config file"
    )
    parser.add_argument(
        "--start_date", type=str, required=True, help="Backtest start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date", type=str, required=True, help="Backtest end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--max_threads", type=int, default=30, help="Max threads for simulation"
    )

    args = parser.parse_args()
    main(args.config_file, args.start_date, args.end_date, args.max_threads)
