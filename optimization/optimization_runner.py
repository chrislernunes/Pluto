# optimization_runner.py

import os
import random
import datetime
import pandas as pd
from typing import List, Dict, Callable
from backtest.sim_new import sim_for_strat
from strategies import strats_dict
from metrics.metrics import apply_slippage, compute_all_metrics, compute_summary_metrics
from backtest.backtester import run_simulation

# def run_simulation(uid: str,
#                   start_date: datetime.date,
#                   end_date: datetime.date,
#                  tradebook_dir: str,
#                   meta_dir: str,
#                   max_threads: int = 15):
#    """
#    Runs simulation for a given UID and saves tradebook and metadata.
#    Returns tradebook DataFrame if successful.
#    """
#    strategy_name = uid.split('_')[0]
#    strat = strats_dict[strategy_name]()
#    strat.set_params_from_uid(uid)

#    print(f"Running simulation for UID: {uid}")
#    tb, meta_data = sim_for_strat(strat, start_date, end_date, strategy_name, max_threads)

#    if tb.empty:
#        print(f"No trades for UID: {uid}")
#        return None

#    tb['timestamp'] = pd.to_datetime(tb['timestamp'])
#    tb['date'] = tb['timestamp'].dt.date
#    tb = tb.sort_values('timestamp').reset_index(drop=True)

#    os.makedirs(tradebook_dir, exist_ok=True)
#    os.makedirs(meta_dir, exist_ok=True)

#    tb.to_csv(os.path.join(tradebook_dir, f"{uid}.csv"), index=False)
#    meta_data.to_csv(os.path.join(meta_dir, f"{uid}.csv"), index=False)

#    return tb

def run_optimization(param_grid, slice_grid, start_date, end_date,
                     tradebook_dir, meta_dir, uid_generator, n_samples=50):
    """
    Generic optimization workflow that:
    - Samples parameter grid
    - Uses UID generator (strategy-specific)
    - Runs backtest and computes metrics
    - Organizes results by slice and parameter
    """
    from itertools import product
    import random

    param_keys = list(param_grid.keys())
    slice_keys = list(slice_grid.keys())

    param_combos = [dict(zip(param_keys, values)) for values in product(*param_grid.values())]
    slice_combos = [dict(zip(slice_keys, values)) for values in product(*slice_grid.values())]

    sampled_param_combos = random.sample(param_combos, min(n_samples, len(param_combos)))
   
    uids = []

    for slice_vals in slice_combos:
        for param_vals in sampled_param_combos:
            config = {**slice_vals, **param_vals}
            uidc = uid_generator(config)
            uids.append(uidc)
    total = len(uids)
    print(f"Generated UID len: {len(uids)}")
    
    print(os.path.abspath(tradebook_dir))
    print(os.path.exists(tradebook_dir))
    
    if os.path.exists(tradebook_dir):    
        existing_files = set(os.listdir(tradebook_dir))
        uids = [uid for uid in uids if f"{uid}.csv" not in existing_files]
        print(len(uids), "Uids after removing existing files")
    
    if not uids:
        print("All UIDs already exist in the tradebook directory. Exiting optimization.")
        return        
    
    print(f"Running optimization for {len(uids)} UIDs out of {total} total UIDs.")
    # Add command to pause the execution
    
    input("Press Enter to continue...")
    print("Starting optimization...")
    
    log_file = os.path.join(tradebook_dir, "optimization_log.txt")
    
    for uidc in uids:
        result = run_simulation(uidc, start_date, end_date, tradebook_dir, meta_dir, 25)
        # Log completion
        if result is not None:
            with open(log_file, 'a') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp}: UID {uidc} simulation completed\n")