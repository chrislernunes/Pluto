import sys, os
sys.path.append('/root/research_framework')
import multiprocessing
from datetime import date

uid_list = []
metrics_list = []

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from backtest.sim_new import sim_pos_for_period
from strategies import *
from metrics.metrics import *

def run_backtest_opt(uid: str,start_date,end_date,output_dir):
    """
    Executes the backtest simulation for a given UID.
    """
    log_file = "/home/pluto/simulation_log.txt"
    # Log completion
    with open(log_file, 'a') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}: UID {uid} simulation started\n")

    # Extract strategy name and initialize
    strategy_key = uid.split('_')[0]
    strategy = strats_dict[strategy_key]()
    strategy.set_params_from_uid(uid)

    # Run the backtest
    trades = sim_pos_for_period(strategy, start_date, end_date)
    # Retrieve and save trade results
    trades_df = pd.DataFrame(trades)
    trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
    trades_df = trades_df.sort_values('timestamp', kind='mergesort').reset_index(drop=True)
    trades_df = apply_slippage(trades_df)
    trades_df = apply_brokerage(tradebook=trades_df, brokerage=0)
    print(f"Completed backtest: {uid}")
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    trades_df.to_csv(f'{output_dir}/{uid}.csv')

    
    with open(log_file, 'a') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}: UID {uid} simulation completed\n")



 

def run_optimization(param_grid, slice_grid, start_date, end_date,
                     tradebook_dir, meta_dir, metrics_file, uid_generator, n_samples=50,margin=100000):
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
    metrics_list = []
    # print(slice_combos)
    # print(sampled_param_combos)
    uids = []
    for slice_vals in slice_combos:
        for param_vals in sampled_param_combos:
            config = {**slice_vals, **param_vals}
            uidc = uid_generator(config)
            uids.append(uidc)
    total = len(uids)
    print(f"Generated UID len: {len(uids)}")
    #Check if the tradebook exists if yes then remove the UID from the list
    # Check if tradebook_dir exists and filter out existing UIDs
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
    with multiprocessing.Pool(processes=26) as pool:
        pool.map(partial(run_backtest_opt,
                     start_date=start_date,
                     end_date=end_date,output_dir=tradebook_dir),uids)   
            
    # for i in uids:
    #     file_path = os.path.join(tradebook_dir, f"{i}.csv")
    #     tb = pd.read_csv(file_path)
    #     tb['date'] = pd.to_datetime(tb['timestamp']).dt.date
    #     metrics = compute_perf_metrics(tb,margin)
    #     metrics_list.append(metrics)

    # all_results = pd.concat(metrics_list, ignore_index=True)
    # all_results.to_csv(metrics_file, index=False)
    # print(f"Saved optimization metrics to {metrics_file}")
