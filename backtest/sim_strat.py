"""
Clean Simulation Runner for Strategy UIDs
-----------------------------------------

This script runs backtests for a list of strategy UIDs over a specified date range,
saves the resulting tradebook and metadata to designated folders.

Dependencies:
- pandas
- numpy
- direct_redis
- sim_for_strat and strats_dict from local modules

"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, time
import datetime
import pandas as pd
import direct_redis
from utils.definitions import *

if REDIS:
    from engine.ems import EventInterface
else:
    from engine.ems_db import EventInterface
    
from backtest.sim_new import sim_for_strat
from strategies import *


# Initialize Redis connection
r = direct_redis.DirectRedis(host='localhost', port=6379, db=0)

start=time.time()


def run_simulation(uid: str, start_date: datetime.date, end_date: datetime.date,
                   output_dir: str, meta_dir: str, max_threads: int = 30):
    """
    Runs the strategy simulation for the given UID and saves results.

    Parameters:
    -----------
    uid : str
        Strategy UID identifying the strategy and parameters.
    start_date : datetime.date
        Start date for backtest.
    end_date : datetime.date
        End date for backtest.
    output_dir : str
        Directory to save tradebook.
    meta_dir : str
        Directory to save metadata.
    max_threads : int
        Max threads for multiprocessing simulation.
    """
    strategy_name = uid.split('_')[0]
    strat_class = strats_dict[strategy_name]
    strat_uid = uid

    print(f"Running simulation for UID: {uid}")
    tb, meta_data = sim_for_strat(strat_class, strat_uid, start_date, end_date, strat_uid, max_threads)

    if tb.empty:
        print(f"No trades recorded for UID: {uid}")
        return

    tb['date'] = tb['timestamp'].dt.date
    tb = tb.sort_values('timestamp').reset_index(drop=True)

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # Save outputs
    tb.to_csv(os.path.join(output_dir, f"{uid}.csv"))
    meta_data.to_csv(os.path.join(meta_dir, f"{uid}.csv"))
    print(f"Simulation complete. Saved tradebook and metadata for UID: {uid}")
    

if __name__ == "__main__":
    # List of UIDs to simulate
    uids= ['tbs_99_x0_0_1_NIFTY_P_20_10_0.25_0.5_True_True',
           'ut_99_s0_0_1_NIFTY_M_-1_10_0.3_0.5_0_True_0.1_0.15',
           'spikesell_NIFTY_0_s0_0_0_1_0_M_-1_10_30_0.3_0.05_PCT_0.4_False_0.75_False_-5000_500_200_100_50']
    
    # 342.5248975753784  dkdb
    # 398.67982029914856 redis
    for uid in uids:
        # Date range for simulation
        start_date = datetime.date(2024, 7, 4)
        end_date = datetime.date(2025, 3, 1)
        strategy_name = uid.split('_')[0]

        # Output directories
        output_dir = os.path.join("storage", "tradebooks", strategy_name)
        meta_dir = os.path.join("storage", "metadata", strategy_name)
        # Ensure output directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
    
        run_simulation(uid, start_date, end_date, output_dir, meta_dir)
        

    print(time.time()-start)