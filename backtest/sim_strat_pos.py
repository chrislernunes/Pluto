"""
Clean Simulation Runner for Positional Strategies
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
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing

from backtest.sim_new import sim_pos_for_period
from strategies import strats_dict
# Set dark background for any matplotlib plots (if used later)
plt.style.use('dark_background')


# Specify global date range for the simulation
START_DATE = datetime.date(2025, 1, 1)
END_DATE = datetime.date(2025, 5, 31)

# Directory to save results
OUTPUT_DIR = 'storage/tradebooks/positional/pbsardelta_nifty'

uids = [
        'pbsardelta_99_x0_0_NIFTY_D_0.7_10_0.99_0.99_False_99_0.005_0.005_5_0.0',
        'pbsardelta_99_x0_0_NIFTY_D_0.65_10_0.99_0.99_False_99_0.004_0.004_5_0.0',
        'pbsardelta_99_x0_0_NIFTY_D_0.65_10_0.99_0.99_False_99_0.0045_0.0045_5_0.0'
]

def run_backtest(uid: str):
    """
    Executes the backtest simulation for a given UID.
    """
    # Extract strategy name and initialize
    strategy_key = uid.split('_')[0]
    strategy = strats_dict[strategy_key]()
    strategy.set_params_from_uid(uid)

    # Run the backtest
    trades = sim_pos_for_period(strategy, START_DATE, END_DATE)

    # Retrieve and save trade results
    trades_df = pd.DataFrame(trades)
    print(f"Completed backtest: {uid}")
    print(trades_df)
    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trades_df.to_csv(f'{OUTPUT_DIR}/{uid}.csv')
    

if __name__ == '__main__':
    # Run all backtests in parallel
    with multiprocessing.Pool() as pool:
        pool.map(run_backtest, uids)
