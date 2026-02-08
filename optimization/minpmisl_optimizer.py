# minpmisl_optimizer_runner.py

import sys, os

from optimization_runner import run_optimization
from datetime import date

def minpmisl_uid_generator(config):
    return f"minpmisl_{config['active_weekday']}_{config['session']}_{config['delay']}_{config['timeframe']}" \
           f"_{config['underlying']}_{config['selector']}_{config['selector_val']}_{config['hedge_shift']}" \
           f"_{config['sl_pct']}_{config['tgt_pct']}_{config['max_reset']}_{config['trail_on']}" \
           f"_{config['Target']}_{config['ps']}"

def run_minpmisl_optimization():
    param_grid = {
        'session': ['x0'],
        'underlying': ['NIFTY'],
        'selector': ['P'],
        'selector_val': [75],
        'hedge_shift': [10],
        'tgt_pct': [0.5],
        'trail_on': [False],
        'Target': [False],
    }

    slice_grid = {
        'active_weekday': [0,1,2,3,4],  
        'delay': [0,120,240,300],
        'sl_pct': [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.99],
        'max_reset': [0, 1, 2, 3, 4, 5],
        'ps': [0.01, 0.02, 0.03, 0.04, 0.05],
        'timeframe': [180,360]
    }

    run_optimization(
        param_grid=param_grid,
        slice_grid=slice_grid,
        start_date=date(2019, 7, 1),
        end_date=date(2025, 8, 1),
        tradebook_dir="storage/tradebooks/minpmisl",
        meta_dir="storage/metadata/minpmisl",
        uid_generator=minpmisl_uid_generator)
    

if __name__ == "__main__":
    run_minpmisl_optimization()
