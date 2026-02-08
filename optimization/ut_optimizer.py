# ut_optimizer_runner.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization_runner import run_optimization
from datetime import date

def ut_uid_generator(config):
    return f"ut_{config['active_weekday']}_{config['session']}_{config['delay']}_{config['timeframe']}" \
           f"_{config['underlying']}_{config['selector']}_{config['selector_val']}" \
           f"_{config['hedge_shift']}_{config['sl_pct']}_{config['tgt_pct']}" \
           f"_{config['max_reset']}_{config['trail_on']}" \
           f"_{config['trigger_pct']}_{config['trail_pct']}"

def run_ut_optimization():
    param_grid = {
        'active_weekday': [99],  # can add [0,1,2,3,4] for weekday-specific tuning
        'session': ['x0', 's0', 's1', 's2'],
        'delay': [0, 30, 60],
        'timeframe': [1],
        'selector': ['M'],
        'selector_val': [-1, 0, 1],
        'hedge_shift': [10],
        'sl_pct': [0.3, 0.4, 0.5, 0.6],
        'tgt_pct': [0.5, 0.7, 0.9],
        'max_reset': [0, 1, 2],
        'trail_on': [True, False],  # True can be added later for trailing behavior testing
        'trigger_pct': [0.1, 0.15, 0.2, 0.25],
        'trail_pct': [0.05, 0.1, 0.15]  # useful if trail_on is True
    }

    slice_grid = {
        'underlying': ['NIFTY', 'SENSEX']
    }

    run_optimization(
        # strategy_name='ut',
        param_grid=param_grid,
        slice_grid=slice_grid,  # No slicing needed; all params are in param_grid
        start_date=date(2024, 1, 1),
        end_date=date(2025, 5, 31),
        tradebook_dir="storage/tradebooks/ut",
        meta_dir="storage/metadata/ut",
        metrics_file="reports/optimization/ut_optimization_results.csv",
        uid_generator=ut_uid_generator,
        n_samples=250
    )

if __name__ == "__main__":
    run_ut_optimization()
