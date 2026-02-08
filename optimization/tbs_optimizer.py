import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# tbs_optimizer_runner.py
from optimization_runner import run_optimization
import datetime

def tbs_uid_generator(config):
    return f"tbs_0_{config['session']}_{config['delay']}_{config['timeframe']}" \
           f"_{config['underlying']}_{config['selector']}_{config['selector_val']}" \
           f"_{config['hedge_shift']}_{config['sl_pct']}_{config['tgt_pct']}" \
           f"_{config['trail_on']}_{config['Target']}"


def run_tbs_optimization():
    from datetime import date
    from optimization_runner import run_optimization

    param_grid = {
        'delay': [0, 15, 30, 60],
        'timeframe': [1],
        'session': ['x0', 's0', 's1', 's2'],
        'selector': ['P'],
        'selector_val': [20, 30, 50, 70, 100],
        'hedge_shift': [10],
        'sl_pct': [0.25, 0.35, 0.5, 0.7],
        'tgt_pct': [0.5, 0.7, 0.9],
        'trail_on': [True, False],
        'Target': [True, False]
    }

    slice_grid = {
        'underlying': ['NIFTY', 'SENSEX']
    }

    run_optimization(
        # strategy_name='tbs',
        param_grid=param_grid,
        slice_grid=slice_grid,
        start_date=date(2024, 1, 1),
        end_date=date(2025, 5, 31),
        tradebook_dir="storage/tradebooks/tbs",
        meta_dir="storage/metadata/tbs",
        metrics_file="reports/optimization/tbs_optimization_results.csv",
        uid_generator=tbs_uid_generator,
        n_samples=20
    )


if __name__ == "__main__":
    run_tbs_optimization()

