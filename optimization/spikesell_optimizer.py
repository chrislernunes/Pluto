import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization_runner import run_optimization
from datetime import date

def spikesell_uid_generator(config):
    return f"spikesell_{config['underlying']}_" \
           f"{config['active_dte']}_" \
           f"{config['session']}_" \
           f"{config['delay']}_" \
           f"{config['haste']}_" \
           f"{config['timeframe']}_" \
           f"{config['offset']}_" \
           f"{config['selector']}_" \
           f"{config['selector_val']}_" \
           f"{config['hedge_shift']}_" \
           f"{config['lookback']}_" \
           f"{config['spike_thresh']}_" \
           f"{config['retrace_thresh']}_" \
           f"{config['sl_type']}_" \
           f"{config['sl_val']}_" \
           f"{config['trail_on']}_" \
           f"{config['tgt_pct']}_" \
           f"{config['lock_profit']}_" \
           f"{config['max_loss']}_" \
           f"{config['lock_profit_trigger']}_" \
           f"{config['lock_profit_to']}_" \
           f"{config['subsequent_profit_step']}_" \
           f"{config['subsequent_profit_amt']}"

def run_spikesell_optimization():
    param_grid = {
        'active_dte': [0],
        'session': ['x0', 's0', 's1', 's2'],
        'delay': [0, 15, 30, 60],
        'haste': [0],
        'timeframe': [1],
        'offset': [0],
        'selector': ['M'],
        'selector_val': [-1, 0, 1],
        'hedge_shift': [10],
        'lookback': [15, 30],
        'spike_thresh': [0.2, 0.3, 0.5],
        'retrace_thresh': [0.02, 0.05],
        'sl_type': ['PCT'],
        'sl_val': [0.25, 0.4, 0.5],
        'trail_on': [True, False],
        'tgt_pct': [0.5, 0.75, 0.9],
        'lock_profit': [False],
        'max_loss': [-1000, -2000, -3000, -5000],
        'lock_profit_trigger': [500],
        'lock_profit_to': [200],
        'subsequent_profit_step': [100],
        'subsequent_profit_amt': [50]
    }

    slice_grid = {
        'underlying': ['NIFTY', 'SENSEX']
    }

    run_optimization(
        # strategy_name='spikesell',
        param_grid=param_grid,
        slice_grid=slice_grid,
        start_date=date(2024, 1, 1),
        end_date=date(2025, 5, 31),
        tradebook_dir="storage/tradebooks/spikesell",
        meta_dir="storage/metadata/spikesell",
        metrics_file="reports/optimization/spikesell_optimization_results.csv",
        uid_generator=spikesell_uid_generator,
        n_samples=250
    )

if __name__ == "__main__":
    run_spikesell_optimization()
