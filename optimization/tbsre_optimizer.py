import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization_runner import run_optimization
from datetime import date

# === UID Generator for TBSRE ===
def tbsre_uid_generator(config):
    return f"tbsre_{config['underlying']}_" \
           f"{config['active_dte']}_" \
           f"{config['session']}_" \
           f"{config['delay']}_" \
           f"{config['haste']}_" \
           f"{config['timeframe']}_" \
           f"{config['offset']}_" \
           f"{config['selector']}_" \
           f"{config['selector_val']}_" \
           f"{config['hedge_shift']}_" \
           f"{config['sl_type']}_" \
           f"{config['sl_val']}_" \
           f"{config['trail_on']}_" \
           f"{config['reentry_type']}_" \
           f"{config['reentry_limit']}_" \
           f"{config['reentry_val']}_" \
           f"{config['tgt_pct']}_" \
           f"{config['reenter_on_tgt']}_" \
           f"{config['lock_profit']}_" \
           f"{config['max_loss']}_" \
           f"{config['lock_profit_trigger']}_" \
           f"{config['lock_profit_to']}_" \
           f"{config['subsequent_profit_step']}_" \
           f"{config['subsequent_profit_amt']}"


# === Main Optimization Function ===
def run_tbsre_optimization():
    param_grid = {
        'active_dte': [0],
        'delay': [0, 15, 30, 60],
        'haste': [0, 15],  
        'timeframe': [1],
        'offset': [0],
        'session': ['x0', 's0', 's1', 's2'],
        'selector': ['P'],
        'selector_val': [20, 30, 50, 70, 100],
        'hedge_shift': [10],
        'sl_type': ['PCT'],
        'sl_val': [0.3, 0.4, 0.5, 0.6, 0.8],
        'tgt_pct': [0.5, 0.7, 0.9],
        'trail_on': [False],
        'reentry_limit': [1, 2, 4],
        'reentry_val': [0.25],
        'reenter_on_tgt': [True],
        'lock_profit': [False],
        'max_loss': [-1000, -2000, -3000, -4000, -5000],
        'lock_profit_trigger': [500],
        'lock_profit_to': [-2000],
        'subsequent_profit_step': [100],
        'subsequent_profit_amt': [100]
    }

    slice_grid = {
        'underlying': ['SENSEX', 'NIFTY'],
        'reentry_type': [0, 1, 2, 3, 4]
    }

    run_optimization(
        # strategy_name='tbsre',
        param_grid=param_grid,
        slice_grid=slice_grid,
        start_date=date(2024, 1, 1),
        end_date=date(2025, 5, 31),
        tradebook_dir="storage/tradebooks/tbsre",
        meta_dir="storage/metadata/tbsre",
        metrics_file="reports/optimization/tbsre_optimization_results.csv",
        uid_generator=tbsre_uid_generator,
        n_samples=100
    )

if __name__ == "__main__":
    run_tbsre_optimization()
