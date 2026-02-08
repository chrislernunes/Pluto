import sys
from optimization.optimization_runner_pos import run_optimization
from datetime import date

def ptheta_uid_generator(config):
    return f"ptheta_{config['active_weekday']}_" \
           f"{config['session']}_" \
           f"{config['delay']}_" \
           f"{config['dte_entry']}_" \
           f"{config['underlying']}_" \
           f"{config['selector']}_" \
           f"{config['sl_pct']}_" \
           f"{config['tgt_pct']}_" \
           f"{config['max_reset']}_" \
           f"{config['trail_on']}_" \
           f"{config['Target']}_" \
           f"{config['ps']}"

def run_spikesell_optimization():
    param_grid = {
        'active_weekday' : [99],
        'session': ['x0'],
        'underlying': ['NIFTY'],
        'selector': ['P'],
        'tgt_pct': [0.99],
        'trail_on': [False],
        'Target': [False],
    }

    slice_grid = {
        'delay': [0,60,120,180,240,300],
        'dte_entry': [0,1,2,3,4],
        'sl_pct': [0.2,0.4,0.6,0.8,0.99],
        'max_reset': [0,1,2,3,4,5,6,8,10],
        'ps': [0.01,0.02,0.03,0.04,0.05]
    }
    
    # slice_grid = {
    #     'selector_val': [0.55,0.6,0.65,0.7,0.75],
    #     'af': [0.001],
    # }

    run_optimization(
        param_grid=param_grid,
        slice_grid=slice_grid,
        start_date=date(2019,7, 1),
        end_date=date(2025, 4, 1),
        tradebook_dir="storage/tradebooks/ptheta_nifty_v3",
        meta_dir="storage/metadata/ptheta_nifty_v3",
        metrics_file="reports/optimization/ptheta_nifty.csv",
        uid_generator=ptheta_uid_generator,
        n_samples=250
)

# Writing Optimization Code

if __name__ == "__main__":
    run_spikesell_optimization()
