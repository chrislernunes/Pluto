import sys
from optimization.optimization_runner_pos import run_optimization
from datetime import date

def psardelta_uid_generator(config):
    return f"pbsaritmpct_{config['active_weekday']}_" \
           f"{config['session']}_" \
           f"{config['delay']}_" \
           f"{config['underlying']}_" \
           f"{config['selector']}_" \
           f"{config['selector_val']}_" \
           f"{config['hedge_shift']}_" \
           f"{config['sl_pct']}_" \
           f"{config['tgt_pct']}_" \
           f"{config['trail_on']}_" \
           f"{config['max_reset']}_" \
           f"{config['af']}_" \
           f"{config['af']}_" \
           f"{config['timeframe']}_" \
           f"{config['itmperc']}"

def run_spikesell_optimization():
    param_grid = {
        'session': ['x0'],
        'delay': [0],
        'active_weekday' : [99],
        'underlying': ['NIFTY'],
        'selector': ['D'],
        'hedge_shift': [10],
        'sl_pct': [0.99],
        'tgt_pct': [0.99],
        'trail_on': [False],
        'max_reset': [99],
        'timeframe': [5],
        'itmperc': [0.0]
    }

    slice_grid = {
        'selector_val': [0.5,1.0],
        'af': [0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.005,0.01,0.015,0.02,0.025],
    }
    
    # slice_grid = {
    #     'selector_val': [0.55,0.6,0.65,0.7,0.75],
    #     'af': [0.001],
    # }

    run_optimization(
        param_grid=param_grid,
        slice_grid=slice_grid,
        start_date=date(2019,7, 1),
        end_date=date(2025,8, 1),
        tradebook_dir="storage/tradebooks/pbsaritm_nifty_monthly",
        meta_dir="storage/metadata/pbsaritm_nifty_monthly",
        metrics_file="reports/optimization/pbsaritm_nifty.csv",
        uid_generator=psardelta_uid_generator,
        n_samples=250
    )

# Writing Optimiztion Code

if __name__ == "__main__":
    run_spikesell_optimization()
