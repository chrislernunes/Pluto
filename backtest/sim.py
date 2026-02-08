"""
Backtest Simulation Framework for Intraday and Positional Strategies
=====================================================================

This module runs parallel backtests on trading strategies using tick-level data from Redis.
It simulates trades over a specified date range and supports both intraday and positional modes.

Main Features:
--------------
- Parallelized backtest execution using `multiprocessing.Pool`
- Real-time-like event streaming to strategy via `.process_event()`
- Redis-based storage for trades, errors, and meta-data across dates
- Support for single-day simulations (`sim_for_date`, `sim_pos_for_date`) and period simulations (`sim_pos_for_period`)
- Intraday filtering of data between 9:07 AM and 3:30 PM
- Error logging and progress tracking through Redis counters

Redis Keys Used:
----------------
- `sims_started_<uid>` and `sims_ended_<uid>` to track simulation progress
- `trades_<uid>` to store trades per date or full period
- `errors_<uid>` for storing errors encountered during simulations
- `meta_data_<uid>` for optional metadata logging (currently commented out)

Dependencies:
-------------
- pandas, datetime, time, traceback
- direct_redis (custom Redis wrapper)
- engine.ems.EventInterface (custom event stream class)

"""

import pandas as pd
import direct_redis
import multiprocessing
from itertools import repeat
import time, datetime
import traceback
from engine.ems import EventInterface as ei

# Initialize Redis and data interface
r = direct_redis.DirectRedis(host='localhost', port=6379, db=0)
data = ei()

# Load and preprocess tick-level NIFTY spot data. This is just for the events. 
fut_data = data.get_all_ticks_by_symbol('NIFTYSPOT')
fut_data['timestamp'] = pd.to_datetime(fut_data['timestamp'])
fut_data['date'] = fut_data['timestamp'].dt.date
fut_data = fut_data[fut_data.date != datetime.date(2021, 2, 24)]  # Exclude corrupt/holiday data
fut_data = fut_data[
    (fut_data.timestamp.dt.time >= datetime.time(9, 7)) & 
    (fut_data.timestamp.dt.time < datetime.time(15, 30))
].drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)

# ...
def sim_for_strat(strat_class, strat_uid, start_date, end_date, sim_uid, max_threads=5, timeout=60): 
    """
    Run backtest simulation for a given strategy over a date range using multiprocessing.
    
    Parameters:
    -----------
    strat_class : class
        Strategy class implementing `process_event()` and containing `all_trades`, `trades`.
    strat_uid : str
        UID string for the strategy parameters.
    start_date : str or datetime.date
        Start date of simulation.
    end_date : str or datetime.date
        End date of simulation.
    sim_uid : str
        Unique ID to track simulation data in Redis.
    max_threads : int
        Number of parallel processes to use.
    timeout : int
        Time to wait before failing the simulation (not used in this implementation).

    Returns:
    --------
    tb : pd.DataFrame
        Concatenated tradebook from all dates.
    md_tb : pd.DataFrame
        Placeholder for metadata (currently not populated).
    """
    if type(start_date) is str:
        start_date = pd.to_datetime(start_date).date()
    if type(end_date) is str:
        end_date = pd.to_datetime(end_date).date()

    dates = list(pd.Series(fut_data[(fut_data.date >= start_date) & (fut_data.date <= end_date)].date).unique())
    
    uid = sim_uid

    r.delete(f'sims_started_{uid}')
    r.delete(f'sims_ended_{uid}')
    r.incr(f'sims_started_{uid}')
    r.incr(f'sims_ended_{uid}')
    r.delete(f'num_of_trades_{uid}')
    r.delete(f'errors_{uid}')
    r.delete(f'trades_{uid}')
    r.delete(f'meta_data_{uid}')
    # ...
    all_trades = []
    # ...

    print('NUM OF DATES:', len(dates))

    with multiprocessing.Pool(processes=max_threads) as pool:
        pool.starmap(sim_for_date, zip(repeat(strat_class), repeat(strat_uid), dates, repeat(uid)))
    pool.close()
    pool.join()
    # ...
    all_errors = r.hgetall(f'errors_{uid}')
    print('LIST OF ERRORS')
    for key, e in all_errors.items():
        print(key, e)
    # ...
    all_bt_trades = r.hgetall(f'trades_{uid}')
    meta_data = r.hgetall(f'meta_data_{uid}')
    # r.hdel(f'trades_{uid}')
    # r.hdel(f'meta_data_{uid}')
    r.delete(f'trades_{uid}')
    r.delete(f'meta_data_{uid}')

    all_trades = []
    for key, val in all_bt_trades.items():
        all_trades.append(pd.DataFrame(val))
    if len(all_trades) > 0:
        tb = pd.concat(all_trades, axis=0).sort_values('timestamp').reset_index(drop=True)
    else:
        tb = pd.DataFrame()
    r.delete(f'sims_started_{uid}')
    r.delete(f'sims_ended_{uid}')
    r.incr(f'sims_started_{uid}')
    r.incr(f'sims_ended_{uid}')
    r.delete(f'num_of_trades_{uid}')
    r.delete(f'errors_{uid}')
    r.delete(f'trades_{uid}')
    r.delete(f'meta_data_{uid}')
    meta_data=pd.DataFrame(meta_data)
    
    return tb, meta_data

# ...
def sim_for_date(strat_class, strat_uid, date, uid=''):
    """
    Run simulation for a single day for the given strategy.
    """
    if date.weekday() in [5, 6]:
        #print('WEEKEND')
        return
    try:
        # Instantiate the strategy and set its UID/params
        strat = strat_class()
        strat.set_params_from_uid(strat_uid)
        fut_data_of_date = fut_data[fut_data.date==date].copy()
        fut_data_of_date = fut_data_of_date.sort_values('timestamp').reset_index(drop=True)
        ########################
        for i, d in fut_data_of_date.iterrows():
            
            d['bar_complete'] = True
            d['timestamp_now'] = d['timestamp']
            # print(d)
            strat.process_event(event=d)
        strat.all_trades += strat.trades

        r.hset(f'trades_{uid}', str(date), strat.trades)
        # r.hset(f'meta_data_{uid}', str(date), [x for x in strat.meta_data if x['timestamp'].date() == date])
       
        r.incr(f'sims_ended_{uid}')
        return 
    except Exception as e:
        print(f'ERROR for {uid} on {date}', '\n', traceback.format_exc())

        err = traceback.format_exc()
        r.hset(f'errors_{uid}', str(date), err)
        
            
        r.incr(f'sims_ended_{uid}')

        return 
    
# POSITIONAL

def sim_pos_for_date(strat, date, uid=''):
    """
    Run positional simulation for a single day.
    """
    if date.weekday() in [5, 6]:
        print('WEEKEND')
        return
    try:
        fut_data_of_date = fut_data[fut_data.date==date].copy()
        fut_data_of_date = fut_data_of_date.sort_values('timestamp').reset_index(drop=True)
        ########################
        for i, d in fut_data_of_date.iterrows():
            
            d['bar_complete'] = True
            d['timestamp_now'] = d['timestamp']
            # print(d)
            strat.process_event(event=d)
        print(date, pd.DataFrame([x for x in strat.all_trades if x['timestamp'].date() == date]))
        r.hset(f'trades_{uid}', str(date), [x for x in strat.all_trades if x['timestamp'].date() == date])
        # r.hset(f'meta_data_{uid}', str(date), [x for x in strat.meta_data if x['timestamp'].date() == date])
       
        r.incr(f'sims_ended_{uid}')
        return 
    except Exception as e:
        print(f'ERROR for {uid} on {date}', '\n', traceback.format_exc())

        err = traceback.format_exc()
        r.hset(f'errors_{uid}', str(date), err)
        
            
        r.incr(f'sims_ended_{uid}')

        return 
    

def sim_pos_for_period(strat, start_date, end_date, uid=''):
    """
    Run positional simulation for a full date range (not per day).
    """
    
    fut_data_of_date = fut_data[(fut_data.date>=start_date) & (fut_data.date<=end_date)].copy()
    fut_data_of_date = fut_data_of_date.sort_values('timestamp').reset_index(drop=True)
    
    try:
        ########################
        dates = []
        for i, d in fut_data_of_date.iterrows():
            if d['timestamp'].weekday() in [5, 6]:
                continue
            
            d['bar_complete'] = True
            d['timestamp_now'] = d['timestamp']
            # print(d)
            strat.process_event(event=d)
        # print(date, pd.DataFrame([x for x in strat.all_trades if x['timestamp'].date() == date]))
        r.set(f'trades_{uid}', strat.all_trades)
        # r.set(f'meta_data_{uid}', strat.meta_data )
        
        # r.hset(f'trades_{uid}', str(date), [x for x in strat.all_trades if x['timestamp'].date() == date])
       
        r.incr(f'sims_ended_{uid}')
        return 
    except Exception as e:
        print(f'ERROR for {uid}', '\n', traceback.format_exc())

        err = traceback.format_exc()
        # r.hset(f'errors_{uid}', err)
        
            
        r.incr(f'sims_ended_{uid}')

        return 
    