import datetime
import os
import multiprocessing
import traceback
from itertools import repeat
from typing import List, Tuple, Any, Optional

import pandas as pd
from tqdm import tqdm

from engine.ems_db import EXCHANGE_MAPPING, EventInterface as ei
from utils.definitions import *
from utils.logging_config import BacktestLogger
import traceback

logger = BacktestLogger.get_logger('SimulationFramework', 'backtest/logs/simulation.log')

EXCHANGE_CONFIG = {
    "NSE": {
        "trading_start": datetime.time(9, 7),
        "trading_end": datetime.time(15, 30),
        "excluded_dates": [datetime.date(2021, 2, 24)],
    },
    "BSE": {
        "trading_start": datetime.time(9, 7),
        "trading_end": datetime.time(15, 30),
        "excluded_dates": [],
    },
    "MCX": {
        "trading_start": datetime.time(9, 0),
        "trading_end": datetime.time(23, 30),
        "excluded_dates": [],
    }
}

SELECTED_EXCHANGE = "NSE"

def _sanitize_uid_for_strategy(uid: str) -> str:
    """
    Sanitize UID by converting boolean strings to integers for strategies that expect integers
    """
    parts = uid.split('_')
    sanitized_parts = []
    
    for part in parts:
        if part == 'True':
            sanitized_parts.append('1')
        elif part == 'False':
            sanitized_parts.append('0')
        else:
            sanitized_parts.append(part)
    
    return '_'.join(sanitized_parts)

def _load_market_data() -> pd.DataFrame:
    """Load and filter market data for simulations"""
    try:
        data_interface = ei()
        
        UNDERLYING = os.getenv("PLUTO_UNDERLYING", EXCHANGE_MAPPING[SELECTED_EXCHANGE][0])
        TRADING_START_TIME = EXCHANGE_CONFIG[SELECTED_EXCHANGE]['trading_start']
        TRADING_END_TIME = EXCHANGE_CONFIG[SELECTED_EXCHANGE]['trading_end']
        EXCLUDED_DATES = EXCHANGE_CONFIG[SELECTED_EXCHANGE]['excluded_dates']
        
        market_data = data_interface.get_all_ticks_by_symbol(UNDERLYING)
        
        if market_data.empty:
            logger.error(f"No market data loaded for {UNDERLYING}")
            return pd.DataFrame()
        
        market_data['ts'] = pd.to_datetime(market_data['ts'])
        market_data['date'] = market_data['ts'].dt.date
        
        market_data = market_data[~market_data.date.isin(EXCLUDED_DATES)]
        market_data = market_data[
            (market_data.ts.dt.time >= TRADING_START_TIME) & 
            (market_data.ts.dt.time < TRADING_END_TIME)
        ].drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
        
        return market_data
        
    except Exception as e:
        logger.error(f"Failed to load market data: {e}")
        return pd.DataFrame()

fut_data = _load_market_data()

def sim_for_strat(strat_class: Any, strat_uid: str, start_date: datetime.date, 
                  end_date: datetime.date, sim_uid: str, max_threads: int = 24, 
                  timeout: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run backtest simulation for strategy over date range using multiprocessing
    
    Args:
        strat_class: Strategy class implementing process_event()
        strat_uid: Strategy parameter UID string
        start_date: Simulation start date
        end_date: Simulation end date
        sim_uid: Unique simulation identifier
        max_threads: Number of parallel processes
        timeout: Process timeout (unused)
    
    Returns:
        Tuple of (tradebook DataFrame, metadata DataFrame)
    """
    start_time = datetime.datetime.now()
    
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()

    if fut_data.empty:
        logger.error("No market data available for simulation")
        return pd.DataFrame(), pd.DataFrame()

    dates = _get_simulation_dates(start_date, end_date)
    logger.info(f"Running simulation for {len(dates)} dates with {max_threads} threads")

    try:
        # Use chunksize to batch work and reduce overhead
        chunksize = max(1, len(dates) // (max_threads * 4))
        
        with multiprocessing.Pool(processes=max_threads) as pool:
            results = list(tqdm(
                pool.starmap(sim_for_date, zip(repeat(strat_class), repeat(strat_uid), dates), chunksize=chunksize),
                total=len(dates),
                desc=f"Simulating {sim_uid}",
                unit="days"
            ))
        
        all_trades, all_meta = _consolidate_results(results)
        
        tb = pd.DataFrame(all_trades).sort_values('ts').reset_index(drop=True) if all_trades else pd.DataFrame()
        md_tb = pd.DataFrame(all_meta).sort_values('ts').reset_index(drop=True) if all_meta else pd.DataFrame()
        
        end_time = datetime.datetime.now()
        BacktestLogger.log_performance(logger, f"Strategy simulation {sim_uid}", start_time, end_time)
        
        return tb, md_tb
        
    except Exception as e:
        logger.error(f"Simulation failed for {sim_uid}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def _get_simulation_dates(start_date: datetime.date, end_date: datetime.date) -> List[datetime.date]:
    """Get list of dates for simulation within date range"""
    date_mask = (fut_data.date >= start_date) & (fut_data.date <= end_date)
    return list(fut_data[date_mask].date.unique())

def _consolidate_results(results: List[Tuple[List, List]]) -> Tuple[List, List]:
    """Consolidate results from parallel processes"""
    all_trades = []
    all_meta = []
    
    for trades, meta in results:
        if trades:
            all_trades.extend(trades)
        if meta:
            all_meta.extend(meta)
    
    return all_trades, all_meta


def sim_for_date(strat_class: Any, strat_uid: str, date: datetime.date) -> Tuple[List, List]:
    """Run simulation for single trading day"""
    if date.weekday() in [5, 6]:
        return [], []

    try:
        strategy = strat_class()
        
        try:
            strategy.set_params_from_uid(strat_uid)
        except (ValueError, IndexError, AssertionError) as uid_error:
            logger.error(f"UID parsing failed for {strat_uid} on {date}: {uid_error}")
            
            try:
                sanitized_uid = _sanitize_uid_for_strategy(strat_uid)
                logger.info(f"Attempting with sanitized UID: {sanitized_uid}")
                strategy.set_params_from_uid(sanitized_uid)
            except Exception as sanitize_error:
                logger.error(f"Sanitized UID also failed for {strat_uid}: {sanitize_error}")
                return [], []

        daily_data = fut_data[fut_data.date == date].copy()
        if daily_data.empty:
            logger.warning(f"No data available for {date}")
            return [], []

        daily_data = daily_data.sort_values('ts').reset_index(drop=True)

        for _, tick_data in daily_data.iterrows():
            event = tick_data.to_dict()
            event['bar_complete'] = True
            event['timestamp_now'] = event['ts']
            strategy.process_event(event=event)

        trades = getattr(strategy, 'trades', [])
        metadata = getattr(strategy, 'meta_data', [])

        return trades, metadata

    except Exception as e:
        context = {'date': str(date), 'strategy_uid': strat_uid}
        BacktestLogger.log_error_with_context(logger, e, context)
        return [], []
        
def sim_pos_for_date(strategy: Any, date: datetime.date) -> List:
    """Run positional simulation for single trading day"""
    if date.weekday() in [5, 6]:
        return []

    try:
        daily_data = fut_data[fut_data.date == date].copy()
        if daily_data.empty:
            logger.warning(f"No data available for positional simulation on {date}")
            return []

        daily_data = daily_data.sort_values('ts').reset_index(drop=True)

        for _, tick_data in daily_data.iterrows():
            event = tick_data.to_dict()
            event['bar_complete'] = True
            event['timestamp_now'] = event['ts']
            strategy.process_event(event=event)

        all_trades = getattr(strategy, 'all_trades', [])
        return [trade for trade in all_trades if trade['ts'].date() == date]

    except Exception as e:
        context = {'date': str(date), 'simulation_type': 'positional'}
        BacktestLogger.log_error_with_context(logger, e, context)
        return []

def sim_pos_for_period(strategy: Any, start_date: datetime.date, end_date: datetime.date) -> List:
    """Run positional simulation for entire period"""
    start_time = datetime.datetime.now()
    
    try:
        period_data = fut_data[(fut_data.date >= start_date) & (fut_data.date <= end_date)].copy()
        if period_data.empty:
            logger.warning(f"No data available for period {start_date} to {end_date}")
            return []

        period_data = period_data.sort_values('ts').reset_index(drop=True)
        logger.info(f"Processing {len(period_data)} ticks for period simulation")

        for _, tick_data in tqdm(period_data.iterrows(), total=len(period_data), desc=f"Period {start_date} to {end_date}"):
            if tick_data['ts'].weekday() in [5, 6]:
                continue
                
            event = tick_data.to_dict()
            event['bar_complete'] = True
            event['timestamp_now'] = event['ts']
            strategy.process_event(event=event)

        end_time = datetime.datetime.now()
        BacktestLogger.log_performance(logger, f"Period simulation {start_date} to {end_date}", start_time, end_time)
        
        return getattr(strategy, 'all_trades', [])

    except Exception as e:
        context = {'start_date': str(start_date), 'end_date': str(end_date), 'simulation_type': 'positional_period'}
        logger.error("Exception occurred", exc_info=True)  # full traceback via logger
        print(traceback.format_exc())  # print full traceback to console
        BacktestLogger.log_error_with_context(logger, e, context)
        return []
