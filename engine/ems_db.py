import datetime
from typing import Optional, Dict, List, Tuple, Union

import duckdb
import numpy as np
import pandas as pd

# from cython_modules.payoff import single_strike_payoff_cal
from utils.logging_config import BacktestLogger
from utils.utility import holidays, stock_tickers

DB_PATH = "/mnt/disk2/bt.db"

lot_size_dict = {
    "BANKNIFTY": 15,
    "NIFTY": 75,
    "FINNIFTY": 65,
    "MIDCPNIFTY": 50,
    "SENSEX": 20,
    "SPXW": 100,
    "NDXP": 100,
    "GOLDM": 1,
}

freeze_qty_dict = {
    "BANKNIFTY": 900,
    "NIFTY": 1800,
    "FINNIFTY": 1800,
    "MIDCPNIFTY": 4200,
    "SENSEX": 1000,
    "SPXW": 99999,
    "NDXP": 99999,
}

strike_diff_dict = {
    "BANKNIFTY": 100,
    "NIFTY": 50,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
    "SENSEX": 100,
    "BANKEX": 100,
    "SPXW": 5,
    "NDXP": 10,
    "GOLDM": 1000,
}

indexes = ("BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTY", "SENSEX", "BANKEX", "SPXW", "NDXP", "GOLDM")

EXCHANGE_MAPPING = {            
    "BSE": ["SENSEX","BANKEX"],
    "NSE": ["NIFTY","BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"],
    "CBOE": ["SPXW"],
    "NASDAQ": ["NDXP"],
    "MCX": ["GOLDM"]
}


def get_underlying(symbol: str) -> Optional[str]:
    """Extract underlying from symbol string"""
    for underlying in indexes:
        if symbol.startswith(underlying):
            return underlying
    return None


class DataInterface:
    def __init__(self, conn: Optional[duckdb.DuckDBPyConnection] = None):
        self._conn = conn
        self.logger = BacktestLogger.get_logger(
            "DataInterface", "backtest/logs/data_interface.log"
        )
        self.dict_of_expiries = {}
        for i in indexes:
            exchange = self.get_exchange(i)
            try:
                tables_query = f"""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name LIKE '{exchange}_Options_Expiry_{i}\\_%' ESCAPE '\\'
                """
                tables_df = self.conn.execute(tables_query).fetchdf()
                expiries = []
                for table_name in tables_df["table_name"]:
                    parts = table_name.split("_")
                    if len(parts) >= 4:
                        expiry_str = parts[4]
                        try:
                            expiry_date = datetime.datetime.strptime(expiry_str, "%Y%m%d")
                            expiries.append(expiry_date)
                        except ValueError:
                            continue
                self.dict_of_expiries[i] = sorted(list(set(expiries)))
            except Exception as e:
                print(f"Error loading expiry dates for {i}: {e}")
                self.dict_of_expiries[i] = []

    def get_exchange(self, underlying: str) -> str:
        """Get exchange for given underlying"""
        for exchange, underlyings in EXCHANGE_MAPPING.items():
            if underlying in underlyings:
                return exchange
        return "NSE"

    def _get_conn(self):
        if self._conn is None:
            try:
                self._conn = duckdb.connect(DB_PATH, read_only=True, config={"memory_limit": "1GB"})
            except Exception as e:
                print(f"Error connecting to database: {e}")
                self._conn = None
        return self._conn

    @property
    def conn(self):
        if self._conn is None:
            return self._get_conn()
        try:
            self._conn.execute("SELECT 1")
        except Exception:
            print("DuckDB connection lost, reconnecting...")
            self._conn = None
            return self._get_conn()
        return self._conn

    def get_lot_size(self, underlying):
        return lot_size_dict.get(underlying)

    def get_freeze_quantity(self, underlying):
        return freeze_qty_dict.get(underlying)

    def get_strike_diff(self, underlying):
        return strike_diff_dict.get(underlying)

    def get_expiry_dates(self, underlying):
        try:
            list_of_expiries = self.dict_of_expiries[underlying]
            return sorted(list_of_expiries)
        except Exception as e:
            print(f"Error fetching expiry dates: {e}")
            return []

    def get_expiry_code(self, timestamp, underlying, expiry_idx):
        expiries = self.get_expiry_dates(underlying)
        future_expiries = [d for d in expiries if d >= pd.Timestamp(timestamp.date())]

        if not future_expiries:
            raise ValueError("No upcoming expiries found")

        expiry_date = future_expiries[expiry_idx]
        return expiry_date.strftime('%y%m%d')

    def get_monthly_expiry_code(self, timestamp, underlying):
        expiries = self.get_expiry_dates(underlying)
        monthly_expiries = [d for d in expiries if d.month == timestamp.month and d.year == timestamp.year]

        if not monthly_expiries:
            raise ValueError("No monthly expiries found")

        expiry_date = sorted(monthly_expiries)[-1]
        return expiry_date.strftime('%y%m%d')

    def parse_date_from_symbol(self, symbol):
        for underlying in indexes + tuple(stock_tickers):
            if symbol.startswith(underlying):
                break

        symbol_wo_underlying = symbol[len(underlying):]
        date_part = symbol_wo_underlying[:6]

        try:
            expiry_date = datetime.datetime.strptime(date_part, "%y%m%d")
            return expiry_date.date()
        except ValueError:
            raise ValueError(f"Invalid expiry format in symbol: {symbol}")

    def parse_strike_from_symbol(self, symbol):
        for underlying in indexes + tuple(stock_tickers):
            if symbol.startswith(underlying):
                break

        symbol_wo_underlying = symbol[len(underlying):]
        remainder = symbol_wo_underlying[6:].rstrip('CE').rstrip('PE')

        try:
            return int(remainder)
        except ValueError:
            raise ValueError(f"Could not parse strike from symbol: {symbol}")

    def find_symbol_by_moneyness(self, timestamp, underlying, expiry_idx, opt_type, otm_count):
        strike_diff = self.get_strike_diff(underlying)
        shifter = 1 if opt_type == 'CE' else -1
        exchange = self.get_exchange(underlying)

        spot = self.get_tick(timestamp, f"{underlying}SPOT")
        spot_price = spot['c'] if spot is not None else 0

        atm_strike = round(spot_price / strike_diff) * strike_diff
        selected_strike = int(atm_strike + otm_count * shifter * strike_diff)

        expiry_code = self.get_expiry_code(timestamp, underlying, expiry_idx)
        symbol = f"{underlying}{expiry_code}{selected_strike}{opt_type}"

        opt_type_db = 'call' if 'CE' in symbol else 'put'
        expiry = self.parse_date_from_symbol(symbol).strftime('%Y%m%d')
        strike = self.parse_strike_from_symbol(symbol)

        table = f"{exchange}_Options_Expiry_{underlying}_{expiry}"

        if isinstance(timestamp, datetime.datetime):
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = str(timestamp)

        result = self.conn.execute(
                f"SELECT * FROM {table} WHERE strike = {strike} AND option_type = '{opt_type_db}' AND ts = '{timestamp_str}' LIMIT 1"
            ).fetchdf()

        if not result.empty:    
            return symbol
        else:
            print("Moneyness Symbol Not found in DB")
            return None

    def find_symbol_by_premium(self, timestamp, underlying, expiry_idx, opt_type, seek_price,
                           seek_type=None, premium_rms_thresh=1, moneyness_rms_thresh=-10,
                           oi_thresh=1000, perform_rms_checks=False, force_atm=False):

        try:
            expiry_code = self.get_expiry_code(timestamp, underlying, expiry_idx)
            expiry_code_str = str(expiry_code)

            if len(expiry_code_str) == 8:
                expiry_date = datetime.datetime.strptime(expiry_code_str, "%Y%m%d")
            elif len(expiry_code_str) == 6:
                expiry_date = datetime.datetime.strptime(expiry_code_str, "%y%m%d")
            else:
                raise ValueError("Invalid expiry code format")

            exchange = self.get_exchange(underlying)

            spot_tick = self.get_tick(timestamp, f"{underlying}SPOT")
            if spot_tick is None or spot_tick['c'] is np.nan:
                return None

            spot_price = float(spot_tick['c'])
            strike_diff = strike_diff_dict[underlying]
            atm_strike = round(spot_price / strike_diff) * strike_diff
            shifter = 1 if opt_type == 'CE' else -1

            table = f"{exchange}_Options_Expiry_{underlying}_{expiry_date.strftime('%Y%m%d')}"
            df = self.conn.execute(f"""
                SELECT * FROM {table}
                WHERE option_type = '{'call' if opt_type == 'CE' else 'put'}'
                AND ts = '{timestamp.strftime('%Y-%m-%d %H:%M:%S')}'
            """).fetchdf()

            if df.empty:
                return None

            df = df[df['c'] > 0]

            if seek_type == 'gt':
                df = df[df['c'] > seek_price]
            elif seek_type == 'lt':
                df = df[df['c'] < seek_price]

            if df.empty:
                return None

            df['premium_diff'] = (df['c'] - seek_price).abs()
            closest = df.sort_values('premium_diff').iloc[0]

            strike = int(closest['strike'])
            symbol = f"{underlying}{expiry_code}{strike}{opt_type}"

            if perform_rms_checks and seek_price > 5:
                tick = self.get_tick(timestamp, symbol)
                if tick is None or tick['c'] is np.nan:
                    return None

                prem = tick['c']
                oi = tick['oi']

                moneyness = shifter * (strike - atm_strike) / strike_diff

                if prem > seek_price * (1 + premium_rms_thresh) or \
                   oi < oi_thresh * self.get_lot_size(underlying) or \
                   moneyness < moneyness_rms_thresh:
                    symbol = None

                if force_atm and moneyness <= -1:
                    symbol = self.find_symbol_by_moneyness(timestamp, underlying, expiry_idx, opt_type, 0)

                if symbol is None:
                    temp_symbol = f"{underlying}{expiry_code}{strike + strike_diff * shifter}{opt_type}"
                    tick = self.get_tick(timestamp, temp_symbol)
                    if tick is not None and tick['c'] <= seek_price * (1 + premium_rms_thresh) and \
                       tick['oi'] >= oi_thresh * self.get_lot_size(underlying):
                        symbol = temp_symbol
                    else:
                        symbol = None

            return symbol

        except Exception as e:
            print(f"Error finding symbol by premium: {e}")
            return None

    def get_tick(self, timestamp, symbol):
        if isinstance(timestamp, datetime.datetime):
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = str(timestamp)

        try:
            instrument = 'Index' if not any(char.isdigit() for char in symbol) else 'Options'
            underlying = next((u for u in indexes if symbol.startswith(u)), None)
            exchange = self.get_exchange(underlying)

            if instrument == 'Options':
                expiry = self.parse_date_from_symbol(symbol).strftime('%Y%m%d')
                strike = self.parse_strike_from_symbol(symbol)
                opt_type = 'call' if 'CE' in symbol else 'put'

            if 'SPOT' in symbol:
                underlying = underlying.replace('SPOT', '')

            if instrument == 'Options':
                table = f"{exchange}_Options_Expiry_{underlying}_{expiry}"
                result = self.conn.execute(
                    f"SELECT * FROM {table} WHERE ts = '{timestamp_str}' AND strike = {strike} AND option_type = '{opt_type}' LIMIT 1"
                ).fetchdf()
            else:
                table = f"{exchange}_Index_{underlying}"
                result = self.conn.execute(
                    f"SELECT * FROM {table} WHERE ts = '{timestamp_str}' LIMIT 1"
                ).fetchdf()

            if not result.empty:
                return result.iloc[0]

            if instrument == 'Options':
                result = self.conn.execute(
                    f"SELECT * FROM {table} WHERE ts > '{timestamp_str}' AND strike = {strike} AND option_type = '{opt_type}' ORDER BY ts ASC LIMIT 1"
                ).fetchdf()
            else:
                result = self.conn.execute(
                    f"SELECT * FROM {table} WHERE ts > '{timestamp_str}' ORDER BY ts ASC LIMIT 1"
                ).fetchdf()

            if not result.empty:
                return result.iloc[0]

            tick = self.get_last_available_tick(symbol)
            if tick is not None:
                return tick

            return None

        except Exception as e:
            print(f"Couldn't find {symbol} tick in TIME AXIS, trying next available tick {timestamp}")
            try:
                all_ticks = self.get_all_ticks_by_symbol(symbol)
                all_ticks['ts'] = pd.to_datetime(all_ticks['ts'])
                tick = all_ticks[all_ticks['ts'] > timestamp].iloc[0]
            except Exception as e:
                print(f'GETTICK ERR: SYMBOL NOT AVAILABLE {symbol}, {timestamp} {e}')
                tick = {'o': np.nan, 'h': np.nan, 'l': np.nan, 'c': np.nan, 'v': np.nan, 'oi': np.nan, 'ts': None}

            return tick

    def get_nearest_expiry(self, timestamp, underlying):
        try:
            exchange = self.get_exchange(underlying)
            expiries = self.get_expiry_dates(underlying)
            future_expiries = [d for d in expiries if d.date() >= timestamp.date()]

            if not future_expiries:
                return None

            return sorted(future_expiries)[0].date()

        except Exception as e:
            print(f"Error fetching nearest expiry: {e}")
            return None

    def get_last_available_tick(self, symbol):
        try:
            instrument = 'Index' if not any(char.isdigit() for char in symbol) else 'Options'
            underlying = next((u for u in indexes if symbol.startswith(u)), None)
            exchange = self.get_exchange(underlying)

            if instrument == 'Options':
                expiry = self.parse_date_from_symbol(symbol).strftime('%Y%m%d')
                strike = self.parse_strike_from_symbol(symbol)
                opt_type = 'call' if 'CE' in symbol else 'put'

            if 'SPOT' in symbol:
                underlying = underlying.replace('SPOT', '')

            if instrument == 'Options':
                table = f"{exchange}_Options_Expiry_{underlying}_{expiry}"
                result = self.conn.execute(f"SELECT * FROM {table} WHERE strike = {strike} AND option_type = '{opt_type}' ORDER BY ts DESC LIMIT 1").fetchdf()
            else:
                table = f"{exchange}_Index_{underlying}"
                result = self.conn.execute(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT 1").fetchdf()

            if not result.empty:
                return result.iloc[0]

            return None

        except Exception as e:
            print(f"Error fetching last tick for {symbol}: {e}")
            return None

    def get_all_ticks_by_symbol(self, symbol):
        try:
            instrument = 'Index' if not any(char.isdigit() for char in symbol) else 'Options'
            underlying = next((u for u in indexes if symbol.startswith(u)), None)
            exchange = self.get_exchange(underlying)

            if instrument == 'Options':
                expiry = self.parse_date_from_symbol(symbol).strftime('%Y%m%d')
                strike = self.parse_strike_from_symbol(symbol)
                opt_type = 'call' if 'CE' in symbol else 'put'

            if 'SPOT' in symbol:
                underlying = underlying.replace('SPOT', '') 

            if instrument == 'Options':
                table = f"{exchange}_Options_Expiry_{underlying}_{expiry}"
                df = self.conn.execute(f"SELECT * FROM {table} WHERE strike = {strike} AND option_type = '{opt_type}'").fetchdf()
            else:
                table = f"{exchange}_Index_{underlying}"
                df = self.conn.execute(f"SELECT * FROM {table}").fetchdf()

            df['ts'] = pd.to_datetime(df['ts'])
            return df.sort_values('ts').reset_index(drop=True)

        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()

    def get_ticks_of_symbol_between_timestamps(self, symbol, from_timestamp=None, to_timestamp=None):
        df = self.get_all_ticks_by_symbol(symbol)
        if from_timestamp:
            df = df[df['ts'] >= pd.to_datetime(from_timestamp)]
        if to_timestamp:
            df = df[df['ts'] <= pd.to_datetime(to_timestamp)]

        return df

    def get_dte(self, timestamp, symbol):
        expiry_date = self.parse_date_from_symbol(symbol)
        start_date = timestamp.date()

        days_list = [
            start_date + datetime.timedelta(days=i)
            for i in range((expiry_date - start_date).days)
            if (start_date + datetime.timedelta(days=i)).weekday() < 5 and (start_date + datetime.timedelta(days=i)) not in holidays
        ]

        dte = len(days_list)

        if dte < 0:
            print(f'WARNING: {symbol} Expired')
            dte = 69

        return int(dte)

    def get_dte_by_underlying(self, timestamp, underlying):
        date_part = self.get_expiry_code(timestamp, underlying, 0)
        expiry_date = datetime.datetime.strptime(date_part, "%y%m%d").date()
        start_date = timestamp.date()

        days_list = [
            start_date + datetime.timedelta(days=i)
            for i in range((expiry_date - start_date).days)
            if (start_date + datetime.timedelta(days=i)).weekday() < 5 and (start_date + datetime.timedelta(days=i)) not in holidays
        ]

        dte = len(days_list)

        if dte < 0:
            print(f'WARNING: {underlying} Expired')
            dte = 69

        return int(dte)

    def replace_strike_in_symbol(self, symbol, strike):
        current_strike = self.parse_strike_from_symbol(symbol)
        return symbol.replace(str(current_strike), str(strike))

    def shift_strike_in_symbol(self, symbol, shift_otm_count):
        strike = self.parse_strike_from_symbol(symbol)

        for underlying in indexes + tuple(stock_tickers):
            if symbol.startswith(underlying):
                break

        strike_diff = self.get_strike_diff(underlying)
        opt_type = symbol[-2:]
        shifter = 1 if opt_type == 'CE' else -1

        new_strike = int(float(strike + shift_otm_count * shifter * strike_diff))

        return symbol.replace(str(strike), str(new_strike))

    def get_delta_by_symbol(self,timestamp,symbol):
        underlying =  get_underlying(symbol)
        exchange = self.get_exchange(underlying)
        expiry = self.parse_date_from_symbol(symbol).strftime('%Y%m%d')
        strike = self.parse_strike_from_symbol(symbol)
        opt_type = 'call' if 'CE' in symbol else 'put'
        table = f"{exchange}_Options_Expiry_{underlying}_{expiry}"

        df = self.conn.execute(f"""
        SELECT delta FROM {table}
        WHERE strike = {strike}
        AND option_type = '{opt_type}'
        AND ts = '{timestamp.strftime('%Y-%m-%d %H:%M:%S')}' """).fetchdf()

        return float(df['delta'])

    def find_symbol_by_delta(self, timestamp, underlying, expiry_idx, opt_type, seek_delta, seek_type=None,volume_filter=None) :
        try:
            expiry_code = self.get_expiry_code(timestamp, underlying, expiry_idx)
            expiry_code_str = str(expiry_code)

            if len(expiry_code_str) == 8:
                expiry_date = datetime.datetime.strptime(expiry_code_str, "%Y%m%d")
            elif len(expiry_code_str) == 6:
                expiry_date = datetime.datetime.strptime(expiry_code_str, "%y%m%d")
            else:
                raise ValueError("Invalid expiry code format")

            exchange = self.get_exchange(underlying)

            table = f"{exchange}_Options_Expiry_{underlying}_{expiry_date.strftime('%Y%m%d')}"
            subset_data = self.conn.execute(f"""
                    SELECT * FROM {table}
                    WHERE option_type = '{'call' if opt_type == 'CE' else 'put'}'
                    AND ts = '{timestamp.strftime('%Y-%m-%d %H:%M:%S')}'
                """).fetchdf()

            if volume_filter is not None:
                subset_data = subset_data[subset_data['v']>volume_filter]

            if subset_data.empty:
                return None,None

            if seek_type == None:
                subset_data['diff'] = abs(subset_data['delta'] - seek_delta)
                strike = int(subset_data.loc[subset_data['diff'].idxmin(),"strike"])
                delta = float(subset_data.loc[subset_data['diff'].idxmin(),"delta"])

            elif seek_type == 'gt':
                eligible_data = subset_data[subset_data['delta'] >= seek_delta]
                if len(eligible_data) > 0:
                    strike = int(eligible_data.loc[eligible_data['delta'].idxmin(),"strike"])
                    delta = float(eligible_data.loc[eligible_data['delta'].idxmin(),"delta"])
                else:
                    subset_data['diff'] = abs(subset_data['delta'] - seek_delta)
                    strike = int(subset_data.loc[subset_data['diff'].idxmin(),"strike"])
                    delta = float(subset_data.loc[subset_data['diff'].idxmin(),"delta"])

            elif seek_type == 'lt':
                eligible_data = subset_data[subset_data['delta'] <= seek_delta]
                if len(eligible_data) > 0:
                    strike = int(eligible_data.loc[eligible_data['delta'].idxmax(),"strike"])
                    delta = float(eligible_data.loc[eligible_data['delta'].idxmax(),"delta"])
                else:
                    subset_data['diff'] = abs(subset_data['delta'] - seek_delta)
                    strike = int(subset_data.loc[subset_data['diff'].idxmin(),"strike"])
                    delta = float(subset_data.loc[subset_data['diff'].idxmin(),"delta"])

            symbol = f"{underlying}{expiry_code}{strike}{opt_type}"
            return symbol,delta

        except Exception as e:
            print(f"Error finding symbol by delta: {e}")
            return None,None

    def find_symbol_by_itm_percent(self, timestamp, underlying, expiry_idx, opt_type, itm_percent, volume_filter=None, default_to_atm=False):
        try:
            spot_price = self.get_tick(timestamp, f"{underlying}SPOT")['c']

            if spot_price is None:
                print("Could not get spot price")
                return None, None
                        
            expiry_code = self.get_expiry_code(timestamp, underlying, expiry_idx)
            expiry_code_str = str(expiry_code)

            if len(expiry_code_str) == 8:
                expiry_date = datetime.datetime.strptime(expiry_code_str, "%Y%m%d")
            elif len(expiry_code_str) == 6:
                expiry_date = datetime.datetime.strptime(expiry_code_str, "%y%m%d")
            else:
                raise ValueError("Invalid expiry code format")

            exchange = self.get_exchange(underlying)

            table = f"{exchange}_Options_Expiry_{underlying}_{expiry_date.strftime('%Y%m%d')}"
            subset_data = self.conn.execute(f"""
            SELECT * FROM {table}
            WHERE option_type = '{'call' if opt_type == 'CE' else 'put'}'
            AND ts = '{timestamp.strftime('%Y-%m-%d %H:%M:%S')}'
                                                                        """).fetchdf()
        
            if volume_filter is not None:
                subset_data = subset_data[subset_data['v'] > volume_filter]
        
            if subset_data.empty:
                if default_to_atm:
                    symbol = self.find_symbol_by_moneyness(timestamp, underlying, expiry_idx, opt_type, 0)    
                    return symbol, 999
                else:
                    return None, None

            if opt_type == 'CE':
                subset_data['itm_percent'] = ((spot_price - subset_data['strike']) / spot_price * 100)
                eligible_data = subset_data[
                    (subset_data['strike'] <= spot_price) &
                    (subset_data['itm_percent'] <= itm_percent)
                ]
            else:
                subset_data['itm_percent'] = ((subset_data['strike'] - spot_price) / spot_price * 100)
                eligible_data = subset_data[
                    (subset_data['strike'] >= spot_price) &
                    (subset_data['itm_percent'] <= itm_percent)
                ]

            if eligible_data.empty:
                print(f"No strikes found with ITM% <= {itm_percent}%")
                if default_to_atm:
                    symbol = self.find_symbol_by_moneyness(timestamp, underlying, expiry_idx, opt_type, 0)
                    return symbol, 999
                else:
                    return None, None

            selected_row = eligible_data.loc[eligible_data['itm_percent'].idxmax()]
            strike = int(selected_row['strike'])
            actual_itm_percent = float(selected_row['itm_percent'])
            delta = float(selected_row['delta'])

            symbol = f"{underlying}{expiry_code}{strike}{opt_type}"

            return symbol, delta

        except Exception as e:
            print(f"Error finding symbol by ITM percent: {e}")
            return None, None    

    def find_symbol_by_itm_percent_v2(self, timestamp, underlying, expiry_idx, opt_type, itm_percent, volume_filter=None):
        try:
            spot_price = self.get_tick(timestamp, f"{underlying}SPOT")['c']

            if spot_price is None:
                print("Could not get spot price")
                return None, None
                        
            expiry_code = self.get_expiry_code(timestamp, underlying, expiry_idx)
            expiry_code_str = str(expiry_code)

            if len(expiry_code_str) == 8:
                expiry_date = datetime.datetime.strptime(expiry_code_str, "%Y%m%d")
            elif len(expiry_code_str) == 6:
                expiry_date = datetime.datetime.strptime(expiry_code_str, "%y%m%d")
            else:
                raise ValueError("Invalid expiry code format")

            exchange = self.get_exchange(underlying)

            table = f"{exchange}_Options_Expiry_{underlying}_{expiry_date.strftime('%Y%m%d')}"
            subset_data = self.conn.execute(f"""
            SELECT * FROM {table}
            WHERE option_type = '{'call' if opt_type == 'CE' else 'put'}'
            AND ts = '{timestamp.strftime('%Y-%m-%d %H:%M:%S')}'
                                                                        """).fetchdf()

            if volume_filter is not None:
                subset_data = subset_data[subset_data['v'] > volume_filter]

            if subset_data.empty:
                return None, None

            if opt_type == 'CE':
                target_strike = spot_price * (1 - itm_percent/100)
            else:
                target_strike = spot_price * (1 + itm_percent/100)

            subset_data['strike_diff'] = abs(subset_data['strike'] - target_strike)
            selected_row = subset_data.loc[subset_data['strike_diff'].idxmin()]

            strike = int(selected_row['strike'])
            actual_moneyness_percent = float(selected_row['moneyness_percent']) if 'moneyness_percent' in selected_row else None
            delta = float(selected_row['delta'])

            symbol = f"{underlying}{expiry_code}{strike}{opt_type}"

            return symbol, delta

        except Exception as e:
            print(f"Error finding symbol by target strike: {e}")
            return None, None        
    
    def find_symbol_by_moneyness_attempt(self, timestamp, underlying, expiry_idx, opt_type, otm_count, max_attempts=100):
        strike_diff = self.get_strike_diff(underlying)
        shifter = 1 if opt_type == 'CE' else -1
        exchange = self.get_exchange(underlying)
        
        spot = self.get_tick(timestamp, f"{underlying}")
        spot_price = spot['c'] if spot is not None else 0
        if spot_price == 0:
            print(f"[WARN] Spot price not available for {underlying} at {timestamp}")
            return None

        atm_strike = round(spot_price / strike_diff) * strike_diff
        expiry_code = self.get_expiry_code(timestamp, underlying, expiry_idx)

        for attempt in range(max_attempts):
            adjusted_otm_count = otm_count - attempt if otm_count >= 0 else otm_count + attempt
            selected_strike = int(atm_strike + adjusted_otm_count * shifter * strike_diff)
            symbol = f"{underlying}{expiry_code}{selected_strike}{opt_type}"

            # Determine option type
            opt_type_str = 'call' if 'CE' in symbol else 'put'
            expiry = self.parse_date_from_symbol(symbol).strftime('%Y%m%d')
            strike = self.parse_strike_from_symbol(symbol)
            table = f"{exchange}_Options_Expiry_{underlying}_{expiry}"

            if isinstance(timestamp, datetime.datetime):
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_str = str(timestamp)
            
            try:
                result = self.conn.execute(
                        f"SELECT * FROM {table} WHERE strike = {strike} AND option_type = '{opt_type_str}' AND ts = '{timestamp_str}' LIMIT 1"
                    ).fetchdf()

                if not result.empty:    
                    # print(f"[INFO] Found symbol {symbol} for {underlying} {opt_type} near {atm_strike}, otm_count={otm_count}, attempt={adjusted_otm_count}")
                    return symbol
            except duckdb.CatalogException:
                continue  # Table does not exist, try next attempt
        print(f"[WARN] Could not find valid option symbol for {underlying} {opt_type} near {atm_strike}, otm_count={otm_count}")
        return None
        
class EventInterface(DataInterface):
    def __init__(self, conn = None):
        super().__init__(conn)
        self.weight = 1
        self.event = None
        self.last_event = None
        self.trades = []
        self.all_trades = []
        self.positions = {}
        self.meta_data = []

    def get_mtm(self):
        total_mtm = 0
        pnls = []
        rms_df = []

        unique_symbols = {x['symbol'] for x in self.trades}

        if len(self.trades) > 0:
            for s in unique_symbols:
                expiry = self.parse_date_from_symbol(s)
                if expiry < self.now.date():
                    continue

                symbol_trades = [x for x in self.trades if x['symbol'] == s]

                buy_trades = [x for x in symbol_trades if x['qty_dir'] > 0]
                buy_qty = sum([x['qty_dir'] for x in buy_trades])
                buy_value = sum([x['value'] for x in buy_trades])
                buy_price = buy_value/buy_qty if buy_qty != 0 else 0

                sell_trades = [x for x in symbol_trades if x['qty_dir'] < 0]
                sell_qty = sum([x['qty_dir'] for x in sell_trades])
                sell_value = sum([x['value'] for x in sell_trades])
                sell_price = sell_value/sell_qty if sell_qty != 0 else 0

                net_qty = buy_qty + sell_qty
                net_value = buy_value + sell_value
                net_price = net_value/net_qty if net_qty != 0 else 0

                ltp = self.get_tick(self.now, s)['c']

                pnl = net_value + (net_qty*ltp)
                pnls.append(pnl)

                rms_df.append({'symbol': s, 'ltp': ltp, 'pnl': pnl, 
                               'net_qty': net_qty, 'net_price': net_price, 'net_value': net_value, 
                               'buy_qty': buy_qty, 'buy_price': buy_price, 'buy_value': buy_value, 
                               'sell_qty': sell_qty, 'sell_price': sell_price, 'sell_value': sell_value})

            total_mtm = sum(pnls)
        else:
            total_mtm = 0

        self.contract_notes = pd.DataFrame(rms_df)
        return total_mtm

    def process_event(self, event):
        self.event = event
        self.now = event['ts']

        self.check_if_new_day()

        if self.last_event is not None:
            if self.last_event['ts']>=self.event['ts']:
                print(self.last_event['ts'], self.event['ts'])
                print('STALE TIMESTAMP !!!')
                return

        self.on_event()

        if self.event['bar_complete']:
            self.on_bar_complete()

        if self.event['ts'].time() >= self.stop_time:
            try:
                assert len(self.positions) == 0
            except AssertionError:
                raise AssertionError(f'Trades open after stop time @ {self.now} @ \n TRADES: {pd.DataFrame(self.trades)}\n POSITIONS {self.positions}')

        self.last_event = self.event

    def get_active_trades(self):
        active_trades = []
        sq_off_trades = []
        pos = {}

        for trade in self.trades:
            symbol = trade['symbol']

            if symbol not in pos:
                pos[symbol] = 0

            pos[symbol] += trade['qty_dir']

            if pos[symbol] != 0:
                active_trades.append(trade)

            if pos[symbol] == 0:
                pos.pop(symbol)
                for t in active_trades:
                    if t['symbol'] == symbol:
                        active_trades.remove(t)
                        if len(sq_off_trades) == 0:
                            sq_off_trades.append(t)
                        else:
                            sq_off_trades.pop()
                            sq_off_trades.append(t)

        assert pos == self.positions, f'POS MISMATCH: {pos} != {self.positions}'

        if len(pos) == 0:
            assert len(active_trades) == 0, f'ACTIVE TRADES NOT EMPTY: {active_trades} BUT POSITIONS EMPTY'

        self.sq_off_trades = sq_off_trades
        self.active_trades = active_trades
        return active_trades

    def get_curr_margin(self):
        var_per = 0.09
        active_trades = self.get_active_trades()

        if len(active_trades) != 0 :
            underlying = get_underlying(active_trades[0]['symbol'])
            strike_diff = self.get_strike_diff(underlying)
            lot_size = self.get_lot_size(underlying)
            spot_price = self.get_tick(self.now, f'{underlying}SPOT')['c']

            min_var_price = spot_price*(1 - var_per)
            max_var_price = spot_price*(1 + var_per)
            atm_strike = round(spot_price/strike_diff) * strike_diff
            min_var_strike = round(min_var_price/strike_diff)*strike_diff
            max_var_strike = round(max_var_price/strike_diff)*strike_diff
            var_range = np.arange(min_var_strike, max_var_strike, strike_diff, dtype=np.float64)

            active_trades_df = pd.DataFrame(active_trades)

            try:
                final_df = active_trades_df.groupby('symbol')[['qty_dir', 'value']].sum()
                final_df['qty_dir'] = final_df['qty_dir'].replace(0, np.nan)
                final_df = final_df.dropna()
                final_df['price'] = abs(final_df['value']/final_df['qty_dir'])

                final_df = final_df.reset_index()
                final_df['strike'] = final_df['symbol'].map(self.parse_strike_from_symbol)
                final_df = final_df.sort_values('strike')
                unhedged_pos_ce = final_df[final_df['symbol'].str[-2:] == 'CE']["qty_dir"].sum()
                unhedged_pos_pe = final_df[final_df['symbol'].str[-2:] == 'PE']["qty_dir"].sum()
                total_hedged_lot = (abs(final_df[final_df['qty_dir'] < 0]['qty_dir'].sum()/lot_size))

                final_dict = final_df.to_dict('records')
                first_loop = True
                final_output = []
                elm_margin = 0

                for symbols in final_dict:
                    temp_array = np.array(single_strike_payoff_cal(symbols['strike'], symbols['symbol'][-2:], symbols['price'], var_range, symbols['qty_dir']))
                    if first_loop:
                        final_output = temp_array
                        first_loop = False
                    else:
                        final_output += temp_array

                    if symbols['qty_dir'] < 0:
                        expiry_date = self.parse_date_from_symbol(symbols['symbol'])
                        dte = (expiry_date-self.now.date()).days
                        if dte == 0:
                            elm_margin += abs(symbols['qty_dir']*spot_price*0.02)

                exposure_margin = total_hedged_lot*spot_price*0.02*lot_size
                total_margin = round(abs(min(final_output))+(exposure_margin) + (elm_margin))
                var = abs(min(final_output))                     

            except Exception as e:
                print('ERROR IN CURRENT MARGIN CALCULATION - ', e)
                total_margin = 0
                var = 0

            return total_margin, var
        else:
            return 0, 0

    def check_if_new_day(self):
        self.new_day = False

        if self.last_event is None:
            self.new_day = True
        elif self.last_event['ts'].date() != self.event['ts'].date():
            self.new_day = True

        if self.new_day:
            if len(self.trades) > 0:
                self.all_trades += self.trades
                self.trades = []

            self.positions = {}
            self.on_new_day()        

    def place_trade(self, timestamp, action, qty, symbol, price=None, note="", signal_number=None,price_mode=""):
        trade = {}

        trade['uid'] = self.uid
        trade['ts'] = timestamp
        trade['dte'] = self.get_dte(timestamp, symbol)
        trade['action'] = action

        if action == 'BUY':
            action_int = 1
        elif action == 'SELL':
            action_int = -1
        
        underlying = get_underlying(symbol)

        trade["underlying"] = self.get_tick(timestamp,f"{underlying}SPOT")['c']

        trade['action_int'] = action_int
        trade['qty'] = int(qty * self.weight)
        trade['qty_dir'] = int(qty * action_int * self.weight)
        trade['symbol'] = symbol
        price_time = timestamp

        price_provided = True
        if price is None:
            try:
                price_data = self.get_tick(timestamp, symbol)
                price = float(price_data['c'])
                price_time = price_data['ts']
                price_provided = False
            except Exception as e:
                print(f'ERRTRADE: {e}')
                return (False, np.nan)

        price = float(price)

        if np.isnan(price):
            return (False, np.nan)            

        if price <= 0:
            print(f'ERR: trade price is <= 0 : {timestamp} {symbol} {price}')
            return (False, price) 

        trade['price'] = price
        trade['price_provided'] = price_provided    
        trade['price_time'] = price_time     

        if trade['symbol'] not in self.positions:
            self.positions[trade['symbol']] = 0

        self.positions[trade['symbol']] += trade['qty_dir']

        if self.positions[trade['symbol']] == 0:
            self.positions.pop(trade['symbol'])

        trade['value'] = trade['price']*trade['qty_dir']*-1
        trade['turnover'] = abs(trade['value'])
        trade['system_timestamp'] = datetime.datetime.now()
        trade['note'] = note

        if signal_number is not None: 
            trade['signal_number'] =  signal_number        

        print('NEWTRADE:', trade)
        self.trades.append(trade)
        return (True, price)

    def place_spread_trade(
        self, timestamp, action, qty, symbol_X=None, symbol_Y=None, price_X=None, price_Y=None, note="", signal_number=None
    ):
        try:
            if price_X is None and symbol_X is not None:
                price_X = float(self.get_tick(timestamp, symbol_X)['c'])
            if price_Y is None and symbol_Y is not None:
                price_Y = float(self.get_tick(timestamp, symbol_Y)['c'])
        except Exception as e:
            print(f'SPREADTRADE ERR: {e}')
            return (False, price_X, price_Y)

        if symbol_X is None:
            price_X = np.nan
        if symbol_Y is None:
            price_Y = np.nan

        if (np.isnan(price_X) and symbol_X is not None) or (np.isnan(price_Y) and symbol_Y is not None):
            return (False, price_X, price_Y)

        if action == 'BUY':
            if symbol_X is not None:
                success_X, _ = self.place_trade(timestamp, 'BUY', qty, symbol_X, price_X, "bX_"+note, signal_number)
            else:
                success_X = True

            if symbol_Y is not None:
                success_Y, _ = self.place_trade(timestamp, 'SELL', qty, symbol_Y, price_Y, "sY_"+note, signal_number)
            else:
                success_Y = True 

        elif action == 'SELL':
            if symbol_X is not None:
                success_X, _ = self.place_trade(timestamp, 'SELL', qty, symbol_X, price_X, "sX_"+note, signal_number)
            else:
                success_X = True

            if symbol_Y is not None:
                success_Y, _ = self.place_trade(timestamp, 'BUY', qty, symbol_Y, price_Y, "bY_"+note, signal_number)
            else:
                success_Y = True

        assert success_X == True and success_Y == True
        return (True, price_X, price_Y)

    def on_start(self):
        """
        Things to do at start of strategy
        Ex: arrays to store continuous price series
        """
        pass

    def on_stop(self):
        """
        Things to do at stop of strategy
        Ex: log some information 
        """
        pass

    def on_new_day(self):
        """
        Things to do at start of new day
        Ex: array to store VWAP as VWAP is new every day
        """
        pass

    def on_event(self):
        """
        Things to do on each new event i.e. every second
        """
        pass

    def on_bar_complete(self):
        """
        Things to do on each new bar i.e. every minute 
        """
        pass

class EventInterfacePositional(DataInterface):
    def __init__(self, conn=None):
        super().__init__(conn)
        self.meta_data = []
        self.trade_count = 0
        self.weight = 1
        self.event = None
        self.last_event = None
        self.trades = []
        self.pseudo_trades = []
        self.all_trades = []
        self.positions = {}
        self.symbol_ce = None
        self.symbol_pe = None
        self.symbol_ce_hedge = None
        self.symbol_pe_hedge = None
        self.reset_count_ce = 0
        self.reset_count_pe = 0
        self.position_ce = 0
        self.position_pe = 0

    def get_mtm(self):
        total_mtm = 0
        pnls = []
        rms_df = []

        unique_symbols = {x['symbol'] for x in self.trades}

        if len(self.trades) > 0:
            for s in unique_symbols:
                expiry = self.parse_date_from_symbol(s)
                if expiry < self.now.date():
                    continue

                symbol_trades = [x for x in self.trades if x['symbol'] == s]

                buy_trades = [x for x in symbol_trades if x['qty_dir'] > 0]
                buy_qty = sum([x['qty_dir'] for x in buy_trades])
                buy_value = sum([x['value'] for x in buy_trades])
                buy_price = buy_value/buy_qty if buy_qty != 0 else 0

                sell_trades = [x for x in symbol_trades if x['qty_dir'] < 0]
                sell_qty = sum([x['qty_dir'] for x in sell_trades])
                sell_value = sum([x['value'] for x in sell_trades])
                sell_price = sell_value/sell_qty if sell_qty != 0 else 0

                net_qty = buy_qty + sell_qty
                net_value = buy_value + sell_value
                net_price = net_value/net_qty if net_qty != 0 else 0

                ltp = self.get_tick(self.now, s)['c']
                pnl = net_value + (net_qty*ltp)
                pnls.append(pnl)

                rms_df.append({'symbol': s, 'ltp': ltp, 'pnl': pnl, 
                               'net_qty': net_qty, 'net_price': net_price, 'net_value': net_value, 
                               'buy_qty': buy_qty, 'buy_price': buy_price, 'buy_value': buy_value, 
                               'sell_qty': sell_qty, 'sell_price': sell_price, 'sell_value': sell_value})

            total_mtm = sum(pnls)
        else:
            total_mtm = 0

        self.contract_notes = pd.DataFrame(rms_df)
        return total_mtm

    def get_net_mtm(self):
        total_mtm = 0
        for trade in self.all_trades:
            symbol = trade['symbol']
            qty = trade['qty_dir']
            entry_price = float(trade['price'])
            current_price = float(self.get_tick(self.now, symbol)['c'])
            mtm = (current_price-entry_price)*qty
            friction = 0
            total_mtm += mtm - friction

        return total_mtm
        
    def process_event(self, event):
        self.event = event
        self.now = event['ts']

        self.check_if_new_day()

        if self.last_event is not None:
            if self.last_event['ts']>=self.event['ts']:
                print('STALE TIMESTAMP !!!')
                return

        self.on_event()

        if self.event['bar_complete']:
            if self.now.time() == datetime.time(15, 29):
                if len(self.positions) > 0:
                    for p in self.positions:
                        symbol = p
                        qty = self.positions[p]
                        if qty > 0:
                            action = 'SELL'
                        elif qty < 0:
                            action = 'BUY'
                        elif qty == 0:
                            continue
                        self.place_pseudo_trade(self.now, action, abs(qty), symbol, note='pseudo_exit')
            else:
                self.on_bar_complete()
                   
        self.last_event = self.event

    def get_active_trades(self):
        active_trades = []
        sq_off_trades = []
        pos = {}

        for trade in self.trades:
            symbol = trade['symbol']

            if symbol not in pos:
                pos[symbol] = 0

            pos[symbol] += trade['qty_dir']

            if pos[symbol] != 0:
                active_trades.append(trade)

            if pos[symbol] == 0:
                pos.pop(symbol)
                for t in active_trades:
                    if t['symbol'] == symbol:
                        active_trades.remove(t)
                        if len(sq_off_trades) == 0:
                            sq_off_trades.append(t)
                        else:
                            sq_off_trades.pop()
                            sq_off_trades.append(t)

        assert pos == self.positions, f'POS MISMATCH: {pos} != {self.positions}'

        if len(pos) == 0:
            assert len(active_trades) == 0, f'ACTIVE TRADES NOT EMPTY: {active_trades} BUT POSITIONS EMPTY'

        self.sq_off_trades = sq_off_trades
        self.active_trades = active_trades
        return active_trades

    def get_curr_margin(self):
        var_per = 0.09
        active_trades = self.get_active_trades()

        if len(active_trades) != 0 :
            underlying = get_underlying(active_trades[0]['symbol'])
            strike_diff = self.get_strike_diff(underlying)
            lot_size = self.get_lot_size(underlying)
            spot_price = self.get_tick(self.now, f'{underlying}SPOT')['c']

            min_var_price = spot_price*(1 - var_per)
            max_var_price = spot_price*(1 + var_per)
            atm_strike = round(spot_price/strike_diff) * strike_diff
            min_var_strike = round(min_var_price/strike_diff)*strike_diff
            max_var_strike = round(max_var_price/strike_diff)*strike_diff
            var_range = np.arange(min_var_strike, max_var_strike, strike_diff, dtype=np.float64)

            active_trades_df = pd.DataFrame(active_trades)

            try:
                final_df = active_trades_df.groupby('symbol')[['qty_dir', 'value']].sum()
                final_df['qty_dir'] = final_df['qty_dir'].replace(0, np.nan)
                final_df = final_df.dropna()
                final_df['price'] = abs(final_df['value']/final_df['qty_dir'])

                final_df = final_df.reset_index()
                final_df['strike'] = final_df['symbol'].map(self.parse_strike_from_symbol)
                final_df = final_df.sort_values('strike')
                unhedged_pos_ce = final_df[final_df['symbol'].str[-2:] == 'CE']["qty_dir"].sum()
                unhedged_pos_pe = final_df[final_df['symbol'].str[-2:] == 'PE']["qty_dir"].sum()
                total_hedged_lot = (abs(final_df[final_df['qty_dir'] < 0]['qty_dir'].sum()/lot_size))

                final_dict = final_df.to_dict('records')
                first_loop = True
                final_output = []
                elm_margin = 0

                for symbols in final_dict:
                    temp_array = np.array(single_strike_payoff_cal(symbols['strike'], symbols['symbol'][-2:], symbols['price'], var_range, symbols['qty_dir']))
                    if first_loop:
                        final_output = temp_array
                        first_loop = False
                    else:
                        final_output += temp_array

                    if symbols['qty_dir'] < 0:
                        expiry_date = self.parse_date_from_symbol(symbols['symbol'])
                        dte = (expiry_date-self.now.date()).days
                        if dte == 0:
                            elm_margin += abs(symbols['qty_dir']*spot_price*0.02)

                exposure_margin = total_hedged_lot*spot_price*0.02*lot_size
                total_margin = round(abs(min(final_output))+(exposure_margin) + (elm_margin))
                var = abs(min(final_output))                     

            except Exception as e:
                print('ERROR IN CURRENT MARGIN CALCULATION - ', e)
                total_margin = 0
                var = 0

            return total_margin, var
        else:
            return 0, 0

    def check_if_new_day(self):
        self.new_day = False

        if self.last_event is None:
            self.new_day = True
        elif self.last_event['ts'].date() != self.event['ts'].date():
            self.new_day = True

        if self.new_day:
            if len(self.trades) > 0:
                self.all_trades += self.trades
                self.trades = []
                
            if len(self.pseudo_trades) > 0:
                pseudo_trades = self.pseudo_trades.copy()
                self.pseudo_trades = []

                for trade in pseudo_trades:
                    note = trade['note']
                    if note == 'pseudo_exit':
                        self.all_trades.append(trade)
                        action = trade['action']
                        if action == 'BUY':
                            action_ = 'SELL'
                        elif action == 'SELL':
                            action_ = 'BUY'
                        qty = trade['qty']
                        symbol = trade['symbol']
                        price = trade['price']

                        self.place_pseudo_trade(self.now, action_, qty, symbol, price, note='pseudo_entry')

                self.trades += self.pseudo_trades

            self.on_new_day() 

    def place_pseudo_trade(self, timestamp, action, qty, symbol, price=None, note="", signal_number=None):
        trade = {}

        trade['uid'] = self.uid
        trade['ts'] = timestamp
        trade['action'] = action

        if action == 'BUY':
            action_int = 1
        elif action == 'SELL':
            action_int = -1

        trade['action_int'] = action_int
        trade['qty'] = int(qty * self.weight)
        trade['qty_dir'] = int(qty * action_int * self.weight)
        trade['symbol'] = symbol

        price_provided = True
        if price is None:
            try:
                price = float(self.get_tick(timestamp, symbol)['c'])
                price_provided = False
            except Exception as e:
                return (False, np.nan)

        price = float(price)

        if np.isnan(price):
            return (False, np.nan)            

        if price <= 0:
            print(f'ERR: trade price is <= 0 : {price}')
            return (False, price) 

        trade['price'] = price
        trade['price_provided'] = price_provided    

        trade['value'] = trade['price']*trade['qty_dir']*-1
        trade['turnover'] = 0
        trade['system_timestamp'] = datetime.datetime.now()
        trade['note'] = note

        if signal_number is not None: 
            trade['signal_number'] =  signal_number

        print('PSEUDOTRADE:', trade)
        self.pseudo_trades.append(trade)
        self.trade_count += 1
        return (True, price)
               
    def place_trade(self, timestamp, action, qty, symbol, price=None, note="", signal_number=None, price_time = None):
        trade = {}

        trade['uid'] = self.uid
        underlying = get_underlying(symbol)
        trade["underlying"] = self.get_tick(timestamp,f"{underlying}SPOT")['c']
        trade['ts'] = timestamp
        trade['action'] = action

        if action == 'BUY':
            action_int = 1
        elif action == 'SELL':
            action_int = -1

        trade['action_int'] = action_int
        trade['qty'] = int(qty * self.weight)
        trade['qty_dir'] = int(qty * action_int * self.weight)
        trade['symbol'] = symbol

        if price_time is None:
            price_time = timestamp

        price_provided = True
        if price is None:
            try:
                price_data = self.get_tick(timestamp, symbol)
                price = float(price_data['c'])
                price_time = price_data['ts']
                price_provided = False
            except Exception as e:
                print(f'ERRTRADE: {e}')
                return (False, np.nan)

        price = float(price)

        if np.isnan(price):
            return (False, np.nan)            

        if price <= 0:
            print(f'ERR: trade price is <= 0 : {price}')
            return (False, price) 

        trade['price'] = price
        trade['price_provided'] = price_provided
        trade['price_time'] = price_time     

        if trade['symbol'] not in self.positions:
            self.positions[trade['symbol']] = 0

        self.positions[trade['symbol']] += trade['qty_dir']

        if self.positions[trade['symbol']] == 0:
            self.positions.pop(trade['symbol'])

        trade['value'] = trade['price']*trade['qty_dir']*-1
        trade['turnover'] = abs(trade['value'])
        trade['system_timestamp'] = datetime.datetime.now()
        trade['note'] = note

        if signal_number is not None: 
            trade['signal_number'] =  signal_number

        print('NEWTRADE:', trade)
        self.trades.append(trade)
        # with open("trades.txt", 'a') as f:
            # f.write(f"{trade}\n")
        return (True, price)

    def place_spread_trade(
        self, timestamp, action, qty, symbol_X=None, symbol_Y=None, price_X=None, price_Y=None, note="", signal_number=None
    ):
        try:
            if price_X is None and symbol_X is not None:
                price_X = float(self.get_tick(timestamp, symbol_X)['c'])
            if price_Y is None and symbol_Y is not None:
                price_Y = float(self.get_tick(timestamp, symbol_Y)['c'])
        except Exception as e:
            print(f'SPREADTRADE ERR: {e}')
            return (False, price_X, price_Y)

        if symbol_X is None:
            price_X = np.nan
        if symbol_Y is None:
            price_Y = np.nan

        if (np.isnan(price_X) and symbol_X is not None) or (np.isnan(price_Y) and symbol_Y is not None):
            return (False, price_X, price_Y)

        if action == 'BUY':
            if symbol_X is not None:
                success_X, _ = self.place_trade(timestamp, 'BUY', qty, symbol_X, price_X, "bX_"+note, signal_number)
            else:
                success_X = True

            if symbol_Y is not None:
                success_Y, _ = self.place_trade(timestamp, 'SELL', qty, symbol_Y, price_Y, "sY_"+note, signal_number)
            else:
                success_Y = True 

        elif action == 'SELL':
            if symbol_X is not None:
                success_X, _ = self.place_trade(timestamp, 'SELL', qty, symbol_X, price_X, "sX_"+note, signal_number)
            else:
                success_X = True

            if symbol_Y is not None:
                success_Y, _ = self.place_trade(timestamp, 'BUY', qty, symbol_Y, price_Y, "bY_"+note, signal_number)
            else:
                success_Y = True

        assert success_X == True and success_Y == True
        return (True, price_X, price_Y)

    def on_start(self):
        """
        Things to do at start of strategy
        Ex: arrays to store continuous price series
        """
        pass

    def on_stop(self):
        """
        Things to do at stop of strategy
        Ex: log some information 
        """
        pass

    def on_new_day(self):
        """
        Things to do at start of new day
        Ex: array to store VWAP as VWAP is new every day
        """
        pass

    def on_event(self):
        """
        Things to do on each new event i.e. every second
        """
        pass

    def on_bar_complete(self):
        """
        Things to do on each new bar i.e. every minute 
        """
        pass