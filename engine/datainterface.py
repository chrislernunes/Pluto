import datetime
import direct_redis
import pandas as pd
import numpy as np
import time
from utils import utility
r = direct_redis.DirectRedis()
from utils.utility import stock_tickers
from utils.utility import holidays
import json 

dict_expiry_month_str_nonterminal = {
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'O',
    11: 'N',
    12: 'D',
}

dict_expiry_month_str_nonterminal_invert = {v: k for k, v in dict_expiry_month_str_nonterminal.items()}

dict_expiry_month_str_terminal = {
    1: 'JAN',
    2: 'FEB',
    3: 'MAR',
    4: 'APR',
    5: 'MAY',
    6: 'JUN',
    7: 'JUL',
    8: 'AUG',
    9: 'SEP',
    10: 'OCT',
    11: 'NOV',
    12: 'DEC',
}

lot_size_dict = {
    # INDEX
    'BANKNIFTY': 15,
    'NIFTY': 75,
    'FINNIFTY': 65,
    'MIDCPNIFTY': 50,
    'SENSEX': 20,
}

freeze_qty_dict = {
  'BANKNIFTY': 900,
  'NIFTY': 1800,
  'FINNIFTY': 1800,
  'MIDCPNIFTY': 4200,
  'SENSEX': 1000
}

strike_diff_dict = {
    # INDEX
    'BANKNIFTY': 100 ,
    'NIFTY': 50,
    'FINNIFTY': 50,
    'MIDCPNIFTY': 25,
    'SENSEX': 100,
    'BANKEX': 100,
}

dict_expiry_month_str_terminal_invert = {v: k for k, v in dict_expiry_month_str_terminal.items()}

indexes = ('BANKNIFTY', 'NIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX', 'BANKEX')

class DataInterface:

    def __init__(self):
        pass
    
    def get_lot_size(self, underlying):
        if underlying in lot_size_dict:
            return lot_size_dict[underlying]
        else:
            raise ValueError(f'Unrecognized Underlying: {underlying}')
        
    def get_freeze_quantity(self, underlying):
        if underlying in freeze_qty_dict:
            return freeze_qty_dict[underlying]
        else:
            raise ValueError(f'Unrecognized Underlying: {underlying}')

    def get_strike_diff(self, underlying):
        if underlying in strike_diff_dict:
            return strike_diff_dict[underlying]
        else:
            raise ValueError(f'Unrecognized Underlying: {underlying}')
        
    def get_nearest_expiry(self, timestamp, underlying):
        list_of_expiries = r.hget('list_of_expiries', underlying)
        list_of_expiries.sort()
        list_of_expiries = np.array(list_of_expiries)
        expiry_date = list_of_expiries[list_of_expiries >= timestamp.date()][0]
        expiry_code = str(expiry_date).replace('-', '')[2:]
        return expiry_date
    
    def get_expiry_code(self, timestamp, underlying, expiry_idx):
        list_of_expiries = r.hget('list_of_expiries', underlying)
        list_of_expiries.sort()
        list_of_expiries = np.array(list_of_expiries)
        expiry_date = list_of_expiries[list_of_expiries >= timestamp.date()][expiry_idx]
        expiry_code = str(expiry_date).replace('-', '')[2:]
        return expiry_code
    
    def get_monthly_expiry_code(self, timestamp, underlying):
        list_of_expiries = r.hget('list_of_expiries', underlying)
        list_of_monthly_expiries = [x for x in list_of_expiries if x.month == timestamp.month and x.year == timestamp.year]
        list_of_monthly_expiries.sort()
        list_of_monthly_expiries = np.array(list_of_monthly_expiries)
        expiry_date = list_of_monthly_expiries[-1]
        expiry_code = str(expiry_date).replace('-', '')[2:]
        return expiry_code
    
    def parse_date_from_symbol(self, symbol):
        if symbol.startswith('BANKNIFTY'):
            underlying = 'BANKNIFTY'
        elif symbol.startswith('NIFTY'):
            underlying = 'NIFTY'
        elif symbol.startswith('FINNIFTY'):
            underlying = 'FINNIFTY'
        elif symbol.startswith('MIDCPNIFTY'):
            underlying = 'MIDCPNIFTY'
        elif symbol.startswith('SENSEX'):
            underlying = 'SENSEX'
        elif symbol.startswith('BANKEX'):
            underlying = 'BANKEX'
        for ticker in stock_tickers:
            if symbol.startswith(ticker) : underlying =  ticker
        return datetime.datetime.strptime(symbol.strip(underlying).strip('CE').strip('PE')[:6], '%y%m%d').date()

    def parse_strike_from_symbol(self, symbol):
        # Define possible underlying symbols
        underlying_options = indexes
        # Identify the underlying symbol
        for underlying in underlying_options:
            if symbol.startswith(underlying):
                break
        else:
            raise ValueError(f'Unrecognized Underlying: {symbol}')
        # Extract and return the strike price
        return int(float(symbol[len(underlying):].strip('CE').strip('PE')[6:]))
    #########################

    def replace_strike_in_symbol(self, symbol, strike):
        current_strike = self.parse_strike_from_symbol(symbol)
        new_symbol = symbol.replace(str(current_strike), str(strike))
        return new_symbol

    def shift_strike_in_symbol(self, symbol, shift_otm_count):
        # ...
        strike = self.parse_strike_from_symbol(symbol)
        # ...
        if symbol.startswith('BANKNIFTY'):
            underlying = 'BANKNIFTY'
        elif symbol.startswith('NIFTY'):
            underlying = 'NIFTY'
        elif symbol.startswith('FINNIFTY'):
            underlying = 'FINNIFTY'
        elif symbol.startswith('MIDCPNIFTY'):
            underlying = 'MIDCPNIFTY'
        elif symbol.startswith('SENSEX'):
            underlying = 'SENSEX'
        elif symbol.startswith('BANKEX'):
            return 'BANKEX'
        for ticker in stock_tickers:
            if symbol.startswith(ticker) : underlying =  ticker
        # ...
        strike_diff = strike_diff_dict[underlying]
        # ...
        opt_type = symbol[-2:]
        if opt_type == 'CE':
            shifter = 1
        else:
            shifter = -1    
        # ...
        new_strike = int(float(strike + shift_otm_count*shifter*strike_diff))
        # ...
        new_symbol = symbol.replace(str(strike), str(new_strike))
        return new_symbol
    
    def get_dte(self, timestamp, symbol):
        expiry_date = self.parse_date_from_symbol(symbol)
        start_date = timestamp.date()
        days_list = [
            start_date + datetime.timedelta(days=i)
            for i in range((expiry_date - start_date).days)
            if ((start_date + datetime.timedelta(days=i)).weekday() < 5) 
            and ((start_date + datetime.timedelta(days=i)) not in holidays) # 0 to 4 = Monday to Friday
        ]
        dte = len(days_list)
        if dte < 0:
            print(f'WARNING: {symbol} Expired')
            dte = 69
        return int(dte)
    
    # ...
    def get_all_ticks_by_timestamp(self, timestamp):
        list_of_ticks = r.hgetall('tick_'+str(timestamp))
        json_data = {x : json.loads(y) for x, y in list_of_ticks.items() if isinstance(y, (bytes, str)) }
        remaining_data = { x : y for x, y in list_of_ticks.items() if not isinstance(y, (bytes, str)) }
        remaining_data.update(json_data)
        ticks = pd.DataFrame(remaining_data).transpose()
        return ticks

    def get_all_ticks_by_symbol(self, symbol, col_str='timestamp,o,h,l,c,v,oi'):
        columns = col_str.split(',')
        ticks = r.hget('instrument', symbol)
        ticks['timestamp'] = pd.to_datetime(ticks['timestamp'])
        cols = []
        for c in columns:
            if c in ticks.columns:
                cols.append(c)
            else:
                print(f'Column {c} not found in data')
        ticks = ticks[cols]
        if ticks is not None and len(ticks) > 0:
            ticks = ticks.sort_values('timestamp').reset_index(drop=True)
        return ticks

    # def get_tick(self, timestamp, symbol):
    #     if symbol is None:
    #         print('SYMBOL IS NONE')
    #         return {'o': np.nan, 'h': np.nan, 'l': np.nan, 'c': np.nan, 'v': np.nan, 'oi': np.nan, 'timestamp': None}
    #     try:
    #         tick = r.hget(f'tick_{timestamp}', symbol)
    #         if isinstance(tick, (bytes, str)):
    #             tick = json.loads(tick)

    #         tick['timestamp']=pd.to_datetime(tick['timestamp'])
    #         tick['timestamp'] = timestamp
    #     except Exception as e:
    #         print(f"Couldn't find {symbol} tick in TIME AXIS, trying old tick {timestamp}")
    #         try:
    #             all_ticks = self.get_all_ticks_by_symbol(symbol)
    #             all_ticks['timestamp']=pd.to_datetime(all_ticks['timestamp'])
    #             tick = all_ticks[all_ticks['timestamp']<=timestamp].iloc[-1]
    #             #tick['timestamp'] = timestamp
    #         except Exception as e:
    #             print(f'GETTICK ERR: SYMBOL NOT AVAILABLE {symbol}, {timestamp} {e} ')
    #             tick = {'o': np.nan, 'h': np.nan, 'l': np.nan, 'c': np.nan, 'v': np.nan, 'oi': np.nan, 'timestamp': None}
    #     return tick
    
    def get_tick(self, timestamp, symbol):
        if isinstance(timestamp, datetime.datetime):
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = str(timestamp)

        try:
            instrument = 'Index' if not any(char.isdigit() for char in symbol) else 'Options'
            underlying = next((u for u in indexes if symbol.startswith(u)), None)

            if instrument == 'Options':
                expiry = self.parse_date_from_symbol(symbol).strftime('%Y%m%d')
                strike = self.parse_strike_from_symbol(symbol)
                opt_type = 'call' if 'CE' in symbol else 'put'

            exchange = 'BSE' if underlying == 'SENSEX' else 'NSE'
            if 'SPOT' in underlying:
                underlying = underlying.replace('SPOT', '')

            if instrument == 'Options':
                table = f"{exchange}_{instrument}_{underlying}_{expiry}_{strike}_{opt_type}"
            else:
                table = f"{exchange}_{instrument}_{underlying}"

            # Try exact match first
            result = self.conn.execute(
                f"SELECT * FROM market_data.{table} WHERE timestamp = '{timestamp_str}' LIMIT 1"
            ).fetchdf()

            if not result.empty:
                return result.iloc[0]

            # If not found, get the next available tick
            result = self.conn.execute(
                f"SELECT * FROM market_data.{table} WHERE timestamp > '{timestamp_str}' ORDER BY timestamp ASC LIMIT 1"
            ).fetchdf()

            if not result.empty:
                return result.iloc[0]

            return None

        except Exception as e:
            print(f"Couldn't find {symbol} tick in TIME AXIS, trying old tick {timestamp}")
            try:
                all_ticks = self.get_all_ticks_by_symbol(symbol)
                all_ticks['timestamp'] = pd.to_datetime(all_ticks['timestamp'])
                tick = all_ticks[all_ticks['timestamp'] > timestamp].iloc[0]
            except Exception as e:
                print(f'GETTICK ERR: SYMBOL NOT AVAILABLE {symbol}, {timestamp} {e}')
                tick = {'o': np.nan, 'h': np.nan, 'l': np.nan, 'c': np.nan, 'v': np.nan, 'oi': np.nan, 'timestamp': None}

        return tick

    def get_ticks_of_symbol_between_timestamps(self, symbol, from_timestamp=None, to_timestamp=None):
        if type(from_timestamp) is str:
            from_timestamp = pd.to_datetime(from_timestamp)
        if type(to_timestamp) is str:
            to_timestamp = pd.to_datetime(to_timestamp)
        ticks = self.get_all_ticks_by_symbol(symbol)
        if from_timestamp is not None:
            ticks = ticks[ticks['timestamp'] >= from_timestamp].copy()
        if to_timestamp is not None:
            ticks = ticks[ticks['timestamp'] <= to_timestamp].copy()
        return ticks

    def find_symbol_by_moneyness(self, timestamp, underlying, expiry_idx, opt_type, otm_count):
        assert otm_count >= -100 # don't go too ITM
        assert otm_count <= 150 # don't go too OTM
        # ...
        if expiry_idx == 'm':
            expiry_code = self.get_monthly_expiry_code(timestamp, underlying)
        else:
            expiry_code = self.get_expiry_code(timestamp, underlying, expiry_idx)
        # ...
        if timestamp.time() in [datetime.time(9, 7) , datetime.time(9, 9)]:
            ts = datetime.datetime.combine(timestamp.date(), datetime.time(9, 15))
            spot_price = float(self.get_tick(ts,f"{underlying}SPOT")['o'])
        else:
            spot_price = float(self.get_tick(timestamp,f"{underlying}SPOT")['c'])
        
        spot_price = float(self.get_tick(timestamp,f"{underlying}SPOT")['c'])#float(data.loc[f"{underlying}SPOT"]['c'])
        #print('FUT:',fut_price)
        strike_diff = strike_diff_dict[underlying]
        # ...
        if opt_type == 'CE':
            shifter = 1
        else:
            shifter = -1
        # ...
        atm_strike = round(spot_price/strike_diff)*strike_diff
        selected_strike = int(atm_strike + otm_count*shifter*strike_diff)
        # ...
        symbol = f"{underlying}{expiry_code}{selected_strike}{opt_type}"
        return symbol
    
    def find_symbol_by_premium(self, timestamp, underlying, expiry_idx, opt_type, seek_price, seek_type=None, premium_rms_thresh=1, moneyness_rms_thresh=-10, oi_thresh=1000, perform_rms_checks=False, force_atm=False) :
        # ...
            # return None
        data = self.get_all_ticks_by_timestamp(timestamp)#r.hget('tick', str(timestamp))
        # print(data)
        spot_price = float(self.get_tick(timestamp,f"{underlying}SPOT")['c'])#float(data.loc[f"{underlying}SPOT"]['c'])
        strike_diff = strike_diff_dict[underlying]
        # ...
        if opt_type == 'CE':
            shifter = 1
        else:
            shifter = -1
        # ...
        atm_strike = round(spot_price/strike_diff)*strike_diff
        data = data.reset_index()
        # ...
        if expiry_idx == 'm':
            expiry_code = self.get_monthly_expiry_code(timestamp, underlying)
        else:
            expiry_code = self.get_expiry_code(timestamp, underlying, expiry_idx)
        subset = data[
                    data['index'].str.startswith(underlying) &
                    data['index'].str.contains(expiry_code, regex=False) &
                    data['index'].str.endswith(opt_type)
                ].copy()

        # ONLY KEEP > 0 PRICES
        if seek_type == None:
            subset['c'] = pd.to_numeric(subset['c'], errors='coerce')  # Convert to numeric, NaN for non-numeric
            subset = subset.dropna(subset=['c'])  # 
            subset = subset.loc[subset['c'].values > 0]
        elif seek_type == 'gt':
            subset = subset.loc[subset['c'].values > seek_price]
        elif seek_type == 'lt':
            subset = subset.loc[subset['c'].values < seek_price]
        if len(subset) > 0:
            idx = (subset['c'].astype(float) - seek_price).abs().idxmin()
            symbol = subset.at[idx, 'index']
        else:
            symbol = None
        # RMS CHECKS
        if perform_rms_checks:
            if symbol is not None and seek_price > 5:
                temp_symbol = symbol
                tick = self.get_tick(timestamp, symbol)
                prem = tick['c']
                oi = tick['oi']
                curr_strike = int(re.sub(f"{underlying}|{expiry_code}|{opt_type}", "", symbol))
                moneyness = shifter*(curr_strike-atm_strike)/strike_diff
                if prem > seek_price*(1+premium_rms_thresh) or oi < oi_thresh*self.get_lot_size(underlying) or moneyness < moneyness_rms_thresh:
                    symbol = None
                if force_atm:
                    if moneyness <= -1:
                        symbol = self.find_symbol_by_moneyness(timestamp, underlying, expiry_idx, opt_type, 0)
                        # print(f'FORCED TO ATM, PREVIOUS SYMBOL : {temp_symbol}, ATM SYMBOL : {symbol}')
                if symbol is None and oi < oi_thresh*self.get_lot_size(underlying):
                    symbol = self.shift_strike_in_symbol(temp_symbol, 1)
                    tick = self.get_tick(timestamp, symbol)
                    prem = tick['c']
                    oi = tick['oi']
                    if prem > seek_price*(1+premium_rms_thresh) or oi < oi_thresh*self.get_lot_size(underlying):
                        symbol = None
        return symbol

    def get_underlying(self, symbol):
        if symbol.startswith('NIFTY') : return 'NIFTY'
        elif symbol.startswith('BANKNIFTY') : return 'BANKNIFTY'
        elif symbol.startswith('FINNIFTY') : return 'FINNIFTY'
        elif symbol.startswith('MIDCPNIFTY') : return 'MIDCPNIFTY'
        elif symbol.startswith('SENSEX') : return 'SENSEX'
        elif symbol.startswith('BANKEX') : return 'BANKEX'
        return None


    def get_iv(self, now, trade_symbol):
        def black_scholes(S, K, T, r, sigma, option_type):
    
            # Black-Scholes calculation for Call and Put options
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)

            if option_type == "CE":
                # Call option formula
                option_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            elif option_type == "PE":
                # Put option formula
                option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else:
                raise ValueError("Invalid option type. Use 'CE' for Call or 'PE' for Put.")
            
            return option_price


        def implied_volatility_BQ(S, K, T, r, market_price, option_type="CE"):
            # Objective function for Brent's method: difference between model price and market price
            def objective_function(s_val):
                return black_scholes(S, K, T, r, s_val, option_type) - market_price

            # Brent's method to find the implied volatility (root of objective function)
            iv = brentq(objective_function, 1e-6, 6.0)  # IV is typically between 0 and 5 (in decimal form)
            
            return iv

        if trade_symbol.endswith('CE') or trade_symbol.endswith('PE') :
            # print(trade_symbol)
            underlying = self.get_underlying(trade_symbol)
            if trade_symbol.endswith('CE') : op = 'CE'
            elif trade_symbol.endswith('PE') : op = 'PE'
            
            S = self.get_tick(now, underlying+'SPOT')['c']
            K = self.parse_strike_from_symbol(trade_symbol)  # Assuming ATM (at the money) for simplicity

            if op == 'CE' and S > K and abs(S - K) > S * 0.001 : 
                return 0
            if op == 'PE' and S < K and abs(S - K) > S * 0.001 : 
                return 0
            if abs(S - K) > S * 0.1 :
                return 0
            
            dte = self.get_dte(now,trade_symbol)
            mins_to_exp = dte*60*24 + 375 - ((now.hour*60 + now.minute) - (9*60 + 15))

            T = mins_to_exp / 525600  # Time to expiration in years
            R = 0.07  # Risk-free rate
            market_price = self.get_tick(now, trade_symbol)['c']  # Market price of the call option
            # print(S, K, op, now, market_price)
            try:
                IV = implied_volatility_BQ(S, K, T, R, market_price,op)
            except:
                print(f'IV CALCULATION FAILED FOR {trade_symbol} {now} {S} {K} {T} {R} {market_price} {op}')
                IV = 0
            return IV
        else:
            return 0
        
    def get_option_delta_iv(self, timestamp, symbol):
        # tick = self.get_tick(timestamp, symbol)
        underlying = self.get_underlying(symbol)
        spot_price = float(self.get_tick(timestamp,f"{underlying}SPOT")['c'])
        strike = self.parse_strike_from_symbol(symbol)
        dte = self.get_dte(timestamp, symbol)
        opt_type = symbol[-2:]
        T = (dte + 1)/365
        eq_data = r.hget('eq_daily_ohlc', f'{underlying}SPOT')
        # print(underlying, symbol, spot_price, strike, dte, T, len(eq_data))
        if eq_data is None:
            return 0
        eq_data['date'] = pd.to_datetime(eq_data['date']).dt.date
        # df['date'] = pd.to_datetime(df['date']).dt.date
        eq_data = eq_data[eq_data['date'] <= timestamp.date()]

        log_returns = np.log(eq_data['c'] / eq_data['c'].shift(1))
        rolling_std = log_returns.rolling(window=252).std()
        volatility = rolling_std * np.sqrt(252)  # Annualize the standard deviation
        # eq_data['Volatility'] = volatility
        # print('Volatility:',volatility)
        sigma = self.get_iv(timestamp, symbol)
        interest_rate = 0.07

        d1 = (np.log(spot_price / strike) + (interest_rate + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if opt_type == 'CE':
            delta = si.norm.cdf(d1)
        elif opt_type == 'PE':
            delta = -si.norm.cdf(-d1)

        return round(delta, 2)

    def get_option_delta_hv(self, timestamp, symbol):
        # tick = self.get_tick(timestamp, symbol)
        underlying = self.get_underlying(symbol)
        spot_price = float(self.get_tick(timestamp,f"{underlying}SPOT")['c'])
        strike = self.parse_strike_from_symbol(symbol)
        dte = self.get_dte(timestamp, symbol)
        opt_type = symbol[-2:]
        T = (dte + 1)/365
        eq_data = r.hget('eq_daily_ohlc', f'{underlying}SPOT')
        # print(underlying, symbol, spot_price, strike, dte, T, len(eq_data))
        if eq_data is None:
            return 0
        eq_data['date'] = pd.to_datetime(eq_data['date']).dt.date
        # df['date'] = pd.to_datetime(df['date']).dt.date
        eq_data = eq_data[eq_data['date'] <= timestamp.date()]

        log_returns = np.log(eq_data['c'] / eq_data['c'].shift(1))
        rolling_std = log_returns.rolling(window=252).std()
        volatility = rolling_std * np.sqrt(252)  # Annualize the standard deviation
        # eq_data['Volatility'] = volatility
        # print('Volatility:',volatility)
        sigma = volatility.iloc[-1]
        interest_rate = 0.07

        d1 = (np.log(spot_price / strike) + (interest_rate + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if opt_type == 'CE':
            delta = si.norm.cdf(d1)
        elif opt_type == 'PE':
            delta = -si.norm.cdf(-d1)

        return round(delta, 2)

    def find_symbol_by_delta(self, timestamp, underlying, expiry_idx, opt_type, seek_delta, seek_type=None) :

        if expiry_idx == 'm':
            expiry_code = self.get_monthly_expiry_code(timestamp, underlying)
        else:
            expiry_code = self.get_expiry_code(timestamp, underlying, expiry_idx)
        # ...
        # print('expiry code:',expiry_code,expiry_idx)
        data = self.get_all_ticks_by_timestamp(timestamp)
        subset_data = data[(data.index.str.startswith(underlying)) & (data.index.str.contains(expiry_code)) & (data.index.str.endswith(opt_type))].copy()
        # subset_data=data[data.index.astypr(str)== f'{underlying}{expiry_code}{opt_type}']
        subset_data['delta'] = subset_data.index.map(lambda x : self.get_option_delta_iv(timestamp, x))

        if seek_type == None:
            subset_data['diff'] = abs(subset_data['delta'] - seek_delta)
            return subset_data.loc[subset_data['diff'].idxmin()].name
        
        elif seek_type == 'gt':
            subset_data = subset_data[subset_data['delta'] >= seek_delta]
            if len(subset_data) > 0:
                return subset_data.loc[subset_data['delta'].idxmin()].name
            else:
                subset_data['diff'] = abs(subset_data['delta'] - seek_delta)
                return subset_data.loc[subset_data['diff'].idxmin()].name
        
        elif seek_type == 'lt':
            subset_data = subset_data[subset_data['delta'] <= seek_delta]
            if len(subset_data) > 0:
                return subset_data.loc[subset_data['delta'].idxmax()].name
            else:
                subset_data['diff'] = abs(subset_data['delta'] - seek_delta)
                return subset_data.loc[subset_data['diff'].idxmin()].name
        
    def get_short_premium(self, underlying):
        max_positions = 2
        spot_price = float(self.get_tick(self.now,f"{underlying}SPOT")['c'])

        strike_diff = strike_diff_dict[underlying]

        # print(spot_price, type(spot_price), strike_diff, type(strike_diff))
        atm_strike = round(spot_price/strike_diff)*strike_diff

        leverage = r.get(f'leverage:{underlying}') 
        ps = r.get(f'ps:{underlying}') 

        if leverage is None or ps is None:
            absolute_path = os.path.abspath('xxx/live_support_data/lev_ps_data.csv')
            lev_ps_data = pd.read_csv(absolute_path)
            lev_ps_data.set_index('field', inplace=True)
            leverage = lev_ps_data.loc['leverage', underlying]
            ps = lev_ps_data.loc['ps', underlying]
        else :
            ps = float(ps)
            leverage = float(leverage)

        lot_size = self.get_lot_size(underlying)

        margin = (atm_strike * lot_size * max_positions) / leverage
        min_premium = (margin * ps) / lot_size
        min_premium = round(min_premium, 2)
        print(f'Index : {underlying}, atm_strike : {atm_strike}, leverage : {leverage}, PS : {ps}, lotSize : {lot_size}, margin : {margin}, min premium : {min_premium}')

        return min_premium
#################

    
#################