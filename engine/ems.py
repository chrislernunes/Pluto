# import datetime, os, glob, re, json
# import direct_redis
# import pandas as pd
# from utils.utility import holidays
# import numpy as np
# import time, traceback, math, sys
# #import scipy.stats as si
# #from scipy.optimize import brentq
# #from scipy.stats import norm
# from engine.datainterface import DataInterface
# sys.path.append("./cython_modules")
# from cython_modules.payoff import single_strike_payoff_cal

# from utils.utility import stock_tickers

# r = direct_redis.DirectRedis()

# lot_size_dict = {
#     # INDEX
#     'BANKNIFTY': 15,
#     'NIFTY': 75,
#     'FINNIFTY': 65,
#     'MIDCPNIFTY': 50,
#     'SENSEX': 20,
# }

# freeze_qty_dict = {
#   'BANKNIFTY': 900,
#   'NIFTY': 1800,
#   'FINNIFTY': 1800,
#   'MIDCPNIFTY': 4200,
#   'SENSEX': 1000
# }

# strike_diff_dict = {
#     # INDEX
#     'BANKNIFTY': 100 ,
#     'NIFTY': 50,
#     'FINNIFTY': 50,
#     'MIDCPNIFTY': 25,
#     'SENSEX': 100,
#     'BANKEX': 100,
# }

# indexes = ('BANKNIFTY', 'NIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX', 'BANKEX')

# def get_underlying(symbol):
#     if symbol.startswith('NIFTY') : return 'NIFTY'
#     elif symbol.startswith('BANKNIFTY') : return 'BANKNIFTY'
#     elif symbol.startswith('FINNIFTY') : return 'FINNIFTY'
#     elif symbol.startswith('MIDCPNIFTY') : return 'MIDCPNIFTY'
#     elif symbol.startswith('SENSEX') : return 'SENSEX'
#     elif symbol.startswith('BANKEX') : return 'BANKEX'
#     return None

# def parse_strike_from_symbol(symbol):
#     # Define possible underlying symbols
#     underlying_options = indexes
#     # Identify the underlying symbol
#     for underlying in underlying_options:
#         if symbol.startswith(underlying):
#             break
#     else:
#         raise ValueError(f'Unrecognized Underlying: {symbol}')
#     # Extract and return the strike price
#     return int(float(symbol[len(underlying):].strip('CE').strip('PE')[6:]))

# # equity_data = r.hgetall('eq_daily_ohlc')
# #########################

# class EventInterface(DataInterface):

#     def __init__(self):
#         # ...
#         super().__init__()
#         # ...
#         self.weight = 1
#         # ...
#         self.event = None
#         self.last_event = None
#         # ...
#         self.trades = []
#         self.all_trades = []
#         self.positions = {}
#         self.meta_data = []
#         #self.positions_SANITY = {}

#     def get_mtm(self):
#         total_mtm = 0
#         pnls = []
#         rms_df = []

#         unique_symbols = {x['symbol'] for x in self.trades} #pd.Series([x['symbol'] for x in self.trades]).unique()
#         # unique_symbols = [x for x in unique_symbols if self.parse_date_from_symbol(x) >= self.now.date()]

#         if len(self.trades) > 0:
#             for s in unique_symbols:

#                 expiry = self.parse_date_from_symbol(s)
#                 if expiry < self.now.date():
#                     continue

#                 symbol_trades = [x for x in self.trades if x['symbol'] == s]
                
#                 buy_trades = [x for x in symbol_trades if x['qty_dir'] > 0]
#                 buy_qty = sum([x['qty_dir'] for x in buy_trades])
#                 buy_value = sum([x['value'] for x in buy_trades])
#                 buy_price = buy_value/buy_qty if buy_qty != 0 else 0

#                 sell_trades = [x for x in symbol_trades if x['qty_dir'] < 0]
#                 sell_qty = sum([x['qty_dir'] for x in sell_trades])
#                 sell_value = sum([x['value'] for x in sell_trades])
#                 sell_price = sell_value/sell_qty if sell_qty != 0 else 0

#                 net_qty = buy_qty + sell_qty
#                 net_value = buy_value + sell_value
#                 net_price = net_value/net_qty if net_qty != 0 else 0

#                 ltp = self.get_tick(self.now, s)['c'] #r.get(f'ltp.{s}')
#                 pnl = net_value + (net_qty*ltp)
#                 pnls.append(pnl)

#                 rms_df.append({'symbol': s, 'ltp': ltp, 'pnl': pnl, 
#                                'net_qty': net_qty, 'net_price': net_price, 'net_value': net_value, 
#                                'buy_qty': buy_qty, 'buy_price': buy_price, 'buy_value': buy_value, 
#                                'sell_qty': sell_qty, 'sell_price': sell_price, 'sell_value': sell_value})
                
#             total_mtm = sum(pnls)
#         else:
#             total_mtm = 0

#         self.contract_notes = pd.DataFrame(rms_df)
#         return total_mtm

    
#     def process_event(self, event):
#         self.event = event
#         self.now = event['timestamp']
#         #print(self.now)
#         #return
#         # ...
#         self.check_if_new_day()
#         #self.run_sanity_check()
#         # SANITY CHECK: don't process stale timestamp
#         if self.last_event is not None:
#             if self.last_event['timestamp']>=self.event['timestamp']:
#                 print(self.last_event['timestamp'], self.event['timestamp'])
#                 print('STALE TIMESTAMP !!!')
#                 return
#         #self.on_event()
#         if self.event['bar_complete']:
#             # self.mtm = self.get_mtm()
#             # self.margin, self.var = self.get_curr_margin()
#             # self.meta_data.append({
#             #     'timestamp': self.now,
#             #     'mtm': self.mtm,
#             #     'margin': self.margin,
#             #     'var': self.var,
#             # })
#             self.on_bar_complete()
            
#         # SANITY CHECK: no positions open after stop time
#         if self.event['timestamp'].time() >= self.stop_time:
#             try:
#                 assert len(self.positions) == 0
#             except AssertionError:
#                 raise AssertionError(f'Trades open after stop time @ {self.now} @ \n TRADES: {pd.DataFrame(self.trades)}\n POSITIONS {self.positions}')
#         # ...
#         self.last_event = self.event

#     def get_active_trades(self):
#         active_trades = []
#         sq_off_trades = []
#         pos = {}
#         for trade in self.trades:
#             symbol = trade['symbol']
            
#             # INITIALIZE NEW POSITION
#             if symbol not in pos:
#                 pos[symbol] = 0
            
#             # ADD QTY
#             pos[symbol] += trade['qty_dir']
            
#             if pos[symbol] != 0:
#                 active_trades.append(trade)

#             # IF POSITION IS CLOSED
#             if pos[symbol] == 0:
#                 pos.pop(symbol)
#                 for t in active_trades:
#                     if t['symbol'] == symbol:
#                         active_trades.remove(t)
#                         if len(sq_off_trades) == 0:
#                             sq_off_trades.append(t)
#                         else:
#                             sq_off_trades.pop()
#                             sq_off_trades.append(t)

                        
#         assert pos == self.positions, f'POS MISMATCH: {pos} != {self.positions}'

#         if len(pos) == 0:
#             assert len(active_trades) == 0, f'ACTIVE TRADES NOT EMPTY: {active_trades} BUT POSITIONS EMPTY'

#         self.sq_off_trades = sq_off_trades
#         self.active_trades = active_trades
#         return active_trades
    

#     def get_curr_margin(self):
#         var_per = 0.09
#         active_trades = self.get_active_trades()
#         if len(active_trades) != 0 :
#             # SINCE ALL THE POSITIONS WILL BE OF SAME INDEX TAKE ANY TRADE TO FIND UNDERLYING AND GET LOT SIZE
#             underlying = self.get_underlying(active_trades[0]['symbol'])
#             strike_diff = self.get_strike_diff(underlying)
#             lot_size = self.get_lot_size(underlying)
#             spot_price = self.get_tick(self.now, f'{underlying}SPOT')['c']

#             min_var_price = spot_price*(1 - var_per)
#             max_var_price = spot_price*(1 + var_per)
#             atm_strike = round(spot_price/strike_diff) * strike_diff
#             min_var_strike = round(min_var_price/strike_diff)*strike_diff
#             max_var_strike = round(max_var_price/strike_diff)*strike_diff
#             var_range = np.arange(min_var_strike, max_var_strike, strike_diff, dtype=np.float64)


#             active_trades_df = pd.DataFrame(active_trades)
#             try:
#                 final_df = active_trades_df.groupby('symbol')[['qty_dir', 'value']].sum()

#                 final_df['qty_dir'] = final_df['qty_dir'].replace(0, np.nan)
#                 final_df = final_df.dropna()
#                 final_df['price'] = abs(final_df['value']/final_df['qty_dir'])
                
#                 final_df = final_df.reset_index()
#                 # final_df['strike'] = final_df['symbol'].apply(lambda x: strike_from_symbol(x))
#                 final_df['strike'] = final_df['symbol'].map(parse_strike_from_symbol)
#                 final_df = final_df.sort_values('strike')
#                 unhedged_pos_ce = final_df[final_df['symbol'].str[-2:] == 'CE']["qty_dir"].sum()
#                 unhedged_pos_pe = final_df[final_df['symbol'].str[-2:] == 'PE']["qty_dir"].sum()
#                 total_hedged_lot = (abs(final_df[final_df['qty_dir'] < 0]['qty_dir'].sum()/lot_size))

#                 final_dict = final_df.to_dict('records')

#                 first_loop = True
#                 final_output = []
#                 elm_margin = 0

#                 for symbols in final_dict:
#                     temp_array = np.array(single_strike_payoff_cal(symbols['strike'], symbols['symbol'][-2:], symbols['price'], var_range, symbols['qty_dir']))
#                     if first_loop:
#                         final_output = temp_array
#                         first_loop = False
#                     else:
#                         final_output += temp_array

#                     if symbols['qty_dir'] < 0:
#                         expiry_date = self.parse_date_from_symbol(symbols['symbol'])
#                         dte = (expiry_date-self.now.date()).days
#                         if dte == 0:
#                             elm_margin += abs(symbols['qty_dir']*spot_price*0.02)
    
#                 exposure_margin = total_hedged_lot*spot_price*0.02*lot_size
#                 # elm_margin = total_hedged_lot*spot_price*0.02*lot_size
#                 total_margin = round(abs(min(final_output))+(exposure_margin) + (elm_margin))
#                 var = abs(min(final_output))                     
            
#             except Exception as e:
#                 print('ERROR IN CURRENT MARGIN CALCULATION - ', e)
#                 total_margin = 0
#                 var = 0

#             return total_margin, var
        
#         else:
#             return 0, 0
    
#     def check_if_new_day(self):
#         self.new_day = False
#         if self.last_event is None:
#             self.new_day = True
#         elif self.last_event['timestamp'].date() != self.event['timestamp'].date():
#             self.new_day = True
#         # ...
#         if self.new_day:
#             # check if no open positions
#             if len(self.trades) > 0:
#                 # assert 0 == pd.DataFrame(self.trades).groupby('symbol')['qty_dir'].sum().abs().sum()
#                 self.all_trades += self.trades
#                 self.trades = []
#             self.positions = {}
#             # ...
#             self.on_new_day()        
        
#     def place_trade(self, timestamp, action, qty, symbol, price=None, note="", signal_number=None):
#         trade = {}
#         # print('yay trade')
#         trade['uid'] = self.uid
#         # ...
#         trade['timestamp'] = timestamp
#         trade['dte'] = self.get_dte(timestamp, symbol)
#         # ...
#         trade['action'] = action
#         if action == 'BUY':
#             action_int = 1
#         elif action == 'SELL':
#             action_int = -1
#         trade['action_int'] = action_int
#         # ...
#         #trade['action_int'] = action_int
#         trade['qty'] = int(qty * self.weight)
#         trade['qty_dir'] = int(qty * action_int * self.weight)
#         trade['symbol'] = symbol
#         # ...
#         price_provided = True
#         if price is None:
#             try:
#                 price = float(self.get_tick(timestamp, symbol)['c'])
#                 price_provided = False
#             except Exception as e:
#                 #print(f'ERRTRADE: {e}')
#                 return (False, np.nan)
#         price = float(price)
#         if np.isnan(price):
#             return (False, np.nan)            
#         if price <= 0:
#             print(f'ERR: trade price is <= 0 : {timestamp} {symbol} {price}')
#             return (False, price) 
#         trade['price'] = price
#         trade['price_provided'] = price_provided    
#         #trade['value'] = 
#         if trade['symbol'] not in self.positions:
#             self.positions[trade['symbol']] = 0
#         self.positions[trade['symbol']] += trade['qty_dir']
#         if self.positions[trade['symbol']] == 0:
#             self.positions.pop(trade['symbol'])
#         # ...
#         trade['value'] = trade['price']*trade['qty_dir']*-1
#         # trade['buy_value'] = 0
#         # trade['sell_value'] = 0
#         trade['turnover'] = abs(trade['value'])
#         # ...
#         trade['system_timestamp'] = datetime.datetime.now()
#         trade['note'] = note
#         if signal_number is not None: trade['signal_number'] =  signal_number
#         print('NEWTRADE:', trade)
#         self.trades.append(trade)
#         return (True, price)
    
#     def place_spread_trade(
#         self, timestamp, action, qty, symbol_X=None, symbol_Y=None, price_X=None, price_Y=None, note="", signal_number=None
#     ):
#         try:
#             if price_X is None and symbol_X is not None:
#                 price_X = float(self.get_tick(timestamp, symbol_X)['c'])
#             if price_Y is None and symbol_Y is not None:
#                 price_Y = float(self.get_tick(timestamp, symbol_Y)['c'])
#         except Exception as e:
#             print(f'SPREADTRADE ERR: {e}')
#             return (False, price_X, price_Y)
#         if symbol_X is None:
#             price_X = np.nan
#         if symbol_Y is None:
#             price_Y = np.nan
#         #print(price_X, type(price_X), price_Y, type(price_Y))
#         if (np.isnan(price_X) and symbol_X is not None) or (np.isnan(price_Y) and symbol_Y is not None):
#             return (False, price_X, price_Y)
#         if action == 'BUY':
#             if symbol_X is not None:
#                 success_X, _ = self.place_trade(timestamp, 'BUY', qty, symbol_X, price_X, "bX_"+note, signal_number)
#             else:
#                 success_X = True
#             if symbol_Y is not None:
#                 success_Y, _ = self.place_trade(timestamp, 'SELL', qty, symbol_Y, price_Y, "sY_"+note, signal_number)
#             else:
#                 success_Y = True 
#         elif action == 'SELL':
#             if symbol_X is not None:
#                 success_X, _ = self.place_trade(timestamp, 'SELL', qty, symbol_X, price_X, "sX_"+note, signal_number)
#             else:
#                 success_X = True
#             if symbol_Y is not None:
#                 success_Y, _ = self.place_trade(timestamp, 'BUY', qty, symbol_Y, price_Y, "bY_"+note, signal_number)
#             else:
#                 success_Y = True
#         assert success_X == True and success_Y == True
#         return (True, price_X, price_Y)
        
        
    
#     #######################################
#     #### DEFINE THE FOLLOWING IN STRAT ####
#     #######################################
    
#     def on_start(self):
#         """
#         Things to do at start of strategy
#         Ex: arrays to store continuous price series
#         """
#         pass

#     def on_stop(self):
#         """
#         Things to do at stop of strategy
#         Ex: log some information 
#         """
#         pass
    
#     def on_new_day(self):
#         """
#         Things to do at start of new day
#         Ex: array to store VWAP as VWAP is new every day
#         """
#         pass
    
#     def on_event(self):
#         """
#         Things to do on each new event i.e. every second
#         """
#         pass

#     def on_bar_complete(self):
#         """
#         Things to do on each new bar i.e. every minute 
#         """
#         pass


# class EventInterfacePositional(DataInterface):

#     def __init__(self):
#         # ...
#         super().__init__()
#         # ...
#         self.meta_data = []
#         self.trade_count = 0
#         self.weight = 1
#         # ...
#         self.event = None
#         self.last_event = None
#         # ...
#         self.trades = []
#         self.pseudo_trades = []
#         self.all_trades = []
#         self.positions = {}

#         self.symbol_ce = None
#         self.symbol_pe = None

#         self.symbol_ce_hedge = None
#         self.symbol_pe_hedge = None

#         self.reset_count_ce = 0
#         self.reset_count_pe = 0

#         self.position_ce = 0
#         self.position_pe = 0

#     def get_mtm(self):
#         total_mtm = 0
#         pnls = []
#         rms_df = []

#         unique_symbols = {x['symbol'] for x in self.trades} #pd.Series([x['symbol'] for x in self.trades]).unique()
#         # unique_symbols = [x for x in unique_symbols if self.parse_date_from_symbol(x) >= self.now.date()]
#         if len(self.trades) > 0:
#             for s in unique_symbols:

#                 expiry = self.parse_date_from_symbol(s)
#                 if expiry < self.now.date():
#                     continue

#                 symbol_trades = [x for x in self.trades if x['symbol'] == s]
                
#                 buy_trades = [x for x in symbol_trades if x['qty_dir'] > 0]
#                 buy_qty = sum([x['qty_dir'] for x in buy_trades])
#                 buy_value = sum([x['value'] for x in buy_trades])
#                 buy_price = buy_value/buy_qty if buy_qty != 0 else 0

#                 sell_trades = [x for x in symbol_trades if x['qty_dir'] < 0]
#                 sell_qty = sum([x['qty_dir'] for x in sell_trades])
#                 sell_value = sum([x['value'] for x in sell_trades])
#                 sell_price = sell_value/sell_qty if sell_qty != 0 else 0

#                 net_qty = buy_qty + sell_qty
#                 net_value = buy_value + sell_value
#                 net_price = net_value/net_qty if net_qty != 0 else 0

#                 ltp = self.get_tick(self.now, s)['c'] #r.get(f'ltp.{s}')
#                 pnl = net_value + (net_qty*ltp)
#                 pnls.append(pnl)

#                 rms_df.append({'symbol': s, 'ltp': ltp, 'pnl': pnl, 
#                                'net_qty': net_qty, 'net_price': net_price, 'net_value': net_value, 
#                                'buy_qty': buy_qty, 'buy_price': buy_price, 'buy_value': buy_value, 
#                                'sell_qty': sell_qty, 'sell_price': sell_price, 'sell_value': sell_value})
                
#             total_mtm = sum(pnls)
#         else:
#             total_mtm = 0

#         self.contract_notes = pd.DataFrame(rms_df)
#         return total_mtm

    

#     def get_net_mtm(self):
#         total_mtm = 0
#         for trade in self.all_trades:
#             symbol = trade['symbol']
#             qty = trade['qty_dir']
#             entry_price = float(trade['price'])
#             current_price = float(self.get_tick(self.now, symbol)['c'])
#             mtm = (current_price-entry_price)*qty
#             friction = 0
#             total_mtm += mtm - friction
#         return total_mtm
        
#     def process_event(self, event):
#         self.event = event
#         self.now = event['timestamp']
        
#         # ...
#         self.check_if_new_day()
#         #self.run_sanity_check()
#         # SANITY CHECK: don't process stale timestamp
#         if self.last_event is not None:
#             if self.last_event['timestamp']>=self.event['timestamp']:
#                 print('STALE TIMESTAMP !!!')
#                 return
#         #self.on_event()
#         if self.event['bar_complete']:
            
#             if self.now.time() == datetime.time(15, 29):
#                 if len(self.positions) > 0:
#                     # positions are open place a reverse pseudo trade
#                     for p in self.positions:
#                         symbol = p
#                         qty = self.positions[p]
#                         if qty > 0:
#                             action = 'SELL'
#                         elif qty < 0:
#                             action = 'BUY'
#                         elif qty == 0:
#                             continue
#                         self.place_pseudo_trade(self.now, action, abs(qty), symbol, note='pseudo_exit')
#             else:
#                 # self.mtm = self.get_mtm()
#                 # self.margin, self.var = self.get_curr_margin()
#                 # self.meta_data.append({
#                 #     'timestamp': self.now,
#                 #     'mtm': self.mtm,
#                 #     'margin': self.margin,
#                 #     'var': self.var,
#                 # })        
#                 self.on_bar_complete()
                   
#         # ...
#         self.last_event = self.event

#     def get_active_trades(self):
#         active_trades = []
#         sq_off_trades = []
#         pos = {}
#         for trade in self.trades:
#             symbol = trade['symbol']
            
#             # INITIALIZE NEW POSITION
#             if symbol not in pos:
#                 pos[symbol] = 0
            
#             # ADD QTY
#             pos[symbol] += trade['qty_dir']
            
#             if pos[symbol] != 0:
#                 active_trades.append(trade)

#             # IF POSITION IS CLOSED
#             if pos[symbol] == 0:
#                 pos.pop(symbol)
#                 for t in active_trades:
#                     if t['symbol'] == symbol:
#                         active_trades.remove(t)
#                         if len(sq_off_trades) == 0:
#                             sq_off_trades.append(t)
#                         else:
#                             sq_off_trades.pop()
#                             sq_off_trades.append(t)

                        
#         assert pos == self.positions, f'POS MISMATCH: {pos} != {self.positions}'

#         if len(pos) == 0:
#             assert len(active_trades) == 0, f'ACTIVE TRADES NOT EMPTY: {active_trades} BUT POSITIONS EMPTY'

#         self.sq_off_trades = sq_off_trades
#         self.active_trades = active_trades
#         return active_trades
    
#     def get_curr_margin(self):
#         var_per = 0.09
#         active_trades = self.get_active_trades()
#         if len(active_trades) != 0 :
#             # SINCE ALL THE POSITIONS WILL BE OF SAME INDEX TAKE ANY TRADE TO FIND UNDERLYING AND GET LOT SIZE
#             underlying = self.get_underlying(active_trades[0]['symbol'])
#             strike_diff = self.get_strike_diff(underlying)
#             lot_size = self.get_lot_size(underlying)
#             spot_price = self.get_tick(self.now, f'{underlying}SPOT')['c']

#             min_var_price = spot_price*(1 - var_per)
#             max_var_price = spot_price*(1 + var_per)
#             atm_strike = round(spot_price/strike_diff) * strike_diff
#             min_var_strike = round(min_var_price/strike_diff)*strike_diff
#             max_var_strike = round(max_var_price/strike_diff)*strike_diff
#             var_range = np.arange(min_var_strike, max_var_strike, strike_diff, dtype=np.float64)


#             active_trades_df = pd.DataFrame(active_trades)
#             try:
#                 final_df = active_trades_df.groupby('symbol')[['qty_dir', 'value']].sum()

#                 final_df['qty_dir'] = final_df['qty_dir'].replace(0, np.nan)
#                 final_df = final_df.dropna()
#                 final_df['price'] = abs(final_df['value']/final_df['qty_dir'])
                
#                 final_df = final_df.reset_index()
#                 # final_df['strike'] = final_df['symbol'].apply(lambda x: strike_from_symbol(x))
#                 final_df['strike'] = final_df['symbol'].map(parse_strike_from_symbol)
#                 final_df = final_df.sort_values('strike')
#                 unhedged_pos_ce = final_df[final_df['symbol'].str[-2:] == 'CE']["qty_dir"].sum()
#                 unhedged_pos_pe = final_df[final_df['symbol'].str[-2:] == 'PE']["qty_dir"].sum()
#                 total_hedged_lot = (abs(final_df[final_df['qty_dir'] < 0]['qty_dir'].sum()/lot_size))

#                 final_dict = final_df.to_dict('records')

#                 first_loop = True
#                 final_output = []
#                 elm_margin = 0

#                 for symbols in final_dict:
#                     temp_array = np.array(single_strike_payoff_cal(symbols['strike'], symbols['symbol'][-2:], symbols['price'], var_range, symbols['qty_dir']))
#                     if first_loop:
#                         final_output = temp_array
#                         first_loop = False
#                     else:
#                         final_output += temp_array

#                     if symbols['qty_dir'] < 0:
#                         expiry_date = self.parse_date_from_symbol(symbols['symbol'])
#                         dte = (expiry_date-self.now.date()).days
#                         if dte == 0:
#                             elm_margin += abs(symbols['qty_dir']*spot_price*0.02)
    
#                 exposure_margin = total_hedged_lot*spot_price*0.02*lot_size
#                 # elm_margin = total_hedged_lot*spot_price*0.02*lot_size
#                 total_margin = round(abs(min(final_output))+(exposure_margin) + (elm_margin))
#                 var = abs(min(final_output))                     
            
#             except Exception as e:
#                 print('ERROR IN CURRENT MARGIN CALCULATION - ', e)
#                 total_margin = 0
#                 var = 0

#             return total_margin, var
        
#         else:
#             return 0, 0
    
#     def check_if_new_day(self):

#         self.new_day = False
#         if self.last_event is None:
#             self.new_day = True
#         elif self.last_event['timestamp'].date() != self.event['timestamp'].date():
#             self.new_day = True
#         # ...
#         if self.new_day:
#             if len(self.trades) > 0:
#                 self.all_trades += self.trades
#                 self.trades = []
                
#             if len(self.pseudo_trades) > 0:
#                 pseudo_trades = self.pseudo_trades.copy()
#                 self.pseudo_trades = []
#                 for trade in pseudo_trades:
#                     # self.all_trades.append(trade)
#                     note = trade['note']
#                     if note == 'pseudo_exit':
#                         self.all_trades.append(trade)
#                         action = trade['action']
#                         if action == 'BUY':
#                             action_ = 'SELL'
#                         elif action == 'SELL':
#                             action_ = 'BUY'
#                         qty = trade['qty']
#                         symbol = trade['symbol']
#                         price = trade['price']
                        
#                         self.place_pseudo_trade(self.now, action_, qty, symbol, price, note='pseudo_entry')
#                 self.trades += self.pseudo_trades

#             # ...
#             self.on_new_day() 

#     def place_pseudo_trade(self, timestamp, action, qty, symbol, price=None, note="", signal_number=None):
#         trade = {}
#         # print('yay trade')
#         trade['uid'] = self.uid
#         # ...
#         trade['timestamp'] = timestamp
#         # trade['dte'] = self.get_dte(timestamp, symbol)
#         # ...
#         trade['action'] = action
#         if action == 'BUY':
#             action_int = 1
#         elif action == 'SELL':
#             action_int = -1
#         trade['action_int'] = action_int
#         # ...
#         #trade['action_int'] = action_int
#         trade['qty'] = int(qty * self.weight)
#         trade['qty_dir'] = int(qty * action_int * self.weight)
#         trade['symbol'] = symbol
#         # ...
#         price_provided = True
#         if price is None:
#             try:
#                 price = float(self.get_tick(timestamp, symbol)['c'])
#                 price_provided = False
#             except Exception as e:
#                 #print(f'ERRTRADE: {e}')
#                 return (False, np.nan)
#         price = float(price)
#         if np.isnan(price):
#             return (False, np.nan)            
#         if price <= 0:
#             print(f'ERR: trade price is <= 0 : {price}')
#             return (False, price) 
#         trade['price'] = price
#         trade['price_provided'] = price_provided    
#         #trade['value'] = 
#         # if trade['symbol'] not in self.positions:
#         #     self.positions[trade['symbol']] = 0
#         # self.positions[trade['symbol']] += trade['qty_dir']
#         # if self.positions[trade['symbol']] == 0:
#         #     self.positions.pop(trade['symbol'])
#         # ...
#         trade['value'] = trade['price']*trade['qty_dir']*-1
#         # trade['buy_value'] = 0
#         # trade['sell_value'] = 0
#         trade['turnover'] = 0
#         # ...
#         trade['system_timestamp'] = datetime.datetime.now()
#         trade['note'] = note
#         if signal_number is not None: trade['signal_number'] =  signal_number
#         print('PSEUDOTRADE:', trade)
#         self.pseudo_trades.append(trade)
#         self.trade_count += 1
#         return (True, price)
               
        
#     def place_trade(self, timestamp, action, qty, symbol, price=None, note="", signal_number=None):
#         trade = {}
#         # print('yay trade')
#         trade['uid'] = self.uid
#         # ...
#         trade['timestamp'] = timestamp
#         # trade['dte'] = self.get_dte(timestamp, symbol)
#         # ...
#         trade['action'] = action
#         if action == 'BUY':
#             action_int = 1
#         elif action == 'SELL':
#             action_int = -1
#         trade['action_int'] = action_int
#         # ...
#         #trade['action_int'] = action_int
#         trade['qty'] = int(qty * self.weight)
#         trade['qty_dir'] = int(qty * action_int * self.weight)
#         trade['symbol'] = symbol
#         # ...
#         price_provided = True
#         if price is None:
#             try:
#                 price = float(self.get_tick(timestamp, symbol)['c'])
#                 price_provided = False
#             except Exception as e:
#                 #print(f'ERRTRADE: {e}')
#                 return (False, np.nan)
#         price = float(price)
#         if np.isnan(price):
#             return (False, np.nan)            
#         if price <= 0:
#             print(f'ERR: trade price is <= 0 : {price}')
#             return (False, price) 
#         trade['price'] = price
#         trade['price_provided'] = price_provided    
#         #trade['value'] = 
#         if trade['symbol'] not in self.positions:
#             self.positions[trade['symbol']] = 0
#         self.positions[trade['symbol']] += trade['qty_dir']
#         if self.positions[trade['symbol']] == 0:
#             self.positions.pop(trade['symbol'])
#         # ...
#         trade['value'] = trade['price']*trade['qty_dir']*-1
#         # trade['buy_value'] = 0
#         # trade['sell_value'] = 0
#         trade['turnover'] = abs(trade['value'])
#         # ...
#         trade['system_timestamp'] = datetime.datetime.now()
#         trade['note'] = note
#         if signal_number is not None: trade['signal_number'] =  signal_number
#         print('NEWTRADE:', trade)
#         self.trades.append(trade)
#         return (True, price)
    
#     def place_spread_trade(
#         self, timestamp, action, qty, symbol_X=None, symbol_Y=None, price_X=None, price_Y=None, note="", signal_number=None
#     ):
#         try:
#             if price_X is None and symbol_X is not None:
#                 price_X = float(self.get_tick(timestamp, symbol_X)['c'])
#             if price_Y is None and symbol_Y is not None:
#                 price_Y = float(self.get_tick(timestamp, symbol_Y)['c'])
#         except Exception as e:
#             print(f'SPREADTRADE ERR: {e}')
#             return (False, price_X, price_Y)
#         if symbol_X is None:
#             price_X = np.nan
#         if symbol_Y is None:
#             price_Y = np.nan
#         #print(price_X, type(price_X), price_Y, type(price_Y))
#         if (np.isnan(price_X) and symbol_X is not None) or (np.isnan(price_Y) and symbol_Y is not None):
#             return (False, price_X, price_Y)
#         if action == 'BUY':
#             if symbol_X is not None:
#                 success_X, _ = self.place_trade(timestamp, 'BUY', qty, symbol_X, price_X, "bX_"+note, signal_number)
#             else:
#                 success_X = True
#             if symbol_Y is not None:
#                 success_Y, _ = self.place_trade(timestamp, 'SELL', qty, symbol_Y, price_Y, "sY_"+note, signal_number)
#             else:
#                 success_Y = True 
#         elif action == 'SELL':
#             if symbol_X is not None:
#                 success_X, _ = self.place_trade(timestamp, 'SELL', qty, symbol_X, price_X, "sX_"+note, signal_number)
#             else:
#                 success_X = True
#             if symbol_Y is not None:
#                 success_Y, _ = self.place_trade(timestamp, 'BUY', qty, symbol_Y, price_Y, "bY_"+note, signal_number)
#             else:
#                 success_Y = True
#         assert success_X == True and success_Y == True
#         return (True, price_X, price_Y)
        
        
    
#     #######################################
#     #### DEFINE THE FOLLOWING IN STRAT ####
#     #######################################
    
#     def on_start(self):
#         """
#         Things to do at start of strategy
#         Ex: arrays to store continuous price series
#         """
#         pass

#     def on_stop(self):
#         """
#         Things to do at stop of strategy
#         Ex: log some information 
#         """
#         pass
    
#     def on_new_day(self):
#         """
#         Things to do at start of new day
#         Ex: array to store VWAP as VWAP is new every day
#         """
#         pass
    
#     def on_event(self):
#         """
#         Things to do on each new event i.e. every second
#         """
#         pass

#     def on_bar_complete(self):
#         """
#         Things to do on each new bar i.e. every minute 
#         """
#         pass

import datetime, os, glob, re, json
import direct_redis
import pandas as pd
from utils.utility import holidays
import numpy as np
import matplotlib.pyplot as plt
import time, traceback, math, sys
import scipy.stats as si
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.append("./cython_modules")
from cython_modules.payoff import single_strike_payoff_cal

from utils.utility import stock_tickers

r = direct_redis.DirectRedis()

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

indexes = ('BANKNIFTY', 'NIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX', 'BANKEX')

def get_underlying(symbol):
    if symbol.startswith('NIFTY') : return 'NIFTY'
    elif symbol.startswith('BANKNIFTY') : return 'BANKNIFTY'
    elif symbol.startswith('FINNIFTY') : return 'FINNIFTY'
    elif symbol.startswith('MIDCPNIFTY') : return 'MIDCPNIFTY'
    elif symbol.startswith('SENSEX') : return 'SENSEX'
    elif symbol.startswith('BANKEX') : return 'BANKEX'
    return None

def parse_strike_from_symbol(symbol):
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

# equity_data = r.hgetall('eq_daily_ohlc')
#########################

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
            underlying = 'BANKEX'
        for ticker in stock_tickers:
            if symbol.startswith(ticker) : underlying =  ticker
        if underlying == '':
            print(f'WARNING: {symbol} underlying not found')
        # ...
        if symbol.startswith("NIFTY-"):
            return symbol
        else:
            return int(float(symbol.strip(underlying).strip('CE').strip('PE')[6:]))

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
        ticks = r.hgetall(symbol)
        ticks = pd.DataFrame.from_dict(ticks, orient='index')

        # ticks['timestamp'] = pd.to_datetime(ticks['timestamp'])
        ticks['timestamp'] = pd.to_datetime(ticks.index)

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

    def get_tick(self, timestamp, symbol):
        if symbol is None:
            print('SYMBOL IS NONE')
            return {'o': np.nan, 'h': np.nan, 'l': np.nan, 'c': np.nan, 'v': np.nan, 'oi': np.nan, 'timestamp': None}
        try:
            tick = r.hget(f'tick_{timestamp}', symbol)
            if isinstance(tick, (bytes, str)):
                tick = json.loads(tick)
            tick['timestamp'] = timestamp
            tick['timestamp']=pd.to_datetime(tick['timestamp'])
            
        except Exception as e:
            print(f"Couldn't find {symbol} tick in TIME AXIS, trying old tick {timestamp}")
            try:
                all_ticks = self.get_all_ticks_by_symbol(symbol)
                all_ticks['timestamp']=pd.to_datetime(all_ticks['timestamp'])
                tick = all_ticks[all_ticks['timestamp']<=timestamp].iloc[-1]
                #tick['timestamp'] = timestamp
            except Exception as e:
                print(f'GETTICK ERR: SYMBOL NOT AVAILABLE {symbol}, {timestamp} {e} ')
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
            underlying = get_underlying(trade_symbol)
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
        underlying = get_underlying(symbol)
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

    def get_underlying(self, symbol):
        underlying_options = indexes
        for underlying in underlying_options:
            if symbol.startswith(underlying):
                return underlying
        print('SYMBOL NOT DEFINED')
        return None  # Explicitl    

    def get_option_delta_hv(self, timestamp, symbol):
        # tick = self.get_tick(timestamp, symbol)
        underlying = get_underlying(symbol)
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

class EventInterface(DataInterface):

    def __init__(self):
        # ...
        super().__init__()
        # ...
        self.weight = 1
        # ...
        self.event = None
        self.last_event = None
        # ...
        self.trades = []
        self.all_trades = []
        self.positions = {}
        self.meta_data = []
        #self.positions_SANITY = {}

    def get_mtm(self):
        total_mtm = 0
        pnls = []
        rms_df = []

        unique_symbols = {x['symbol'] for x in self.trades} #pd.Series([x['symbol'] for x in self.trades]).unique()
        # unique_symbols = [x for x in unique_symbols if self.parse_date_from_symbol(x) >= self.now.date()]

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

                ltp = self.get_tick(self.now, s)['c'] #r.get(f'ltp.{s}')
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
        self.now = event['timestamp']
        #print(self.now)
        #return
        # ...
        self.check_if_new_day()
        #self.run_sanity_check()
        # SANITY CHECK: don't process stale timestamp
        if self.last_event is not None:
            if self.last_event['timestamp']>=self.event['timestamp']:
                print(self.last_event['timestamp'], self.event['timestamp'])
                print('STALE TIMESTAMP !!!')
                return
        #self.on_event()
        if self.event['bar_complete']:
            # self.mtm = self.get_mtm()
            # self.margin, self.var = self.get_curr_margin()
            # self.meta_data.append({
            #     'timestamp': self.now,
            #     'mtm': self.mtm,
            #     'margin': self.margin,
            #     'var': self.var,
            # })
            self.on_bar_complete()
            
        # SANITY CHECK: no positions open after stop time
        if self.event['timestamp'].time() >= self.stop_time:
            try:
                assert len(self.positions) == 0
            except AssertionError:
                raise AssertionError(f'Trades open after stop time @ {self.now} @ \n TRADES: {pd.DataFrame(self.trades)}\n POSITIONS {self.positions}')
        # ...
        self.last_event = self.event

    def get_active_trades(self):
        active_trades = []
        sq_off_trades = []
        pos = {}
        for trade in self.trades:
            symbol = trade['symbol']
            
            # INITIALIZE NEW POSITION
            if symbol not in pos:
                pos[symbol] = 0
            
            # ADD QTY
            pos[symbol] += trade['qty_dir']
            
            if pos[symbol] != 0:
                active_trades.append(trade)

            # IF POSITION IS CLOSED
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
            # SINCE ALL THE POSITIONS WILL BE OF SAME INDEX TAKE ANY TRADE TO FIND UNDERLYING AND GET LOT SIZE
            underlying = self.get_underlying(active_trades[0]['symbol'])
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
                # final_df['strike'] = final_df['symbol'].apply(lambda x: strike_from_symbol(x))
                final_df['strike'] = final_df['symbol'].map(parse_strike_from_symbol)
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
                # elm_margin = total_hedged_lot*spot_price*0.02*lot_size
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
        elif self.last_event['timestamp'].date() != self.event['timestamp'].date():
            self.new_day = True
        # ...
        if self.new_day:
            # check if no open positions
            if len(self.trades) > 0:
                # assert 0 == pd.DataFrame(self.trades).groupby('symbol')['qty_dir'].sum().abs().sum()
                self.all_trades += self.trades
                self.trades = []
            self.positions = {}
            # ...
            self.on_new_day()        
        
    def place_trade(self, timestamp, action, qty, symbol, price=None, note="", signal_number=None):
        trade = {}
        # print('yay trade')
        trade['uid'] = self.uid
        # ...
        trade['timestamp'] = timestamp
        trade['dte'] = self.get_dte(timestamp, symbol)
        # ...
        trade['action'] = action
        if action == 'BUY':
            action_int = 1
        elif action == 'SELL':
            action_int = -1
        trade['action_int'] = action_int
        # ...
        #trade['action_int'] = action_int
        trade['qty'] = int(qty * self.weight)
        trade['qty_dir'] = int(qty * action_int * self.weight)
        trade['symbol'] = symbol
        # ...
        price_provided = True
        if price is None:
            try:
                price = float(self.get_tick(timestamp, symbol)['c'])
                price_provided = False
            except Exception as e:
                #print(f'ERRTRADE: {e}')
                return (False, np.nan)
        price = float(price)
        if np.isnan(price):
            return (False, np.nan)            
        if price <= 0:
            print(f'ERR: trade price is <= 0 : {timestamp} {symbol} {price}')
            return (False, price) 
        trade['price'] = price
        trade['price_provided'] = price_provided    
        #trade['value'] = 
        if trade['symbol'] not in self.positions:
            self.positions[trade['symbol']] = 0
        self.positions[trade['symbol']] += trade['qty_dir']
        if self.positions[trade['symbol']] == 0:
            self.positions.pop(trade['symbol'])
        # ...
        trade['value'] = trade['price']*trade['qty_dir']*-1
        # trade['buy_value'] = 0
        # trade['sell_value'] = 0
        trade['turnover'] = abs(trade['value'])
        # ...
        trade['system_timestamp'] = datetime.datetime.now()
        trade['note'] = note
        if signal_number is not None: trade['signal_number'] =  signal_number
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
        #print(price_X, type(price_X), price_Y, type(price_Y))
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
        
        
    
    #######################################
    #### DEFINE THE FOLLOWING IN STRAT ####
    #######################################
    
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

    def __init__(self):
        # ...
        super().__init__()
        # ...
        self.meta_data = []
        self.trade_count = 0
        self.weight = 1
        # ...
        self.event = None
        self.last_event = None
        # ...
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

        unique_symbols = {x['symbol'] for x in self.trades} #pd.Series([x['symbol'] for x in self.trades]).unique()
        # unique_symbols = [x for x in unique_symbols if self.parse_date_from_symbol(x) >= self.now.date()]
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

                ltp = self.get_tick(self.now, s)['c'] #r.get(f'ltp.{s}')
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
        self.now = event['timestamp']
        
        # ...
        self.check_if_new_day()
        #self.run_sanity_check()
        # SANITY CHECK: don't process stale timestamp
        if self.last_event is not None:
            if self.last_event['timestamp']>=self.event['timestamp']:
                print('STALE TIMESTAMP !!!')
                return
        #self.on_event()
        if self.event['bar_complete']:
            
            if self.now.time() == datetime.time(15, 29):
                if len(self.positions) > 0:
                    # positions are open place a reverse pseudo trade
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
                # self.mtm = self.get_mtm()
                # self.margin, self.var = self.get_curr_margin()
                # self.meta_data.append({
                #     'timestamp': self.now,
                #     'mtm': self.mtm,
                #     'margin': self.margin,
                #     'var': self.var,
                # })        
                self.on_bar_complete()
                   
        # ...
        self.last_event = self.event

    def get_active_trades(self):
        active_trades = []
        sq_off_trades = []
        pos = {}
        for trade in self.trades:
            symbol = trade['symbol']
            
            # INITIALIZE NEW POSITION
            if symbol not in pos:
                pos[symbol] = 0
            
            # ADD QTY
            pos[symbol] += trade['qty_dir']
            
            if pos[symbol] != 0:
                active_trades.append(trade)

            # IF POSITION IS CLOSED
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
            # SINCE ALL THE POSITIONS WILL BE OF SAME INDEX TAKE ANY TRADE TO FIND UNDERLYING AND GET LOT SIZE
            underlying = self.get_underlying(active_trades[0]['symbol'])
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
                # final_df['strike'] = final_df['symbol'].apply(lambda x: strike_from_symbol(x))
                final_df['strike'] = final_df['symbol'].map(parse_strike_from_symbol)
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
                # elm_margin = total_hedged_lot*spot_price*0.02*lot_size
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
        elif self.last_event['timestamp'].date() != self.event['timestamp'].date():
            self.new_day = True
        # ...
        if self.new_day:
            if len(self.trades) > 0:
                self.all_trades += self.trades
                self.trades = []
                
            if len(self.pseudo_trades) > 0:
                pseudo_trades = self.pseudo_trades.copy()
                self.pseudo_trades = []
                for trade in pseudo_trades:
                    # self.all_trades.append(trade)
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

            # ...
            self.on_new_day() 

    def place_pseudo_trade(self, timestamp, action, qty, symbol, price=None, note="", signal_number=None):
        trade = {}
        # print('yay trade')
        trade['uid'] = self.uid
        # ...
        trade['timestamp'] = timestamp
        # trade['dte'] = self.get_dte(timestamp, symbol)
        # ...
        trade['action'] = action
        if action == 'BUY':
            action_int = 1
        elif action == 'SELL':
            action_int = -1
        trade['action_int'] = action_int
        # ...
        #trade['action_int'] = action_int
        trade['qty'] = int(qty * self.weight)
        trade['qty_dir'] = int(qty * action_int * self.weight)
        trade['symbol'] = symbol
        # ...
        price_provided = True
        if price is None:
            try:
                price = float(self.get_tick(timestamp, symbol)['c'])
                price_provided = False
            except Exception as e:
                #print(f'ERRTRADE: {e}')
                return (False, np.nan)
        price = float(price)
        if np.isnan(price):
            return (False, np.nan)            
        if price <= 0:
            print(f'ERR: trade price is <= 0 : {price}')
            return (False, price) 
        trade['price'] = price
        trade['price_provided'] = price_provided    
        #trade['value'] = 
        # if trade['symbol'] not in self.positions:
        #     self.positions[trade['symbol']] = 0
        # self.positions[trade['symbol']] += trade['qty_dir']
        # if self.positions[trade['symbol']] == 0:
        #     self.positions.pop(trade['symbol'])
        # ...
        trade['value'] = trade['price']*trade['qty_dir']*-1
        # trade['buy_value'] = 0
        # trade['sell_value'] = 0
        trade['turnover'] = 0
        # ...
        trade['system_timestamp'] = datetime.datetime.now()
        trade['note'] = note
        if signal_number is not None: trade['signal_number'] =  signal_number
        print('PSEUDOTRADE:', trade)
        self.pseudo_trades.append(trade)
        self.trade_count += 1
        return (True, price)
               
        
    def place_trade(self, timestamp, action, qty, symbol, price=None, note="", signal_number=None):
        trade = {}
        # print('yay trade')
        trade['uid'] = self.uid
        # ...
        trade['timestamp'] = timestamp
        # trade['dte'] = self.get_dte(timestamp, symbol)
        # ...
        trade['action'] = action
        if action == 'BUY':
            action_int = 1
        elif action == 'SELL':
            action_int = -1
        trade['action_int'] = action_int
        # ...
        #trade['action_int'] = action_int
        trade['qty'] = int(qty * self.weight)
        trade['qty_dir'] = int(qty * action_int * self.weight)
        trade['symbol'] = symbol
        # ...
        price_provided = True
        if price is None:
            try:
                price = float(self.get_tick(timestamp, symbol)['c'])
                price_provided = False
            except Exception as e:
                #print(f'ERRTRADE: {e}')
                return (False, np.nan)
        price = float(price)
        if np.isnan(price):
            return (False, np.nan)            
        if price <= 0:
            print(f'ERR: trade price is <= 0 : {price}')
            return (False, price) 
        trade['price'] = price
        trade['price_provided'] = price_provided    
        #trade['value'] = 
        if trade['symbol'] not in self.positions:
            self.positions[trade['symbol']] = 0
        self.positions[trade['symbol']] += trade['qty_dir']
        if self.positions[trade['symbol']] == 0:
            self.positions.pop(trade['symbol'])
        # ...
        trade['value'] = trade['price']*trade['qty_dir']*-1
        # trade['buy_value'] = 0
        # trade['sell_value'] = 0
        trade['turnover'] = abs(trade['value'])
        # ...
        trade['system_timestamp'] = datetime.datetime.now()
        trade['note'] = note
        if signal_number is not None: trade['signal_number'] =  signal_number
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
        #print(price_X, type(price_X), price_Y, type(price_Y))
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
        
        
    
    #######################################
    #### DEFINE THE FOLLOWING IN STRAT ####
    #######################################
    
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
