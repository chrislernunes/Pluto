# import datetime
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# from utils.definitions import *

# if REDIS:
#     from engine.ems import EventInterfacePositional, lot_size_dict, strike_diff_dict
# else:
#     from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

# class c(EventInterfacePositional):
#     def __init__(self, conn=None):
#         super().__init__(conn)
#         self.strat_id = self.__class__.__name__.lower()
#         self.uid = ""
#         self.var_level = 0.1
#         self.margin = 1e7
#         self.positions_df = None
#         self.margin_df_nifty = None
#         self.margin_df_sensex = None
#         self.hedge_positions = []

#     def set_params_from_uid(self, uid):
#         self.uid = uid
#         parts = uid.split('_', 4)
#         assert parts[0] == self.strat_id
#         self.var_level = float(parts[1])
#         self.margin = float(parts[2])
#         self.max_hedge = float(parts[3])
#         self.basket_id = parts[4]

#     def on_new_day(self):
#         self.start_time = datetime.time(9, 15)
#         self.end_time = datetime.time(15, 30)
#         self.find_dte = True
#         if self.positions_df is None:
#             self.positions_df = self.load_positions_df()
#         if self.margin_df_nifty is None:
#             self.margin_df_nifty = self.load_eq_curve("NIFTY")
#         if self.margin_df_sensex is None:    
#             self.margin_df_sensex = self.load_eq_curve("SENSEX")
#         self.updated_margin = {}
#         try:
#             self.updated_margin["NIFTY"] = self.margin + self.margin_df_nifty[self.margin_df_nifty['date'] <= self.now.date()]['cumulative_value'].iloc[-1]
#         except:
#             print("No trades yet for NIFTY")
#         try:
#             self.updated_margin["SENSEX"] = self.margin + self.margin_df_sensex[self.margin_df_sensex['date'] <= self.now.date()]['cumulative_value'].iloc[-1]
#         except:
#             print("No trades yet for SENSEX")
#     def load_positions_df(self):
#         path = f'/home/mridul/jupiter/storage/portfolio/{self.basket_id}/combined_active_positions.csv'
#         df = pd.read_csv(path)
#         df['ts'] = pd.to_datetime(df['ts'])
#         df['date'] = df['ts'].dt.date
#         df = df.sort_values('ts')

#         # # Keep rows where timestamp is the max for that date
#         # df = df[df['ts'] == df.groupby('date')['ts'].transform('max')]

#         # # Drop the helper 'date' column if you want
#         # df = df.drop(columns=['date'])

#         # # Change time to 15:20 for all entries
#         # df['ts'] = df['ts'].dt.floor('d') + pd.Timedelta(hours=15, minutes=20)
#         return df

#     def load_eq_curve(self,underlying):
#         path = f'/home/mridul/jupiter/storage/portfolio/{self.basket_id}/combined_tradebook.csv'
#         df = pd.read_csv(path,usecols=['ts','value','symbol'], parse_dates=['ts'])
#         df = df[df['symbol'].str.startswith(underlying)]
#         df['date'] = df['ts'].dt.date
#         # Group by date and aggregate value by sum
#         final = df.groupby('date').agg({'value': 'sum'}).reset_index()
#         final["date"] = pd.to_datetime(final["date"]).dt.date
#         final = final.sort_values('date')
#         # Create a column with cumulative sum of value
#         final['cumulative_value'] = final['value'].cumsum()
#         return final

#     def get_symbol_params(self, symbol):
#         if symbol.startswith("NIFTY"):
#             return "NIFTY", "NIFTYSPOT", lot_size_dict["NIFTY"], strike_diff_dict["NIFTY"]
#         if symbol.startswith("BANKNIFTY"):
#             return "BANKNIFTY", "BANKNIFTYSPOT", lot_size_dict["BANKNIFTY"], strike_diff_dict["BANKNIFTY"]
#         if symbol.startswith("FINNIFTY"):
#             return "FINNIFTY", "FINNIFTYSPOT", lot_size_dict["FINNIFTY"], strike_diff_dict["FINNIFTY"]
#         elif symbol.startswith("SENSEX"):
#             return "SENSEX", "SENSEXSPOT", lot_size_dict["SENSEX"], strike_diff_dict["SENSEX"]
#         else:
#             raise ValueError(f"Unknown underlying in symbol: {symbol}")

#     def on_bar_complete(self):
#         now = self.now
#         if self.positions_df is None or not (self.start_time <= now.time() <= self.end_time):
#             return

#         # Close all current hedges before evaluating new hedge
#         if self.now.time() == self.start_time:
#             self.close_all_hedges(now)

#         # Get exact match for this timestamp
#         current_positions = self.positions_df[self.positions_df['ts'] == self.now]
#         if current_positions.empty:
#             # print(self.now, "Current Positions are empty")
#             return  # No action if no positions at this timestamp

#         grouped = defaultdict(list)
#         for _, row in current_positions.iterrows():
#             symbol = row['symbol']
#             qty = row['qty']
#             current_price = self.get_tick(self.now,symbol)['c'] 
#             premium = row['qty']*current_price
#             grouped[symbol].append({'qty': qty, 'premium': premium})

#         positions_by_underlying = defaultdict(lambda: defaultdict(list))

#         if self.hedge_positions == []:
#             print(self.now, "No existing hedge positions - First Strike Evaluation")
#         else:
#             for hedge in self.hedge_positions:
#                 symbol = hedge["symbol"]
#                 qty = hedge["qty"]
#                 current_price = self.get_tick(self.now,symbol)['c'] 
#                 premium = qty*current_price
#                 grouped[symbol].append({'qty': qty, 'premium': premium})


#         for symbol in grouped:
#             try:
#                 underlying, spot_sym, lot_size, strike_diff = self.get_symbol_params(symbol)
#                 opt_type = symbol[-2:]
#                 positions_by_underlying[underlying][opt_type].append((symbol, grouped[symbol]))
#             except:
#                 continue

#         for underlying in positions_by_underlying:
#             self.process_hedge(now, underlying, positions_by_underlying[underlying])

#         # Store current net positions for reference and manage hedge rollover
#         # self.roll_over_hedges_if_expiry()

#     def close_all_hedges(self, now):
#         new_hedge_positions = []
#         for hedge in self.hedge_positions:
#             symbol = hedge["symbol"]
#             qty = hedge["qty"]
#             if qty > 0:
#                 success, _ = self.place_trade(now, "SELL", qty, symbol, note="CLOSE HEDGE")
#                 if not success:
#                     print(f"Failed to close hedge for {symbol}, keeping in list")
#                     new_hedge_positions.append(hedge)  # retain if not closed
#         self.hedge_positions = new_hedge_positions

#     def process_hedge(self, now, underlying, symbol_list):
#         try:
#             spot = self.get_tick(now, f"{underlying}SPOT")['c']
#         except:
#             return

#         lot_size = lot_size_dict[underlying]
#         strike_diff = strike_diff_dict[underlying]
#         max_loss_allowed = self.var_level * self.updated_margin[underlying]
        
#         if self.find_dte:
#             self.dte = self.get_dte_by_underlying(now, underlying)
#             self.find_dte = False
        
#         # Check if hedge is even needed
#         net_qty_ce = sum(pos['qty'] for symbol, entries in symbol_list["CE"] for pos in entries)
#         net_qty_pe = sum(pos['qty'] for symbol, entries in symbol_list["PE"] for pos in entries)
#         # print(net_qty_ce, net_qty_pe, now)
#         # print(symbol_list)
#         if net_qty_ce >= 0 and net_qty_pe >= 0:
#             print("Net long or flat → no hedge needed", underlying) 
#             return

#         # Simulate worst-case loss
#         hedge_found_ce = False
#         hedge_found_pe = False
#         pct_move = self.max_hedge
#         if True:
#             moved_spot = spot * (1 + pct_move)
#             self.total_loss_up = 0
#             for opt_type in ["CE","PE"]:
#                 for symbol, entries in symbol_list[opt_type]:
#                     strike = self.parse_strike_from_symbol(symbol)
#                     for pos in entries:
#                         qty = pos['qty']
#                         premium = abs(pos['premium']) / abs(qty)
#                         intrinsic = max(moved_spot - strike, 0) if opt_type == 'CE' else max(strike - moved_spot, 0)
#                         if qty < 0:
#                             loss = (intrinsic - premium) * abs(qty)
#                         else:
#                             loss = (premium - intrinsic) * abs(qty)
#                         self.total_loss_up += loss
#             if self.total_loss_up >= max_loss_allowed:
#                 hedge_found_ce = True
                
#         if True:
#             moved_spot = spot * (1 - pct_move)
#             self.total_loss_down = 0
#             for opt_type in ["CE","PE"]:
#                 for symbol, entries in symbol_list[opt_type]:
#                     strike = self.parse_strike_from_symbol(symbol)
#                     for pos in entries:
#                         qty = pos['qty']
#                         premium = abs(pos['premium']) / abs(qty)
#                         intrinsic = max(moved_spot - strike, 0) if opt_type == 'CE' else max(strike - moved_spot, 0)
#                         if qty < 0:
#                             loss = (intrinsic - premium) * abs(qty)
#                         else:
#                             loss = (premium - intrinsic) * abs(qty)
#                         self.total_loss_down += loss
#             if self.total_loss_down >= max_loss_allowed:
#                 hedge_found_pe = True

#         # log_file = "/home/mridul/jupiter/simulation_logssss.txt"
#         #     # Log completion
#         # with open(log_file, 'a') as f:
#         #     f.write(f"{self.now},{spot}, {self.updated_margin},{self.total_loss_down},{self.total_loss_up},{max_loss_allowed}\n")
                

#         # Choose strike for hedge
#         if hedge_found_ce:
#             loss_pct_up = self.total_loss_up/self.updated_margin[underlying]
#             self.hedge_pct_ce = self.var_level*self.max_hedge/loss_pct_up
#             hedge_strike_ce = int(np.ceil((spot * (1 + self.hedge_pct_ce)) / strike_diff) * strike_diff)
#             strikes_away_ce = min(abs(hedge_strike_ce - spot) // strike_diff, 149)
#             if self.dte == 0:
#                 expiry = 1
#             else:
#                 expiry = 0
#             hedge_symbol_ce = self.find_symbol_by_moneyness_attempt(now, underlying, expiry, "CE", strikes_away_ce)
#             total_qty_ce = abs(net_qty_ce)
#             # log_file = "/home/mridul/jupiter/simulation_logssss.txt"
#             # Log completion
#             # with open(log_file, 'a') as f:
#                 # f.write(f"{self.now}, {hedge_symbol_ce}, {spot}, {hedge_strike_ce}, {self.updated_margin},{self.total_loss_down},{self.total_loss_up},{max_loss_allowed}\n")
#             if hedge_symbol_ce is not None:
#                 success, _ = self.place_trade(now, 'BUY', total_qty_ce, hedge_symbol_ce, note=f"ADD HEDGE {underlying} CE")
#                 if success:
#                     self.hedge_positions.append({"symbol": hedge_symbol_ce, "qty": total_qty_ce})
#                 else:
#                     print(f"COULD NOT PLACE HEDGE TRADE FOR {hedge_symbol_ce}")
                
#         if hedge_found_pe:
#             loss_pct_down = self.total_loss_down/self.updated_margin[underlying]
#             self.hedge_pct_pe = self.var_level*self.max_hedge/loss_pct_down
#             hedge_strike_pe = int(np.floor((spot * (1 - self.hedge_pct_pe)) / strike_diff) * strike_diff)
#             strikes_away_pe = min(abs(hedge_strike_pe - spot) // strike_diff, 149)
#             if self.dte == 0:
#                 expiry = 1
#             else:
#                 expiry = 0
#             print(loss_pct_down,self.hedge_pct_pe,hedge_strike_pe,strikes_away_pe)
#             hedge_symbol_pe = self.find_symbol_by_moneyness_attempt(now, underlying, expiry, "PE", strikes_away_pe)
#             total_qty_pe = abs(net_qty_pe)
#             # log_file = "/home/mridul/jupiter/simulation_logssss.txt"
#             # # Log completion
#             # with open(log_file, 'a') as f:
#             #     f.write(f"{self.now}, {hedge_symbol_pe}, {spot}, {hedge_strike_pe}, {self.updated_margin},{self.total_loss_down},{self.total_loss_up},{max_loss_allowed} \n")
#             if hedge_symbol_pe is not None:
#                 success, _ = self.place_trade(now, 'BUY', total_qty_pe, hedge_symbol_pe, note=f"ADD HEDGE {underlying} PE")
#                 if success:
#                     self.hedge_positions.append({"symbol": hedge_symbol_pe, "qty": total_qty_pe})
#                 else:            
#                     print(f"COULD NOT PLACE HEDGE TRADE FOR {hedge_symbol_pe}")
    

# import datetime
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# from utils.definitions import *

# if REDIS:
#     from engine.ems import EventInterfacePositional, lot_size_dict, strike_diff_dict
# else:
#     from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

# class HEDGEINTRADAYLTP(EventInterfacePositional):
#     def __init__(self, conn=None):
#         super().__init__(conn)
#         self.strat_id = self.__class__.__name__.lower()
#         self.uid = ""
#         self.var_level = 0.1
#         self.margin = 1e7
#         self.positions_df = None
#         self.margin_df_nifty = None
#         self.margin_df_sensex = None
#         self.hedge_positions = {}  # {underlying: {"CE": {...}, "PE": {...}}}
#         self.hedge_initialized_today = False

#     def set_params_from_uid(self, uid):
#         self.uid = uid
#         parts = uid.split('_', 4)
#         assert parts[0] == self.strat_id
#         self.var_level = float(parts[1])
#         self.margin = float(parts[2])
#         self.max_hedge = float(parts[3])
#         self.basket_id = parts[4]

#     def on_new_day(self):
#         self.start_time = datetime.time(9, 15)
#         self.end_time = datetime.time(15, 30)
#         self.find_dte = True
#         self.hedge_initialized_today = False
#         self.hedge_closed_today = False
#         if self.positions_df is None:
#             self.positions_df = self.load_positions_df()
#         if self.margin_df_nifty is None:
#             self.margin_df_nifty = self.load_eq_curve("NIFTY")
#         if self.margin_df_sensex is None:    
#             self.margin_df_sensex = self.load_eq_curve("SENSEX")
#         self.updated_margin = {}
#         try:
#             self.updated_margin["NIFTY"] = self.margin + self.margin_df_nifty[self.margin_df_nifty['date'] <= self.now.date()]['cumulative_value'].iloc[-1]
#         except:
#             print("No trades yet for NIFTY")
#         try:
#             self.updated_margin["SENSEX"] = self.margin + self.margin_df_sensex[self.margin_df_sensex['date'] <= self.now.date()]['cumulative_value'].iloc[-1]
#         except:
#             print("No trades yet for SENSEX")
#     def load_positions_df(self):
#         path = f'/home/mridul/jupiter/storage/portfolio/{self.basket_id}/combined_active_positions.csv'
#         df = pd.read_csv(path)
#         df['ts'] = pd.to_datetime(df['ts'])
#         df['date'] = df['ts'].dt.date
#         df = df.sort_values('ts')

#         # # Keep rows where timestamp is the max for that date
#         # df = df[df['ts'] == df.groupby('date')['ts'].transform('max')]

#         # # Drop the helper 'date' column if you want
#         # df = df.drop(columns=['date'])

#         # # Change time to 15:20 for all entries
#         # df['ts'] = df['ts'].dt.floor('d') + pd.Timedelta(hours=15, minutes=20)
#         return df

#     def load_eq_curve(self,underlying):
#         path = f'/home/mridul/jupiter/storage/portfolio/{self.basket_id}/combined_tradebook.csv'
#         df = pd.read_csv(path,usecols=['ts','value','symbol'], parse_dates=['ts'])
#         df = df[df['symbol'].str.startswith(underlying)]
#         df['date'] = df['ts'].dt.date
#         # Group by date and aggregate value by sum
#         final = df.groupby('date').agg({'value': 'sum'}).reset_index()
#         final["date"] = pd.to_datetime(final["date"]).dt.date
#         final = final.sort_values('date')
#         # Create a column with cumulative sum of value
#         final['cumulative_value'] = final['value'].cumsum()
#         return final

#     def get_symbol_params(self, symbol):
#         if symbol.startswith("NIFTY"):
#             return "NIFTY", "NIFTYSPOT", lot_size_dict["NIFTY"], strike_diff_dict["NIFTY"]
#         if symbol.startswith("BANKNIFTY"):
#             return "BANKNIFTY", "BANKNIFTYSPOT", lot_size_dict["BANKNIFTY"], strike_diff_dict["BANKNIFTY"]
#         if symbol.startswith("FINNIFTY"):
#             return "FINNIFTY", "FINNIFTYSPOT", lot_size_dict["FINNIFTY"], strike_diff_dict["FINNIFTY"]
#         elif symbol.startswith("SENSEX"):
#             return "SENSEX", "SENSEXSPOT", lot_size_dict["SENSEX"], strike_diff_dict["SENSEX"]
#         else:
#             raise ValueError(f"Unknown underlying in symbol: {symbol}")

#     def on_bar_complete(self):
#         now = self.now
#         if self.positions_df is None or not (self.start_time <= now.time() <= self.end_time):
#             return

#         # Close all current hedges at start time and reset
#         if self.now.time() >= self.start_time and not self.hedge_closed_today:
#             self.close_all_hedges(now)
#             self.hedge_closed_today = True
#             return  # Don't process hedges on the same bar as close

#         # Get exact match for this timestamp
#         current_positions = self.positions_df[self.positions_df['ts'] == self.now]
#         if current_positions.empty:
#             return  # No action if no positions at this timestamp

#         # Build grouped positions
#         grouped = defaultdict(list)
#         for _, row in current_positions.iterrows():
#             symbol = row['symbol']
#             qty = row['qty']
#             current_price = self.get_tick(self.now, symbol)['c'] 
#             premium = row['qty'] * current_price
#             grouped[symbol].append({'qty': qty, 'premium': premium})

#         positions_by_underlying = defaultdict(lambda: defaultdict(list))

#         for symbol in grouped:
#             try:
#                 underlying, spot_sym, lot_size, strike_diff = self.get_symbol_params(symbol)
#                 opt_type = symbol[-2:]
#                 positions_by_underlying[underlying][opt_type].append((symbol, grouped[symbol]))
#             except:
#                 continue

#         # Initialize hedges once per day (after start_time close)
#         if not self.hedge_initialized_today:
#             for underlying in positions_by_underlying:
#                 self.initialize_hedge(now, underlying, positions_by_underlying[underlying])
#             self.hedge_initialized_today = True
#         else:
#             # Adjust existing hedge quantities to maintain VaR
#             for underlying in positions_by_underlying:
#                 self.adjust_hedge_qty(now, underlying, positions_by_underlying[underlying])

#     def close_all_hedges(self, now):
#         """Close all existing hedges and reset for fresh initialization"""
#         for underlying in list(self.hedge_positions.keys()):
#             for opt_type in ["CE", "PE"]:
#                 if opt_type in self.hedge_positions[underlying]:
#                     hedge_data = self.hedge_positions[underlying][opt_type]
#                     if hedge_data and hedge_data["qty"] > 0:
#                         symbol = hedge_data["symbol"]
#                         qty = hedge_data["qty"]
#                         success, _ = self.place_trade(now, "SELL", qty, symbol, note="CLOSE HEDGE")
#                         if success:
#                             print(f"Closed hedge: {symbol} qty {qty}")
#                             del self.hedge_positions[underlying][opt_type]
#                         else:
#                             print(f"Failed to close hedge for {symbol}")

#         for underlying in ['NIFTY', 'SENSEX']:
#             self.hedge_positions[underlying] = {"CE": None, "PE": None}
#         self.hedge_initialized_today = False

#     def initialize_hedge(self, now, underlying, symbol_list):
#         """Initialize hedge strikes and quantities at start of day"""
#         try:
#             spot = self.get_tick(now, f"{underlying}SPOT")['c']
#         except:
#             return

#         lot_size = lot_size_dict[underlying]
#         strike_diff = strike_diff_dict[underlying]
#         max_loss_allowed = self.var_level * self.updated_margin[underlying]
        
#         if self.find_dte:
#             self.dte = self.get_dte_by_underlying(now, underlying)
#             self.find_dte = False
        
#         # Check if hedge is needed
#         net_qty_ce = sum(pos['qty'] for symbol, entries in symbol_list["CE"] for pos in entries)
#         net_qty_pe = sum(pos['qty'] for symbol, entries in symbol_list["PE"] for pos in entries)
        
#         if net_qty_ce >= 0 and net_qty_pe >= 0:
#             print(f"{self.now} {underlying}: Net long or flat → no hedge needed")
#             return

#         # Initialize hedge_positions for this underlying if not exists
#         if underlying not in self.hedge_positions:
#             self.hedge_positions[underlying] = {"CE": None, "PE": None}

#         # Simulate worst-case loss scenarios
#         hedge_needed_ce = False
#         hedge_needed_pe = False
#         pct_move = self.max_hedge

#         # Upside loss scenario
#         moved_spot = spot * (1 + pct_move)
#         self.total_loss_up = 0
#         for opt_type in ["CE", "PE"]:
#             for symbol, entries in symbol_list[opt_type]:
#                 strike = self.parse_strike_from_symbol(symbol)
#                 for pos in entries:
#                     qty = pos['qty']
#                     premium = abs(pos['premium']) / abs(qty) if qty != 0 else 0
#                     intrinsic = max(moved_spot - strike, 0) if opt_type == 'CE' else max(strike - moved_spot, 0)
#                     if qty < 0:
#                         loss = (intrinsic - premium) * abs(qty)
#                     else:
#                         loss = (premium - intrinsic) * abs(qty)
#                     self.total_loss_up += loss
        
#         if self.total_loss_up >= max_loss_allowed:
#             hedge_needed_ce = True

#         # Downside loss scenario
#         moved_spot = spot * (1 - pct_move)
#         self.total_loss_down = 0
#         for opt_type in ["CE", "PE"]:
#             for symbol, entries in symbol_list[opt_type]:
#                 strike = self.parse_strike_from_symbol(symbol)
#                 for pos in entries:
#                     qty = pos['qty']
#                     premium = abs(pos['premium']) / abs(qty) if qty != 0 else 0
#                     intrinsic = max(moved_spot - strike, 0) if opt_type == 'CE' else max(strike - moved_spot, 0)
#                     if qty < 0:
#                         loss = (intrinsic - premium) * abs(qty)
#                     else:
#                         loss = (premium - intrinsic) * abs(qty)
#                     self.total_loss_down += loss
        
#         if self.total_loss_down >= max_loss_allowed:
#             hedge_needed_pe = True

#         # Place CE hedge
#         if hedge_needed_ce:
#             loss_pct_up = self.total_loss_up / self.updated_margin[underlying]
#             self.hedge_pct_ce = self.var_level * self.max_hedge / loss_pct_up
#             hedge_strike_ce = int(np.ceil((spot * (1 + self.hedge_pct_ce)) / strike_diff) * strike_diff)
#             strikes_away_ce = min(abs(hedge_strike_ce - spot) // strike_diff, 149)
#             expiry = 1 if self.dte == 0 else 0
#             hedge_symbol_ce = self.find_symbol_by_moneyness_attempt(now, underlying, expiry, "CE", strikes_away_ce)
#             total_qty_ce = abs(net_qty_ce)
            
#             if hedge_symbol_ce is not None:
#                 success, _ = self.place_trade(now, 'BUY', total_qty_ce, hedge_symbol_ce, note=f"INIT HEDGE {underlying} CE")
#                 if success:
#                     self.hedge_positions[underlying]["CE"] = {"symbol": hedge_symbol_ce, "qty": total_qty_ce}
#                     print(f"{self.now} Initialized CE hedge: {hedge_symbol_ce} qty {total_qty_ce}")
#                 else:
#                     print(f"Failed to place CE hedge for {hedge_symbol_ce}")

#         # Place PE hedge
#         if hedge_needed_pe:
#             loss_pct_down = self.total_loss_down / self.updated_margin[underlying]
#             self.hedge_pct_pe = self.var_level * self.max_hedge / loss_pct_down
#             hedge_strike_pe = int(np.floor((spot * (1 - self.hedge_pct_pe)) / strike_diff) * strike_diff)
#             strikes_away_pe = min(abs(hedge_strike_pe - spot) // strike_diff, 149)
#             expiry = 1 if self.dte == 0 else 0
#             hedge_symbol_pe = self.find_symbol_by_moneyness_attempt(now, underlying, expiry, "PE", strikes_away_pe)
#             total_qty_pe = abs(net_qty_pe)
            
#             if hedge_symbol_pe is not None:
#                 success, _ = self.place_trade(now, 'BUY', total_qty_pe, hedge_symbol_pe, note=f"INIT HEDGE {underlying} PE")
#                 if success:
#                     self.hedge_positions[underlying]["PE"] = {"symbol": hedge_symbol_pe, "qty": total_qty_pe}
#                     print(f"{self.now} Initialized PE hedge: {hedge_symbol_pe} qty {total_qty_pe}")
#                 else:
#                     print(f"Failed to place PE hedge for {hedge_symbol_pe}")

#     def adjust_hedge_qty(self, now, underlying, symbol_list):
#         """Adjust existing hedge quantities to maintain VaR by solving for exact required qty"""
#         if underlying not in self.hedge_positions:
#             return

#         try:
#             spot = self.get_tick(now, f"{underlying}SPOT")['c']
#         except:
#             return

#         max_loss_allowed = self.var_level * self.updated_margin[underlying]
#         pct_move = self.max_hedge

#         # Helper function to calculate portfolio loss without hedge
#         def calculate_portfolio_loss(moved_spot):
#             loss = 0
#             for opt_type in ["CE", "PE"]:
#                 for symbol, entries in symbol_list[opt_type]:
#                     strike = self.parse_strike_from_symbol(symbol)
#                     for pos in entries:
#                         qty = pos['qty']
#                         premium = abs(pos['premium']) / abs(qty) if qty != 0 else 0
#                         intrinsic = max(moved_spot - strike, 0) if opt_type == 'CE' else max(strike - moved_spot, 0)
#                         if qty < 0:
#                             loss_val = (intrinsic - premium) * abs(qty)
#                         else:
#                             loss_val = (premium - intrinsic) * abs(qty)
#                         loss += loss_val
#             return loss

#         # Get hedge strikes and current quantities
#         hedge_strike_ce = None
#         hedge_strike_pe = None
#         current_qty_ce = 0
#         current_qty_pe = 0
        
#         if self.hedge_positions[underlying]["CE"] is not None:
#             hedge_symbol_ce = self.hedge_positions[underlying]["CE"]["symbol"]
#             hedge_strike_ce = self.parse_strike_from_symbol(hedge_symbol_ce)
#             current_qty_ce = self.hedge_positions[underlying]["CE"]["qty"]
        
#         if self.hedge_positions[underlying]["PE"] is not None:
#             hedge_symbol_pe = self.hedge_positions[underlying]["PE"]["symbol"]
#             hedge_strike_pe = self.parse_strike_from_symbol(hedge_symbol_pe)
#             current_qty_pe = self.hedge_positions[underlying]["PE"]["qty"]

#         # Upside scenario: adjust CE hedge qty
#         if self.hedge_positions[underlying]["CE"] is not None:
#             moved_spot = spot * (1 + pct_move)
#             portfolio_loss_up = calculate_portfolio_loss(moved_spot)
            
#             if portfolio_loss_up < max_loss_allowed:
#                 # Loss is within limit, can reduce hedge
#                 required_qty_ce = 0
#             else:
#                 # Loss exceeds limit, need to buy more hedge
#                 # Solve: portfolio_loss_up - hedge_qty_ce * max(moved_spot - hedge_strike_ce, 0) = max_loss_allowed
#                 current_price = self.get_tick(self.now, hedge_symbol_ce)['c']
#                 hedge_payoff_ce = max(moved_spot - hedge_strike_ce, 0) - current_price 
#                 if hedge_payoff_ce > 0:
#                     required_qty_ce = max(0, int(np.ceil((portfolio_loss_up - max_loss_allowed) / hedge_payoff_ce)))
#                 else:
#                     raise ValueError(f"Error 1: {symbol}")
            
#             if required_qty_ce != current_qty_ce:
#                 qty_diff = required_qty_ce - current_qty_ce
#                 symbol = self.hedge_positions[underlying]["CE"]["symbol"]
                
#                 if qty_diff > 0:
#                     success, _ = self.place_trade(now, 'BUY', qty_diff, symbol, note=f"ADD CE HEDGE {underlying}")
#                     if success:
#                         self.hedge_positions[underlying]["CE"]["qty"] = required_qty_ce
#                         print(f"{self.now} Added CE hedge qty: {qty_diff} for {symbol}")
#                 else:
#                     success, _ = self.place_trade(now, 'SELL', abs(qty_diff), symbol, note=f"REDUCE CE HEDGE {underlying}")
#                     if success:
#                         if required_qty_ce == 0:
#                             self.hedge_positions[underlying]["CE"] = None
#                         else:
#                             self.hedge_positions[underlying]["CE"]["qty"] = required_qty_ce
#                         print(f"{self.now} Reduced CE hedge qty: {abs(qty_diff)} for {symbol}")

#         # Downside scenario: adjust PE hedge qty
#         if self.hedge_positions[underlying]["PE"] is not None:
#             moved_spot = spot * (1 - pct_move)
#             portfolio_loss_down = calculate_portfolio_loss(moved_spot)
            
#             if portfolio_loss_down < max_loss_allowed:
#                 # Loss is within limit, can reduce hedge
#                 required_qty_pe = 0
#             else:
#                 # Loss exceeds limit, need to buy more hedge
#                 # Solve: portfolio_loss_down - hedge_qty_pe * max(hedge_strike_pe - moved_spot, 0) = max_loss_allowed
#                 current_price = self.get_tick(self.now, hedge_symbol_pe)['c']
#                 hedge_payoff_pe = max(hedge_strike_pe - moved_spot, 0) - current_price 
#                 if hedge_payoff_pe > 0:
#                     required_qty_pe = max(0, int(np.ceil((portfolio_loss_down - max_loss_allowed) / hedge_payoff_pe)))
#                 else:
#                     raise ValueError(f"Error 2 {symbol}")
            
#             print(portfolio_loss_down,max_loss_allowed,now,hedge_payoff_pe,hedge_strike_pe,moved_spot,current_price,required_qty_pe)
            

#             if required_qty_pe != current_qty_pe:
#                 qty_diff = required_qty_pe - current_qty_pe
#                 symbol = self.hedge_positions[underlying]["PE"]["symbol"]
                
#                 if qty_diff > 0:
#                     success, _ = self.place_trade(now, 'BUY', qty_diff, symbol, note=f"ADD PE HEDGE {underlying}")
#                     if success:
#                         self.hedge_positions[underlying]["PE"]["qty"] = required_qty_pe
#                         print(f"{self.now} Added PE hedge qty: {qty_diff} for {symbol}")
#                 else:
#                     success, _ = self.place_trade(now, 'SELL', abs(qty_diff), symbol, note=f"REDUCE PE HEDGE {underlying}")
#                     if success:
#                         if required_qty_pe == 0:
#                             self.hedge_positions[underlying]["PE"] = None
#                         else:
#                             self.hedge_positions[underlying]["PE"]["qty"] = required_qty_pe
#                         print(f"{self.now} Reduced PE hedge qty: {abs(qty_diff)} for {symbol}")


import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.definitions import *

if REDIS:
    from engine.ems import EventInterfacePositional, lot_size_dict, strike_diff_dict
else:
    from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

class HEDGEINTRADAYLTP(EventInterfacePositional):
    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()
        self.uid = ""
        self.var_level = 0.1
        self.margin = 1e7
        self.positions_df = None
        self.margin_df_nifty = None
        self.margin_df_sensex = None
        self.hedge_positions = {}  # {underlying: {"CE": {...}, "PE": {...}}}
        
        
    def set_params_from_uid(self, uid):
        self.uid = uid
        parts = uid.split('_', 4)
        assert parts[0] == self.strat_id
        self.var_level = float(parts[1])
        self.margin = float(parts[2])
        self.max_hedge = float(parts[3])
        self.basket_id = parts[4]

    def on_new_day(self):
        self.start_time = datetime.time(9, 21)
        self.end_time = datetime.time(15, 30)
        self.find_dte = True
        self.hedge_closed_today = False
        if self.positions_df is None:
            self.positions_df = self.load_positions_df()
        self.load_eq_curve()
    
    def load_positions_df(self):
        path = f'/home/mridul/jupiter/storage/portfolio/{self.basket_id}/combined_active_positions.csv'
        df = pd.read_csv(path)
        df['ts'] = pd.to_datetime(df['ts'])
        df['date'] = df['ts'].dt.date
        df = df.sort_values('ts')

        # # Keep rows where timestamp is the max for that date
        # df = df[df['ts'] == df.groupby('date')['ts'].transform('max')]

        # # Drop the helper 'date' column if you want
        # df = df.drop(columns=['date'])

        # # Change time to 15:20 for all entries
        # df['ts'] = df['ts'].dt.floor('d') + pd.Timedelta(hours=15, minutes=20)
        return df

    def load_eq_curve(self):
        path = f'/home/mridul/jupiter/storage/portfolio/{self.basket_id}/combined_tradebook.csv'
        df = pd.read_csv(path,usecols=['ts','value','symbol'], parse_dates=['ts'])
        underlyings = (
                            df['symbol']
                            .str.extract(r'^([A-Z]+)')[0]
                            .unique()
                            .tolist()
                        ) 
        self.margin_df = {}
        for underlying in underlyings:
            final = df[df['symbol'].str.startswith(underlying)]            
            final = final.groupby('ts').agg({'value': 'sum'}).reset_index()
            final["ts"] = pd.to_datetime(final["ts"])
            final = final.sort_values('ts')
            # Create a column with cumulative sum of value
            final['cumulative_value'] = final['value'].cumsum()
            self.margin_df[underlying] = final

    def get_symbol_params(self, symbol):
        if symbol.startswith("NIFTY"):
            return "NIFTY", "NIFTYSPOT", lot_size_dict["NIFTY"], strike_diff_dict["NIFTY"]
        if symbol.startswith("BANKNIFTY"):
            return "BANKNIFTY", "BANKNIFTYSPOT", lot_size_dict["BANKNIFTY"], strike_diff_dict["BANKNIFTY"]
        if symbol.startswith("FINNIFTY"):
            return "FINNIFTY", "FINNIFTYSPOT", lot_size_dict["FINNIFTY"], strike_diff_dict["FINNIFTY"]
        elif symbol.startswith("SENSEX"):
            return "SENSEX", "SENSEXSPOT", lot_size_dict["SENSEX"], strike_diff_dict["SENSEX"]
        else:
            raise ValueError(f"Unknown underlying in symbol: {symbol}")

    def on_bar_complete(self):
        now = self.now
        if self.positions_df is None or not (self.start_time <= now.time() <= self.end_time):
            return

        # Close all current hedges at start time and reset
        if self.now.time() >= self.start_time and not self.hedge_closed_today:
            self.close_all_hedges(now)
            self.hedge_closed_today = True
            return  # Don't process hedges on the same bar as close

        # Get exact match for this timestamp
        current_positions = self.positions_df[self.positions_df['ts'] == self.now]
        if current_positions.empty:
            return  # No action if no positions at this timestamp

        # Build grouped positions
        grouped = defaultdict(list)
        for _, row in current_positions.iterrows():
            symbol = row['symbol']
            qty = row['qty']
            current_price = self.get_tick(self.now,symbol)['c'] 
            premium = row['qty']*current_price
            grouped[symbol].append({'qty': qty, 'premium': premium})

        positions_by_underlying = defaultdict(lambda: defaultdict(list))

        for symbol in grouped:
            try:
                underlying, spot_sym, lot_size, strike_diff = self.get_symbol_params(symbol)
                opt_type = symbol[-2:]
                positions_by_underlying[underlying][opt_type].append((symbol, grouped[symbol]))
            except:
                continue

        # Calculating Margin i.e. Portfolio Value to be protected
        self.updated_margin = {}
        for underlying in positions_by_underlying:
            try:
                self.updated_margin[underlying] = self.margin + self.margin_df[underlying][self.margin_df[underlying]['ts'] <= self.now]['cumulative_value'].iloc[-1]
            except:
                print(f"No trades yet for {underlying}")               

        # For each underlying, initialize hedge if not exists, otherwise adjust qty
        for underlying in positions_by_underlying:
            if underlying not in self.hedge_positions:
                self.hedge_positions[underlying] = {"CE": None, "PE": None}
            
            # Check if CE hedge needs initialization or adjustment
            if self.hedge_positions[underlying]["CE"] is None:
                self.initialize_hedge_side(now, underlying, positions_by_underlying[underlying], "CE")
            else:
                self.adjust_hedge_qty_side(now, underlying, positions_by_underlying[underlying], "CE")
            
            # Check if PE hedge needs initialization or adjustment
            if self.hedge_positions[underlying]["PE"] is None:
                self.initialize_hedge_side(now, underlying, positions_by_underlying[underlying], "PE")
            else:
                self.adjust_hedge_qty_side(now, underlying, positions_by_underlying[underlying], "PE")

    def close_all_hedges(self, now):
        """Close all existing hedges and reset for fresh initialization"""
        for underlying in list(self.hedge_positions.keys()):
            for opt_type in ["CE", "PE"]:
                if opt_type in self.hedge_positions[underlying]:
                    hedge_data = self.hedge_positions[underlying][opt_type]
                    if hedge_data and hedge_data["qty"] > 0:
                        symbol = hedge_data["symbol"]
                        qty = hedge_data["qty"]
                        success, _ = self.place_trade(now, "SELL", qty, symbol, note="CLOSE HEDGE")
                        if success:
                            print(f"Closed hedge: {symbol} qty {qty}")
                            del self.hedge_positions[underlying][opt_type]
                        else:
                            print(f"Failed to close hedge for {symbol}")

        self.hedge_positions = {}

    def initialize_hedge_side(self, now, underlying, symbol_list, opt_type):
        """Initialize hedge for a specific side (CE or PE)"""
        try:
            spot = self.get_tick(now, f"{underlying}SPOT")['c']
        except:
            return

        strike_diff = strike_diff_dict[underlying]
        max_loss_allowed = self.var_level * self.updated_margin[underlying]
        print(max_loss_allowed,self.updated_margin[underlying])
        if self.find_dte:
            self.dte = self.get_dte_by_underlying(now, underlying)
            self.find_dte = False
        
        # Get net qty for this side
        net_qty = sum(pos['qty'] for symbol, entries in symbol_list[opt_type] for pos in entries)
        
        if net_qty >= 0:
            print("Net long or flat → no hedge needed", underlying) 
            return
        
        if opt_type == "CE":
            # Upside loss scenario
            moved_spot = spot * (1 + self.max_hedge)
            portfolio_loss = 0
            for opt in ["CE", "PE"]:
                for symbol, entries in symbol_list[opt]:
                    strike = self.parse_strike_from_symbol(symbol)
                    for pos in entries:
                        qty = pos['qty']
                        premium = abs(pos['premium']) / abs(qty) if qty != 0 else 0
                        intrinsic = max(moved_spot - strike, 0) if opt == 'CE' else max(strike - moved_spot, 0)
                        if qty < 0:
                            loss = (intrinsic - premium) * abs(qty)
                        else:
                            loss = (premium - intrinsic) * abs(qty)
                        portfolio_loss += loss
            
            if portfolio_loss >= max_loss_allowed:
                loss_pct = portfolio_loss / self.updated_margin[underlying]
                hedge_pct = self.var_level * self.max_hedge / loss_pct
                hedge_strike = int(np.floor((spot * (1 + hedge_pct)) / strike_diff) * strike_diff)
                strikes_away = min(abs(hedge_strike - spot) // strike_diff, 149)
                expiry = 1 if self.dte == 0 else 0
                hedge_symbol = self.find_symbol_by_moneyness_attempt(now, underlying, expiry, "CE", strikes_away)
                total_qty = abs(net_qty)
                print(self.now,portfolio_loss,max_loss_allowed,loss_pct,hedge_pct,hedge_strike,strikes_away,hedge_symbol,total_qty)
                if hedge_symbol is not None:
                    success, _ = self.place_trade(now, 'BUY', total_qty, hedge_symbol, note=f"INIT HEDGE {underlying} CE")
                    if success:
                        self.hedge_positions[underlying]["CE"] = {"symbol": hedge_symbol, "qty": total_qty}
                        print(f"{self.now} Initialized CE hedge: {hedge_symbol} qty {total_qty}")
                    else:
                        print(f"Failed to place CE hedge for {hedge_symbol}")
        
        elif opt_type == "PE":
            # Downside loss scenario
            moved_spot = spot * (1 - self.max_hedge)
            portfolio_loss = 0
            for opt in ["CE", "PE"]:
                for symbol, entries in symbol_list[opt]:
                    strike = self.parse_strike_from_symbol(symbol)
                    for pos in entries:
                        qty = pos['qty']
                        premium = abs(pos['premium']) / abs(qty) if qty != 0 else 0
                        intrinsic = max(moved_spot - strike, 0) if opt == 'CE' else max(strike - moved_spot, 0)
                        if qty < 0:
                            loss = (intrinsic - premium) * abs(qty)
                        else:
                            loss = (premium - intrinsic) * abs(qty)
                        portfolio_loss += loss
            
            if portfolio_loss >= max_loss_allowed:
                loss_pct = portfolio_loss / self.updated_margin[underlying]
                hedge_pct = self.var_level * self.max_hedge / loss_pct
                hedge_strike = int(np.ceil((spot * (1 - hedge_pct)) / strike_diff) * strike_diff)
                strikes_away = min(abs(hedge_strike - spot) // strike_diff, 149)
                expiry = 1 if self.dte == 0 else 0
                hedge_symbol = self.find_symbol_by_moneyness_attempt(now, underlying, expiry, "PE", strikes_away)
                total_qty = abs(net_qty)
                
                if hedge_symbol is not None:
                    success, _ = self.place_trade(now, 'BUY', total_qty, hedge_symbol, note=f"INIT HEDGE {underlying} PE")
                    if success:
                        self.hedge_positions[underlying]["PE"] = {"symbol": hedge_symbol, "qty": total_qty}
                        print(f"{self.now} Initialized PE hedge: {hedge_symbol} qty {total_qty}")
                    else:
                        print(f"Failed to place PE hedge for {hedge_symbol}")

    def adjust_hedge_qty_side(self, now, underlying, symbol_list, opt_type):
        """Adjust existing hedge quantity for a specific side (CE or PE) to maintain VaR"""
        if underlying not in self.hedge_positions or self.hedge_positions[underlying][opt_type] is None:
            raise ValueError("Hedge position does not exist for adjustment")
            return

        try:
            spot = self.get_tick(now, f"{underlying}SPOT")['c']
        except:
            return

        max_loss_allowed = self.var_level * self.updated_margin[underlying]
        pct_move = self.max_hedge
        print(max_loss_allowed,self.updated_margin[underlying])
        net_qty = sum(pos['qty'] for symbol, entries in symbol_list[opt_type] for pos in entries)
        

        # Helper function to calculate portfolio loss without hedge
        def calculate_portfolio_loss(moved_spot):
            loss = 0
            for opt in ["CE", "PE"]:
                for symbol, entries in symbol_list[opt]:
                    strike = self.parse_strike_from_symbol(symbol)
                    for pos in entries:
                        qty = pos['qty']
                        premium = abs(pos['premium']) / abs(qty) if qty != 0 else 0
                        intrinsic = max(moved_spot - strike, 0) if opt == 'CE' else max(strike - moved_spot, 0)
                        if qty < 0:
                            loss_val = (intrinsic - premium) * abs(qty)
                        else:
                            loss_val = (premium - intrinsic) * abs(qty)
                        loss += loss_val
            return loss

        hedge_symbol = self.hedge_positions[underlying][opt_type]["symbol"]
        hedge_strike = self.parse_strike_from_symbol(hedge_symbol)
        current_qty = self.hedge_positions[underlying][opt_type]["qty"]

        if opt_type == "CE" and net_qty < 0:
            moved_spot = spot * (1 + pct_move)
            portfolio_loss = calculate_portfolio_loss(moved_spot)
            
            if portfolio_loss < max_loss_allowed:
                # Loss is within limit, can reduce hedge
                required_qty = 0
            else:
                # Loss exceeds limit, need to buy more hedge
                # Payoff is intrinsic value at moved spot
                current_price = self.get_tick(self.now, hedge_symbol)['c']
                hedge_payoff = max(moved_spot - hedge_strike, 0) - current_price 
                if hedge_payoff > 0:
                    required_qty = max(0, int(np.floor((portfolio_loss - max_loss_allowed) / hedge_payoff)))
                else:
                    required_qty = current_qty
        
        elif opt_type == "PE" and net_qty < 0:
            moved_spot = spot * (1 - pct_move)
            portfolio_loss = calculate_portfolio_loss(moved_spot)
            
            if portfolio_loss < max_loss_allowed:
                # Loss is within limit, can reduce hedge
                required_qty = 0
            else:
                # Loss exceeds limit, need to buy more hedge
                # Payoff is intrinsic value at moved spot
                current_price = self.get_tick(self.now, hedge_symbol)['c']
                hedge_payoff = max(hedge_strike - moved_spot, 0) - current_price
                if hedge_payoff > 0:
                    required_qty = max(0, int(np.ceil((portfolio_loss - max_loss_allowed) / hedge_payoff)))
                else:
                    required_qty = current_qty
        else:
            required_qty = 0 # No hedge needed if net_qty >= 0
        
        if required_qty != current_qty:
            qty_diff = required_qty - current_qty   
            
            if qty_diff > 0:
                success, _ = self.place_trade(now, 'BUY', qty_diff, hedge_symbol, note=f"ADD {opt_type} HEDGE {underlying}")
                if success:
                    self.hedge_positions[underlying][opt_type]["qty"] = required_qty
                    print(f"{self.now} Added {opt_type} hedge qty: {qty_diff} for {hedge_symbol}")
            else:
                success, _ = self.place_trade(now, 'SELL', abs(qty_diff), hedge_symbol, note=f"REDUCE {opt_type} HEDGE {underlying}")
                if success:
                    if required_qty == 0:
                        self.hedge_positions[underlying][opt_type] = None
                    else:
                        self.hedge_positions[underlying][opt_type]["qty"] = required_qty
                    print(f"{self.now} Reduced {opt_type} hedge qty: {abs(qty_diff)} for {hedge_symbol}")