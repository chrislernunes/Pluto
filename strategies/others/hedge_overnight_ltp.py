# ###################################
# #OVERNIGHT HEDGE WITH NIFTY AND SENSEX
# ####################################

# import datetime
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# from utils.definitions import *

# if REDIS:
#     from engine.ems import EventInterfacePositional, lot_size_dict, strike_diff_dict
# else:
#     from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

# class HEDGEOVERNIGHTLTP(EventInterfacePositional):
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

#         # Keep rows where timestamp is the max for that date
#         df = df[df['ts'] == df.groupby('date')['ts'].transform('max')]

#         # Drop the helper 'date' column if you want
#         df = df.drop(columns=['date'])

#         # Change time to 15:20 for all entries
#         df['ts'] = df['ts'].dt.floor('d') + pd.Timedelta(hours=15, minutes=20)
#         return df

#     def load_eq_curve(self,underlying):
#         path = f'/home/mridul/jupiter/storage/portfolio/{self.basket_id}/combined_tradebook.csv'
#         df = pd.read_csv(path,usecols=['ts','value','symbol'], parse_dates=['ts'])
            
#         df['date'] = df['ts'].dt.date
#         df = df[df['symbol'].str.startswith(underlying)]
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
        
#         self.dte = self.get_dte_by_underlying(now, underlying)
        
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
#         self.max_hedge = self.max_hedge
#         if True:
#             moved_spot = spot * (1 + self.max_hedge)
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
#             moved_spot = spot * (1 - self.max_hedge)
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
    
#     def roll_over_hedges_if_expiry(self):
#         if self.now.time() != datetime.time(15, 20):
#             return

#         for hedge_symbol, qty in list(self.hedge_positions.items()):
#             # Get DTE and check if today is expiry
#             dte = self.get_dte(self.now, hedge_symbol)
#             if dte > 0:
#                 continue  # Not expiring today

#             # Parse option type and underlying
#             option_type = hedge_symbol[-2:]
#             strike = self.parse_strike_from_symbol(hedge_symbol)
#             underlying = None
#             if hedge_symbol.startswith("NIFTY"):
#                 underlying = "NIFTY"
#             elif hedge_symbol.startswith("SENSEX"):
#                 underlying = "SENSEX"
#             else:
#                 continue  # Unknown underlying

#             # Sell existing hedge
#             self.place_trade(self.now, 'SELL', qty, hedge_symbol, note="ROLLOVER EXIT")

#             # Buy same strike on next expiry
#             next_expiry_symbol = self.find_symbol_by_moneyness(self.now, underlying, expiry=1, option_type=option_type,
#                                                             moneyness=abs(strike - self.get_tick(self.now, f"{underlying}SPOT")['c']) // strike_diff_dict[underlying])
#             if next_expiry_symbol:
#                 success, _ = self.place_trade(self.now, 'BUY', qty, next_expiry_symbol, note="ROLLOVER ENTRY")
#                 if success:
#                     self.hedge_positions[next_expiry_symbol] = qty

#             # Clean up old hedge entry
#             del self.hedge_positions[hedge_symbol]


###################################
## HEDGE OVERNIGHT WITH NIFTY ONLY
####################################

import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.definitions import *

if REDIS:
    from engine.ems import EventInterfacePositional, lot_size_dict, strike_diff_dict
else:
    from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

class HEDGEOVERNIGHTLTP(EventInterfacePositional):
    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()
        self.uid = ""
        self.var_level = 0.1
        self.margin = 1e7
        self.positions_df = None
        self.margin_df_nifty = None
        self.margin_df_sensex = None
        self.hedge_positions = []

    def set_params_from_uid(self, uid):
        self.uid = uid
        parts = uid.split('_', 4)
        assert parts[0] == self.strat_id
        self.var_level = float(parts[1])
        self.margin = float(parts[2])
        self.max_hedge = float(parts[3])
        self.basket_id = parts[4]

    def on_new_day(self):
        self.start_time = datetime.time(9, 15)
        self.end_time = datetime.time(15, 30)
        self.find_dte = True
        if self.positions_df is None:
            self.positions_df = self.load_positions_df()
        if self.margin_df_nifty is None:
            self.margin_df_nifty = self.load_eq_curve("NIFTY")
        if self.margin_df_sensex is None:    
            self.margin_df_sensex = self.load_eq_curve("SENSEX")
        self.updated_margin = {}
        try:
            self.updated_margin["NIFTY"] = self.margin + self.margin_df_nifty[self.margin_df_nifty['date'] <= self.now.date()]['cumulative_value'].iloc[-1]
        except:
            print("No trades yet for NIFTY")
        try:
            self.updated_margin["SENSEX"] = self.margin + self.margin_df_sensex[self.margin_df_sensex['date'] <= self.now.date()]['cumulative_value'].iloc[-1]
        except:
            print("No trades yet for SENSEX")
            
    def load_positions_df(self):
        path = f'/home/mridul/jupiter/storage/portfolio/{self.basket_id}/combined_active_positions.csv'
        df = pd.read_csv(path)
        df['ts'] = pd.to_datetime(df['ts'])
        df['date'] = df['ts'].dt.date
        df = df.sort_values('ts')

        # Keep rows where timestamp is the max for that date
        df = df[df['ts'] == df.groupby('date')['ts'].transform('max')]

        # Drop the helper 'date' column if you want
        df = df.drop(columns=['date'])

        # Change time to 15:20 for all entries
        df['ts'] = df['ts'].dt.floor('d') + pd.Timedelta(hours=15, minutes=20)
        return df

    def load_eq_curve(self,underlying):
        path = f'/home/mridul/jupiter/storage/portfolio/{self.basket_id}/combined_tradebook.csv'
        df = pd.read_csv(path,usecols=['ts','value','symbol'], parse_dates=['ts'])
            
        df['date'] = df['ts'].dt.date
        df = df[df['symbol'].str.startswith(underlying)]
        # Group by date and aggregate value by sum
        final = df.groupby('date').agg({'value': 'sum'}).reset_index()
        final["date"] = pd.to_datetime(final["date"]).dt.date
        final = final.sort_values('date')
        # Create a column with cumulative sum of value
        final['cumulative_value'] = final['value'].cumsum()
        return final

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

        # Close all current hedges before evaluating new hedge
        if self.now.time() == self.start_time:
            self.close_all_hedges(now)

        # Get exact match for this timestamp
        current_positions = self.positions_df[self.positions_df['ts'] == self.now]
        if current_positions.empty:
            # print(self.now, "Current Positions are empty")
            return  # No action if no positions at this timestamp

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

        # Calculate max loss for each underlying
        underlying_loss_data = {}
        for underlying in positions_by_underlying:
            loss_data = self.calculate_max_loss(now, underlying, positions_by_underlying[underlying])
            if loss_data:
                underlying_loss_data[underlying] = loss_data

        # Now place hedges based on aggregated losses
        if underlying_loss_data:
            self.place_aggregated_hedge(now, underlying_loss_data)

        # Store current net positions for reference and manage hedge rollover
        # self.roll_over_hedges_if_expiry()

    def close_all_hedges(self, now):
        new_hedge_positions = []
        for hedge in self.hedge_positions:
            symbol = hedge["symbol"]
            qty = hedge["qty"]
            if qty > 0:
                success, _ = self.place_trade(now, "SELL", qty, symbol, note="CLOSE HEDGE")
                if not success:
                    print(f"Failed to close hedge for {symbol}, keeping in list")
                    new_hedge_positions.append(hedge)  # retain if not closed
        self.hedge_positions = new_hedge_positions

    def calculate_max_loss(self, now, underlying, symbol_list):
        """Calculate max loss for a single underlying without placing trades"""
        try:
            spot = self.get_tick(now, f"{underlying}SPOT")['c']
        except:
            return None

        lot_size = lot_size_dict[underlying]
        strike_diff = strike_diff_dict[underlying]
        max_loss_allowed = self.var_level * self.updated_margin[underlying]
        
        
        # Check if hedge is even needed
        net_qty_ce = sum(pos['qty'] for symbol, entries in symbol_list["CE"] for pos in entries)
        net_qty_pe = sum(pos['qty'] for symbol, entries in symbol_list["PE"] for pos in entries)

        if net_qty_ce >= 0 and net_qty_pe >= 0:
            print(f"Net long or flat → no hedge needed for {underlying}") 
            return None

        # Simulate worst-case loss for upward move
        moved_spot_up = spot * (1 + self.max_hedge)
        total_loss_up = 0
        
        for opt_type in ["CE", "PE"]:
            for symbol, entries in symbol_list[opt_type]:
                strike = self.parse_strike_from_symbol(symbol)
                for pos in entries:
                    qty = pos['qty']
                    premium = abs(pos['premium']) / abs(qty)
                    intrinsic = max(moved_spot_up - strike, 0) if opt_type == 'CE' else max(strike - moved_spot_up, 0)
                    if qty < 0:
                        loss = (intrinsic - premium) * abs(qty)
                    else:
                        loss = (premium - intrinsic) * abs(qty)
                    print(f"Upside - {symbol}, Qty: {qty}, Premium: {premium}, Intrinsic: {intrinsic}, Loss: {loss}")
                    total_loss_up += loss

        # Simulate worst-case loss for downward move
        moved_spot_down = spot * (1 - self.max_hedge)
        total_loss_down = 0
        
        for opt_type in ["CE", "PE"]:
            for symbol, entries in symbol_list[opt_type]:
                strike = self.parse_strike_from_symbol(symbol)
                for pos in entries:
                    qty = pos['qty']
                    premium = abs(pos['premium']) / abs(qty)
                    intrinsic = max(moved_spot_down - strike, 0) if opt_type == 'CE' else max(strike - moved_spot_down, 0)
                    if qty < 0:
                        loss = (intrinsic - premium) * abs(qty)
                    else:
                        loss = (premium - intrinsic) * abs(qty)
                    print(f"Downside - {symbol}, Qty: {qty}, Premium: {premium}, Intrinsic: {intrinsic}, Loss: {loss}")
                    total_loss_down += loss

        return {
            'underlying': underlying,
            'spot': spot,
            'net_qty_ce': net_qty_ce,
            'net_qty_pe': net_qty_pe,
            'total_loss_up': total_loss_up,
            'total_loss_down': total_loss_down,
            'max_loss_allowed': max_loss_allowed,
            'strike_diff': strike_diff,
            'lot_size': lot_size
        }

    def place_aggregated_hedge(self, now, underlying_loss_data):
        """Place hedge trades on NIFTY based on aggregated losses from all underlyings"""
        
        # Sum up all losses across underlyings
        total_loss_up_all = sum(data['total_loss_up'] for data in underlying_loss_data.values())
        total_loss_down_all = sum(data['total_loss_down'] for data in underlying_loss_data.values())
        total_max_loss_allowed = sum(data['max_loss_allowed'] for data in underlying_loss_data.values())
        total_margin = sum(self.updated_margin[data['underlying']] for data in underlying_loss_data.values())

        # Get NIFTY spot price
        try:
            nifty_spot = self.get_tick(now, "NIFTYSPOT")['c']
        except:
            print("Could not get NIFTY spot price")
            return
        
        nifty_strike_diff = strike_diff_dict['NIFTY']
        nifty_lot_size = lot_size_dict['NIFTY']
        
        # Determine DTE for NIFTY
        nifty_dte = self.get_dte_by_underlying(now, 'NIFTY')
        expiry = 1 if nifty_dte == 0 else 0
        
        hedge_found_ce = total_loss_up_all >= total_max_loss_allowed
        hedge_found_pe = total_loss_down_all >= total_max_loss_allowed
        
        print(f"{self.now} Aggregated losses - Up: {total_loss_up_all:.2f}, Down: {total_loss_down_all:.2f}, Max allowed: {total_max_loss_allowed:.2f}, Total margin: {total_margin:.2f}")
        
        # Place CE hedge if needed
        if hedge_found_ce:
            loss_pct_up = total_loss_up_all / total_margin
            hedge_pct_ce = self.var_level * self.max_hedge / loss_pct_up
            hedge_strike_ce = int(np.ceil((nifty_spot * (1 + hedge_pct_ce)) / nifty_strike_diff) * nifty_strike_diff)
            strikes_away_ce = min(abs(hedge_strike_ce - nifty_spot) // nifty_strike_diff, 149)
            
            hedge_symbol_ce = self.find_symbol_by_moneyness_attempt(now, 'NIFTY', expiry, "CE", strikes_away_ce)
            
            # Calculate total quantity needed (sum of all CE net positions)
            total_qty_ce = sum(abs(data['net_qty_ce']) for data in underlying_loss_data.values())
            
            if hedge_symbol_ce is not None:
                success, _ = self.place_trade(now, 'BUY', total_qty_ce, hedge_symbol_ce, note=f"ADD AGGREGATED HEDGE NIFTY CE")
                if success:
                    self.hedge_positions.append({"symbol": hedge_symbol_ce, "qty": total_qty_ce})
                    print(f"Placed aggregated CE hedge: {hedge_symbol_ce}, qty: {total_qty_ce}")
                else:
                    print(f"COULD NOT PLACE HEDGE TRADE FOR {hedge_symbol_ce}")
        
        # Place PE hedge if needed
        if hedge_found_pe:
            loss_pct_down = total_loss_down_all / total_margin
            hedge_pct_pe = self.var_level * self.max_hedge / loss_pct_down
            hedge_strike_pe = int(np.floor((nifty_spot * (1 - hedge_pct_pe)) / nifty_strike_diff) * nifty_strike_diff)
            strikes_away_pe = min(abs(hedge_strike_pe - nifty_spot) // nifty_strike_diff, 149)
            
            print(f"PE hedge params - loss_pct: {loss_pct_down}, hedge_pct: {hedge_pct_pe}, strike: {hedge_strike_pe}, strikes_away: {strikes_away_pe}")
            
            hedge_symbol_pe = self.find_symbol_by_moneyness_attempt(now, 'NIFTY', expiry, "PE", strikes_away_pe)
            
            # Calculate total quantity needed (sum of all PE net positions)
            total_qty_pe = sum(abs(data['net_qty_pe']) for data in underlying_loss_data.values())
            
            if hedge_symbol_pe is not None:
                success, _ = self.place_trade(now, 'BUY', total_qty_pe, hedge_symbol_pe, note=f"ADD AGGREGATED HEDGE NIFTY PE")
                if success:
                    self.hedge_positions.append({"symbol": hedge_symbol_pe, "qty": total_qty_pe})
                    print(f"Placed aggregated PE hedge: {hedge_symbol_pe}, qty: {total_qty_pe}")
                else:            
                    print(f"COULD NOT PLACE HEDGE TRADE FOR {hedge_symbol_pe}")