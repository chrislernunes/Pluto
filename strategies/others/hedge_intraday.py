import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.definitions import *

if REDIS:
    from engine.ems import EventInterfacePositional, lot_size_dict, strike_diff_dict
else:
    from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

class HEDGEINTRADAY(EventInterfacePositional):
    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()
        self.uid = ""
        self.var_level = 0.1
        self.margin = 1e7
        self.positions_df = None
        self.margin_df = None
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
        if self.margin_df is None:
            self.margin_df = self.load_eq_curve()
        self.updated_margin = self.margin + self.margin_df[self.margin_df['date'] <= self.now.date()]['cumulative_value'].iloc[-1]

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

    def load_eq_curve(self):
        path = f'/home/mridul/jupiter/storage/portfolio/{self.basket_id}/combined_tradebook.csv'
        df = pd.read_csv(path,usecols=['ts','value'], parse_dates=['ts'])
        df['date'] = df['ts'].dt.date
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
            return  # No action if no positions at this timestamp

        grouped = defaultdict(list)
        for _, row in current_positions.iterrows():
            symbol = row['symbol']
            qty = row['qty']
            premium = row['premium']
            grouped[symbol].append({'qty': qty, 'premium': premium})

        positions_by_underlying = defaultdict(lambda: defaultdict(list))

        for symbol in grouped:
            try:
                underlying, spot_sym, lot_size, strike_diff = self.get_symbol_params(symbol)
                opt_type = symbol[-2:]
                positions_by_underlying[underlying][opt_type].append((symbol, grouped[symbol]))
            except:
                continue

        for underlying in positions_by_underlying:
            self.process_hedge(now, underlying, positions_by_underlying[underlying])

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

    def process_hedge(self, now, underlying, symbol_list):
        try:
            spot = self.get_tick(now, f"{underlying}SPOT")['c']
        except:
            return

        lot_size = lot_size_dict[underlying]
        strike_diff = strike_diff_dict[underlying]
        max_loss_allowed = self.var_level * self.updated_margin
        
        if self.find_dte:
            self.dte = self.get_dte_by_underlying(now, underlying)
            self.find_dte = False
        
        # Check if hedge is even needed
        net_qty_ce = sum(pos['qty'] for symbol, entries in symbol_list["CE"] for pos in entries)
        net_qty_pe = sum(pos['qty'] for symbol, entries in symbol_list["PE"] for pos in entries)
        # print(net_qty_ce, net_qty_pe, now)
        if net_qty_ce >= 0 and net_qty_pe >= 0:
            print("Net long or flat → no hedge needed", underlying) 
            return

        # Simulate worst-case loss
        hedge_found_ce = False
        hedge_found_pe = False
        pct_move = self.max_hedge
        if net_qty_ce < 0:
            moved_spot = spot * (1 + pct_move)
            self.total_loss_up = 0
            for opt_type in ["CE","PE"]:
                for symbol, entries in symbol_list[opt_type]:
                    strike = self.parse_strike_from_symbol(symbol)
                    for pos in entries:
                        qty = pos['qty']
                        premium = abs(pos['premium']) / abs(qty)
                        intrinsic = max(moved_spot - strike, 0) if opt_type == 'CE' else max(strike - moved_spot, 0)
                        if qty < 0:
                            loss = (intrinsic - premium) * abs(qty)
                        else:
                            loss = (premium - intrinsic) * abs(qty)
                        self.total_loss_up += loss
            if self.total_loss_up >= max_loss_allowed:
                hedge_found_ce = True
                
        if net_qty_pe < 0:
            moved_spot = spot * (1 - pct_move)
            self.total_loss_down = 0
            for opt_type in ["CE","PE"]:
                for symbol, entries in symbol_list[opt_type]:
                    strike = self.parse_strike_from_symbol(symbol)
                    for pos in entries:
                        qty = pos['qty']
                        premium = abs(pos['premium']) / abs(qty)
                        intrinsic = max(moved_spot - strike, 0) if opt_type == 'CE' else max(strike - moved_spot, 0)
                        if qty < 0:
                            loss = (intrinsic - premium) * abs(qty)
                        else:
                            loss = (premium - intrinsic) * abs(qty)
                        self.total_loss_down += loss
            if self.total_loss_down >= max_loss_allowed:
                hedge_found_pe = True
                
        # Choose strike for hedge
        if hedge_found_ce:
            loss_pct_up = self.total_loss_up/self.updated_margin
            self.hedge_pct_ce = self.var_level*self.max_hedge/loss_pct_up
            hedge_strike_ce = int(np.ceil((spot * (1 + self.hedge_pct_ce)) / strike_diff) * strike_diff)
            strikes_away_ce = min(abs(hedge_strike_ce - spot) // strike_diff, 149)
            if self.dte == 0:
                expiry = 1
            else:
                expiry = 0
            hedge_symbol_ce = self.find_symbol_by_moneyness_attempt(now, underlying, expiry, "CE", strikes_away_ce)
            total_qty_ce = abs(net_qty_ce)
            log_file = "/home/mridul/jupiter/simulation_logssss.txt"
            # Log completion
            with open(log_file, 'a') as f:
                f.write(f"{self.now}, {hedge_symbol_ce}, {spot}, {hedge_strike_ce}, {self.updated_margin},{self.total_loss_down},{self.total_loss_up},{max_loss_allowed}\n")
            if hedge_symbol_ce is not None:
                success, _ = self.place_trade(now, 'BUY', total_qty_ce, hedge_symbol_ce, note=f"ADD HEDGE {underlying} CE")
                if success:
                    self.hedge_positions.append({"symbol": hedge_symbol_ce, "qty": total_qty_ce})
                else:
                    print(f"COULD NOT PLACE HEDGE TRADE FOR {hedge_symbol_ce}")
                
        if hedge_found_pe:
            loss_pct_down = self.total_loss_down/self.updated_margin
            self.hedge_pct_pe = self.var_level*self.max_hedge/loss_pct_down
            hedge_strike_pe = int(np.floor((spot * (1 - self.hedge_pct_pe)) / strike_diff) * strike_diff)
            strikes_away_pe = min(abs(hedge_strike_pe - spot) // strike_diff, 149)
            if self.dte == 0:
                expiry = 1
            else:
                expiry = 0
            print(loss_pct_down,self.hedge_pct_pe,hedge_strike_pe,strikes_away_pe)
            hedge_symbol_pe = self.find_symbol_by_moneyness_attempt(now, underlying, expiry, "PE", strikes_away_pe)
            total_qty_pe = abs(net_qty_pe)
            log_file = "/home/mridul/jupiter/simulation_logssss.txt"
            # Log completion
            with open(log_file, 'a') as f:
                f.write(f"{self.now}, {hedge_symbol_pe}, {spot}, {hedge_strike_pe}, {self.updated_margin},{self.total_loss_down},{self.total_loss_up},{max_loss_allowed} \n")
            if hedge_symbol_pe is not None:
                success, _ = self.place_trade(now, 'BUY', total_qty_pe, hedge_symbol_pe, note=f"ADD HEDGE {underlying} PE")
                if success:
                    self.hedge_positions.append({"symbol": hedge_symbol_pe, "qty": total_qty_pe})
                else:            
                    print(f"COULD NOT PLACE HEDGE TRADE FOR {hedge_symbol_pe}")
    
    def roll_over_hedges_if_expiry(self):
        if self.now.time() != datetime.time(15, 20):
            return

        for hedge_symbol, qty in list(self.hedge_positions.items()):
            # Get DTE and check if today is expiry
            dte = self.get_dte(self.now, hedge_symbol)
            if dte > 0:
                continue  # Not expiring today

            # Parse option type and underlying
            option_type = hedge_symbol[-2:]
            strike = self.parse_strike_from_symbol(hedge_symbol)
            underlying = None
            if hedge_symbol.startswith("NIFTY"):
                underlying = "NIFTY"
            elif hedge_symbol.startswith("SENSEX"):
                underlying = "SENSEX"
            else:
                continue  # Unknown underlying

            # Sell existing hedge
            self.place_trade(self.now, 'SELL', qty, hedge_symbol, note="ROLLOVER EXIT")

            # Buy same strike on next expiry
            next_expiry_symbol = self.find_symbol_by_moneyness(self.now, underlying, expiry=1, option_type=option_type,
                                                            moneyness=abs(strike - self.get_tick(self.now, f"{underlying}SPOT")['c']) // strike_diff_dict[underlying])
            if next_expiry_symbol:
                success, _ = self.place_trade(self.now, 'BUY', qty, next_expiry_symbol, note="ROLLOVER ENTRY")
                if success:
                    self.hedge_positions[next_expiry_symbol] = qty

            # Clean up old hedge entry
            del self.hedge_positions[hedge_symbol]



#############################################################
#############################################################
#############################################################


# import datetime
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# from utils.definitions import *

# if REDIS:
#     from engine.ems import EventInterfacePositional, lot_size_dict, strike_diff_dict
# else:
#     from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

# class HEDGEINTRADAY(EventInterfacePositional):
#     def __init__(self, conn=None):
#         super().__init__(conn)
#         self.strat_id = self.__class__.__name__.lower()
#         self.uid = ""
#         self.var_level = 0.1
#         self.margin = 1e7
#         self.positions_df = None
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
#         self.positions_df = self.load_positions_df()

#     def load_positions_df(self):
#         path = f'/home/mridul/jupiter/storage/portfolio/{self.basket_id}/combined_active_positions.csv'
#         df = pd.read_csv(path, converters={"positions": eval})
#         df['ts'] = pd.to_datetime(df['ts'])
#         df['date'] = df['ts'].dt.date
#         df = df.sort_values('ts')
#         df = df.groupby('date').tail(1)
#         positions_list = []
#         for _, row in df.iterrows():
#             ts = row['ts']
#             for p in row['positions']:
#                 symbol = p['symbol']
#                 qty = p['net_qty']
#                 premium = p['premium']
#                 if qty != 0:
#                     positions_list.append({'ts': ts, 'symbol': symbol, 'qty': qty, 'premium': premium})
#         return pd.DataFrame(positions_list)

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

#     def get_strike(self, symbol):
#         return int(symbol[-7:-2])

#     def on_bar_complete(self):
#         now = self.now
#         if self.positions_df is None or not (self.start_time <= now.time() <= self.end_time):
#             return

#         # Get exact match for this timestamp
#         current_positions = self.positions_df[self.positions_df['ts'] == self.now]
#         if current_positions.empty:
#             return  # No action if no positions at this timestamp

#         grouped = defaultdict(list)
#         for _, row in current_positions.iterrows():
#             symbol = row['symbol']
#             qty = row['qty']
#             premium = row['premium']
#             grouped[symbol].append({'qty': qty, 'premium': premium})

#         positions_by_underlying = defaultdict(lambda: defaultdict(list))

#         for symbol in grouped:
#             try:
#                 underlying, spot_sym, lot_size, strike_diff = self.get_symbol_params(symbol)
#                 opt_type = symbol[-2:]
#                 positions_by_underlying[underlying][opt_type].append((symbol, grouped[symbol]))
#             except:
#                 continue

#         # Close all current hedges before evaluating new hedge
#         self.close_all_hedges(now)

#         for underlying in positions_by_underlying:
#             for opt_type in positions_by_underlying[underlying]:
#                 self.process_hedge(now, underlying, opt_type, positions_by_underlying[underlying][opt_type])
        
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

#     def process_hedge(self, now, underlying, option_type, symbol_list):
#         try:
#             spot = self.get_tick(now, f"{underlying}SPOT")['c']
#         except:
#             return

#         lot_size = lot_size_dict[underlying]
#         strike_diff = strike_diff_dict[underlying]
#         max_loss_allowed = self.var_level * self.margin

#         # Check if hedge is even needed
#         net_qty = sum(pos['qty'] for symbol, entries in symbol_list for pos in entries)
#         if net_qty >= 0:
#             print("Net long or flat → no hedge needed")
#             return

#         # Simulate worst-case loss
#         hedge_found = False
#         for pct_move in np.linspace(0, self.max_hedge, 10):
#             moved_spot = spot * (1 + pct_move) if option_type == 'CE' else spot * (1 - pct_move)
#             total_loss = 0
#             for symbol, entries in symbol_list:
#                 strike = self.parse_strike_from_symbol(symbol)
#                 for pos in entries:
#                     qty = pos['qty']
#                     premium = abs(pos['premium']) / abs(qty)
#                     intrinsic = max(moved_spot - strike, 0) if option_type == 'CE' else max(strike - moved_spot, 0)
#                     if qty < 0:
#                         loss = (intrinsic - premium) * abs(qty)
#                     else:
#                         loss = (premium - intrinsic) * abs(qty)
#                     total_loss += loss
#             if total_loss >= max_loss_allowed:
#                 hedge_found = True
#                 break

#         # Choose strike for hedge
#         if hedge_found:
#             hedge_strike = int(np.ceil((spot * (1 + pct_move)) / strike_diff) * strike_diff) if option_type == 'CE' \
#                         else int(np.floor((spot * (1 - pct_move)) / strike_diff) * strike_diff)
#         else:
#             hedge_strike = int(np.ceil((spot * (1 + self.max_hedge)) / strike_diff) * strike_diff) if option_type == 'CE' \
#                         else int(np.floor((spot * (1 - self.max_hedge)) / strike_diff) * strike_diff)

#         strikes_away = min(abs(hedge_strike - spot) // strike_diff, 149)
#         expiry = 0
#         print(strikes_away,self.now,hedge_found,option_type)
#         hedge_symbol = self.find_symbol_by_moneyness(now, underlying, expiry, option_type, strikes_away)
#         if hedge_symbol is None:
#             return

#         total_qty = abs(sum(pos['qty'] for symbol, entries in symbol_list for pos in entries))
#         if total_qty == 0:
#             return

#         # Place new hedge
#         success, _ = self.place_trade(now, 'BUY', total_qty, hedge_symbol, note=f"ADD HEDGE {underlying} {option_type}")
#         if success:
#             self.hedge_positions.append({"symbol": hedge_symbol, "qty": total_qty})
#         else:
#             print(f"COULD NOT PLACE HEDGE TRADE FOR {hedge_symbol}")
        
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