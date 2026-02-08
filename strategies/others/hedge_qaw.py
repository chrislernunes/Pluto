###########################################################
# APPROCH 1: Using 5% Fixed OTM Hedge with QAW Equity Curve
###########################################################



# import datetime
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# from utils.definitions import *

# from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

# class HEDGEQAW(EventInterfacePositional):
#     def __init__(self, conn=None):
#         super().__init__(conn)
#         self.strat_id = self.__class__.__name__.lower()
#         self.uid = ""
#         self.var_level = 0.1
#         self.margin = 1e7
#         self.position_pe = 0
#         self.margin_df = None
#         self.underlying = "NIFTY"

        
        
#     def set_params_from_uid(self, uid):
#         self.uid = uid
#         parts = uid.split('_', 4)
#         assert parts[0] == self.strat_id
#         self.var_level = float(parts[1])
#         self.margin = float(parts[2])
#         self.max_hedge = float(parts[3])
#         self.otm_perc = float(parts[4])

#     def on_new_day(self):
#         self.start_time = datetime.time(9, 21)
#         self.end_time = datetime.time(14, 30)
#         self.find_dte = True
#         self.hedge_closed_today = False
#         # if self.positions_df is None:
#             # self.positions_df = self.load_positions_df()
#         if self.margin_df is None:  
#             self.margin_df = self.load_eq_curve()
#         self.dte = self.get_dte_by_underlying(self.now, self.underlying)
#         today = pd.Timestamp(self.now).normalize()
#         self.curr_margin = float(self.margin_df[self.margin_df['date'] == today]['qaw'].iloc[0])
#         self.var = self.var_level * self.curr_margin
#         print(self.now.date(), "DTE:", self.dte, "Curr Margin:", self.curr_margin, "Var:", self.var)
        
    
#     def load_eq_curve(self):
#         path = f'/home/mridul/jupiter/storage/qaw_nav.csv'
#         df = pd.read_csv(path,usecols=['date','qaw'], parse_dates=['date'],dayfirst=True)
#         df['qaw'] = pd.to_numeric(df['qaw'], errors='coerce')
#         # df['qaw'] = df['qaw'] * float(self.margin)
#         return df

#     def on_bar_complete(self):
#         now = self.now

#         # Exit on Expiry Day
#         if self.position_pe == 1 and now.time() == self.end_time and self.dte == 0:
#             self.success_pe, _ = self.place_trade(self.now, 'SELL', self.position_qty, self.symbol_pe,note="EXIT HEDGE")
#             if self.success_pe:
#                 self.symbol_pe = None
#                 self.position_pe = 0
#                 self.position_qty = 0

#         # Entry on Expiry Day
#         if self.position_pe == 0 and now.time() > self.start_time:
#             if self.dte == 0:
#                 expiry_idx  = 1
#             else:
#                 expiry_idx = 0
#             self.symbol_pe, _  = self.find_symbol_by_itm_percent_v2(self.now, self.underlying, expiry_idx, 'PE', -self.otm_perc)
#             if self.symbol_pe is None:
#                 return
#             self.strike_pe = self.parse_strike_from_symbol(self.symbol_pe)
#             self.spot = self.get_tick(self.now, f"{self.underlying}SPOT")['c']
#             self.moved_spot = self.spot * (1 - self.max_hedge)
#             current_price = self.get_tick(self.now, self.symbol_pe)['c']
#             hedge_payoff = max(self.strike_pe - self.moved_spot, 0) - current_price 
#             print("Hedge Symbol PE:", self.symbol_pe, self.spot, self.moved_spot, self.strike_pe, current_price, hedge_payoff)            
#             if hedge_payoff <=0:
#                 raise ValueError("Hedge payoff is non-positive, cannot enter hedge")    
#             self.position_qty = int(self.var / hedge_payoff)

#             if self.symbol_pe is not None:
#                 self.success_pe, _ = self.place_trade(self.now, 'BUY', self.position_qty, self.symbol_pe,note="ENTRY HEDGE")
#             if self.success_pe:
#                 self.position_pe = 1



#############################################################################
# APPROCH 2: Using Fixed Qantity and Finding Strike using Payoff Requirement
#############################################################################


# import datetime
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# from utils.definitions import *

# from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

# class HEDGEQAW(EventInterfacePositional):
#     def __init__(self, conn=None):
#         super().__init__(conn)
#         self.strat_id = self.__class__.__name__.lower()
#         self.uid = ""
#         self.var_level = 0.1
#         self.margin = 1e7
#         self.position_pe = 0
#         self.margin_df = None
#         self.underlying = "NIFTY"

        
        
#     def set_params_from_uid(self, uid):
#         self.uid = uid
#         parts = uid.split('_', 4)
#         assert parts[0] == self.strat_id
#         self.var_level = float(parts[1])
#         self.margin = float(parts[2])
#         self.max_hedge = float(parts[3])
#         self.otm_perc = float(parts[4])

#     def on_new_day(self):
#         self.start_time = datetime.time(9, 21)
#         self.end_time = datetime.time(14, 30)
#         self.find_dte = True
#         self.hedge_closed_today = False
#         # if self.positions_df is None:
#             # self.positions_df = self.load_positions_df()
#         if self.margin_df is None:  
#             self.margin_df = self.load_eq_curve()
#         self.dte = self.get_dte_by_underlying(self.now, self.underlying)
#         today = pd.Timestamp(self.now).normalize()
#         self.curr_margin = float(self.margin_df[self.margin_df['date'] == today]['qaw'].iloc[0])
#         self.var = (self.max_hedge-self.var_level) * self.curr_margin
#         print(self.now.date(), "DTE:", self.dte, "Curr Margin:", self.curr_margin, "Var:", self.var)
        
    
#     def load_eq_curve(self):
#         path = f'/home/mridul/jupiter/storage/qaw_nav.csv'
#         df = pd.read_csv(path,usecols=['date','qaw'], parse_dates=['date'],dayfirst=True)
#         df['qaw'] = pd.to_numeric(df['qaw'], errors='coerce')
#         # df['qaw'] = df['qaw'] * float(self.margin)
#         return df

#     def on_bar_complete(self):
#         now = self.now

#         # Exit on Expiry Day
#         if self.position_pe == 1 and now.time() == self.end_time and self.dte == 0:
#             self.success_pe, _ = self.place_trade(self.now, 'SELL', self.position_qty, self.symbol_pe,note="EXIT HEDGE")
#             if self.success_pe:
#                 self.symbol_pe = None
#                 self.position_pe = 0
#                 self.position_qty = 0

#         # Entry on Expiry Day
#         if self.position_pe == 0 and now.time() > self.start_time:
#             if self.dte == 0:
#                 expiry_idx  = 1
#             else:
#                 expiry_idx = 0

#             self.spot = self.get_tick(self.now, f"{self.underlying}SPOT")['c']
#             self.moved_spot = self.spot * (1 - self.max_hedge)
#             self.position_qty = int(np.ceil(self.curr_margin / self.spot))
#             payoff_req = self.var / self.position_qty
#             # Round the strike to nearest strike interval
#             strike_diff = strike_diff_dict[self.underlying]
#             self.strike_pe = round((self.moved_spot + payoff_req) / strike_diff) * strike_diff
#             expiry_code = self.get_expiry_code(self.now, self.underlying, expiry_idx)
#             self.symbol_pe = f"{self.underlying}{expiry_code}{self.strike_pe}PE"

#             print("Hedge Symbol PE:", self.symbol_pe, self.spot, self.moved_spot, self.strike_pe, expiry_code, self.position_qty, self.curr_margin, payoff_req)            
#             if self.symbol_pe is not None:
#                 self.success_pe, _ = self.place_trade(self.now, 'BUY', self.position_qty, self.symbol_pe,note="ENTRY HEDGE")
#             if self.success_pe:
#                 self.position_pe = 1


#############################################################################
# APPROCH 3: Overnight Hedge with the above Approach
#############################################################################


# import datetime
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# from utils.definitions import *

# from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

# class HEDGEQAW(EventInterfacePositional):
#     def __init__(self, conn=None):
#         super().__init__(conn)
#         self.strat_id = self.__class__.__name__.lower()
#         self.uid = ""
#         self.var_level = 0.1
#         self.margin = 1e7
#         self.position_pe = 0
#         self.margin_df = None
#         self.entry_date = None
#         self.underlying = "NIFTY"

        
        
#     def set_params_from_uid(self, uid):
#         self.uid = uid
#         parts = uid.split('_', 4)
#         assert parts[0] == self.strat_id
#         self.var_level = float(parts[1])
#         self.margin = float(parts[2])
#         self.max_hedge = float(parts[3])
#         self.otm_perc = float(parts[4])

#     def on_new_day(self):
        
#         self.start_time = datetime.time(9, 45)
#         self.end_time = datetime.time(15,10)
#         self.hedge_opened_today = False
#         self.hedge_closed_today = False

#         if self.margin_df is None:  
#             self.margin_df = self.load_eq_curve()

#         self.dte = self.get_dte_by_underlying(self.now, self.underlying)

#         today = pd.Timestamp(self.now).normalize()
#         self.curr_margin = float(self.margin_df[self.margin_df['date'] == today]['qaw'].iloc[0])
#         self.var = (self.max_hedge - self.var_level) * self.curr_margin
#         print(self.now.date(), "DTE:", self.dte,"Curr Margin:", self.curr_margin, "Var:", self.var)
    
#     def load_eq_curve(self):
#         path = f'/home/mridul/jupiter/storage/qaw_nav.csv'
#         df = pd.read_csv(path,usecols=['date','qaw'], parse_dates=['date'],dayfirst=True)
#         df['qaw'] = pd.to_numeric(df['qaw'], errors='coerce')
#         # df['qaw'] = df['qaw'] * float(self.margin)
#         return df

#     def on_bar_complete(self):

#         # Exit
#         if (self.position_pe == 1 and not self.hedge_closed_today and self.now.time() >= self.start_time and self.now.date() > self.entry_date):
#             self.success_pe, _ = self.place_trade(self.now,'SELL',self.position_qty,self.symbol_pe,note="OVERNIGHT EXIT HEDGE")
#             if self.success_pe:
#                 self.symbol_pe = None
#                 self.position_pe = 0
#                 self.position_qty = 0
#                 self.hedge_closed_today = True

#         # Entry
#         if (self.position_pe == 0 and not self.hedge_opened_today and self.now.time() >= self.end_time):
            
#             if self.dte == 0:
#                 expiry_idx  = 1
#             else:
#                 expiry_idx = 0

#             self.spot = self.get_tick(self.now, f"{self.underlying}SPOT")['c']
#             self.position_qty = int(np.ceil(self.curr_margin / self.spot))
#             payoff_req = self.var / self.position_qty
#             self.moved_spot = self.spot * (1 - self.max_hedge)
#             # Round the strike to nearest strike interval
#             strike_diff = strike_diff_dict[self.underlying]
#             self.strike_pe = round((self.moved_spot + payoff_req) / strike_diff) * strike_diff
#             expiry_code = self.get_expiry_code(self.now, self.underlying, expiry_idx)
#             self.symbol_pe = f"{self.underlying}{expiry_code}{self.strike_pe}PE"
            
#             print("Hedge Symbol PE:", self.symbol_pe, self.spot, self.moved_spot, self.strike_pe, expiry_code, self.position_qty, self.curr_margin, payoff_req)            
#             if self.symbol_pe is not None:
#                 self.success_pe, _ = self.place_trade(self.now,'BUY',self.position_qty,self.symbol_pe,note="OVERNIGHT ENTRY HEDGE")
#                 if self.success_pe:
#                     self.position_pe = 1
#                     self.hedge_opened_today = True
#                     self.entry_date = self.now.date()        


#############################################################################
# APPROCH 4: Overnight Hedge with only opening when not in PP
#############################################################################


import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.definitions import *

from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

class HEDGEQAW(EventInterfacePositional):
    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()
        self.uid = ""
        self.var_level = 0.1
        self.margin = 1e7
        self.position_pe = 0
        self.margin_df = None
        self.entry_date = None
        self.underlying = "NIFTY"

        
        
    def set_params_from_uid(self, uid):
        self.uid = uid
        parts = uid.split('_', 4)
        assert parts[0] == self.strat_id
        self.var_level = float(parts[1])
        self.margin = float(parts[2])
        self.max_hedge = float(parts[3])
        self.otm_perc = float(parts[4])

    def on_new_day(self):
        
        self.start_time = datetime.time(9, 45)
        self.end_time = datetime.time(15,10)
        self.hedge_opened_today = False
        self.hedge_closed_today = False

        if self.margin_df is None:  
            self.margin_df = self.load_eq_curve()

        self.dte = self.get_dte_by_underlying(self.now, self.underlying)

        today = pd.Timestamp(self.now).normalize()
        self.curr_margin = float(self.margin_df[self.margin_df['date'] == today]['qaw'].iloc[0])
        self.var = (self.max_hedge - self.var_level) * self.curr_margin
        print(self.now.date(), "DTE:", self.dte,"Curr Margin:", self.curr_margin, "Var:", self.var)
        self.pp_dates = self.load_put_protection_dates()
            
    def load_eq_curve(self):
        path = f'/home/mridul/jupiter/storage/qaw_nav.csv'
        df = pd.read_csv(path,usecols=['date','qaw'], parse_dates=['date'],dayfirst=True)
        df['qaw'] = pd.to_numeric(df['qaw'], errors='coerce')
        # df['qaw'] = df['qaw'] * float(self.margin)
        return df

    def load_put_protection_dates(self):
        path = f'/home/mridul/jupiter/notebooks/Mridul/pptimings.csv'
        df = pd.read_csv(path)
        df['entry_ts'] = pd.to_datetime(df['entry'], format="%d-%m-%Y %H:%M")
        df['exit_ts'] = pd.to_datetime(df['exittime'], format="%d-%m-%Y %H:%M")
        return df
    
    def is_pp_active(self, now):
        return ((self.pp_dates["entry_ts"] <= now) & (now <= self.pp_dates["exit_ts"])).any()

    def on_bar_complete(self):


        
        # Exit
        if (self.position_pe == 1 and not self.hedge_closed_today and self.now.time() >= self.start_time and self.now.date() > self.entry_date):
            self.success_pe, _ = self.place_trade(self.now,'SELL',self.position_qty,self.symbol_pe,note="OVERNIGHT EXIT HEDGE")
            if self.success_pe:
                self.symbol_pe = None
                self.position_pe = 0
                self.position_qty = 0
                self.hedge_closed_today = True

        # Check if in Put Protection
        if self.is_pp_active(self.now):
            print(self.now, "In Put Protection, skipping hedge actions")
            return

        # Entry
        if (self.position_pe == 0 and not self.hedge_opened_today and self.now.time() == self.end_time):
            
            if self.dte == 0:
                expiry_idx  = 1
            else:
                expiry_idx = 0

            self.spot = self.get_tick(self.now, f"{self.underlying}SPOT")['c']
            self.position_qty = int(np.ceil(self.curr_margin / self.spot))
            payoff_req = self.var / self.position_qty       
            self.moved_spot = self.spot * (1 - self.max_hedge)
            # Round the strike to nearest strike interval
            strike_diff = strike_diff_dict[self.underlying]
            self.strike_pe = round((self.moved_spot + payoff_req) / strike_diff) * strike_diff
            expiry_code = self.get_expiry_code(self.now, self.underlying, expiry_idx)
            self.symbol_pe = f"{self.underlying}{expiry_code}{self.strike_pe}PE"
            
            print("Hedge Symbol PE:", self.symbol_pe, self.spot, self.moved_spot, self.strike_pe, expiry_code, self.position_qty, self.curr_margin, payoff_req)            
            if self.symbol_pe is not None:
                self.success_pe, _ = self.place_trade(self.now,'BUY',self.position_qty,self.symbol_pe,note="OVERNIGHT ENTRY HEDGE")
                if self.success_pe:
                    self.position_pe = 1
                    self.hedge_opened_today = True
                    self.entry_date = self.now.date()                            