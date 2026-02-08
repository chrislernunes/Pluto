import datetime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from utils.definitions import *
from utils.sessions import *
import direct_redis

if REDIS:
    from engine.ems import EventInterfacePositional, lot_size_dict, strike_diff_dict
else:
    from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

# r = direct_redis.DirectRedis()

import datetime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
# from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict
from utils.definitions import *
from utils.sessions import *
# from xxx.strat_live.general import *

# import direct_redis

# r = direct_redis.DirectRedis()

class PUTRCS(EventInterfacePositional):

    def __init__(self):
        super().__init__()
        self.strat_id = self.__class__.__name__.lower()
        self.signal = 0
        self.last_expiry = None
        self.futprice = None
        self.old_p1 = None
        self.curr_pos = {}   # holds legs expiring today
        self.next_pos = {}   # holds legs expiring next week
        self.pending_entry = None  # Track pending entry params

    def get_random_uid(self):
        # Select
        self.session = 'x0'# np.random.choice(sessions)
        self.underlying = np.random.choice(underlyings)
        self.selector = 'PCT' # np.random.choice(selectors)
        if self.selector == 'M':
            self.selector_val = np.random.choice(moneynesses)
        elif self.selector == 'P':
            self.selector_val = np.random.choice(seek_prices)
        elif self.selector == 'PCT':
            self.selector_val = np.random.choice([0.0025, 0.0015, 0.00075, 0.0001, 0.0002, 0.00015, 0.0005, 0.0003, 0.00045])
        self.sl_pct = round(.05 * round(np.random.choice(np.random.rand(10)*0.5).round(2)/.05), 2)
        self.tgt_pct = round(.05 * round(np.random.choice(np.random.random(1)).round(2)/.05), 2) #np.random.choice(tgt_pcts)
        self.delay = np.random.choice(range(0, 30, 5))
        self.mult_1 = 0.5
        self.mult_2 = 0.25
        self.window  = 5
        return self.get_uid_from_params()

    def set_params_from_uid(self, uid):
        # print(uid)
        s = uid.split('_')
        try:
            assert s[0] == self.strat_id
        except AssertionError:
            raise ValueError(f'Invalid UID {uid} for strat ID {self.strat_id}')
        s = s[1:]
        self.session = s.pop(0)
        self.delay = int(s.pop(0))#=='True'
        self.underlying = s.pop(0)
        self.selector = s.pop(0)
        if self.selector == 'P' or self.selector == 'M':
            self.selector_val = int(s.pop(0))
        elif self.selector == 'PCT':
            self.selector_val = float(s.pop(0))
        self.sl_pct = float(s.pop(0))
        self.tgt_pct = float(s.pop(0))
        self.mult_1 = float(s.pop(0))
        self.mult_2 = float(s.pop(0))
        self.window = int(s.pop(0))
        # CROSS CHECK
        assert len(s)==0
        self.gen_uid = self.get_uid_from_params()
        #print(self.gen_uid)
        assert uid == self.gen_uid
        self.uid = uid
        # print(self.uid)
    
    def get_uid_from_params(self):
        return f"""
        {self.strat_id}_
        {self.session}_
        {self.delay}_
        {self.underlying}_
        {self.selector}_
        {self.selector_val}_
        {self.sl_pct}_
        {self.tgt_pct}_
        {self.mult_1}_
        {self.mult_2}_
        {self.window}_
        """.replace('\n', '').replace(' ', '').strip('_')

    def get_nifty_weekly_movement_V2(self):
        df = self.get_all_ticks_by_symbol(self.underlying)
        df.index = pd.to_datetime(df.timestamp)  # Ensure datetime index

        # Filter to only up to 'now'
        df = df[df.index <= self.now]
        
        # Resample to daily close
        daily_close = df["c"].resample("D").last().dropna()

        window = self.window * 2  # Convert weeks to approximate trading days
        if len(daily_close) < window + 1:
            return None  # not enough history yet

        # Calculate SMA using rolling window
        sma = daily_close.rolling(window=window).mean()
        std = daily_close.rolling(window=window).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std

        prev_day_close = daily_close.iloc[-2]   # day just before the most recent
        prev_day_sma = sma.iloc[-2]             # SMA value for the same day
        prev_day_upper = upper_band.iloc[-2]
        prev_day_lower = lower_band.iloc[-2]

        # if prev_day_close > prev_day_sma:
        #     if prev_day_close < prev_day_upper:
        #         return "uptrend"
        #     else:
        #         return "reversion_up"  # Close above upper band
        # else:
        #     if prev_day_close > prev_day_lower:
        #         return "downtrend"
        #     else:
        #         return "reversion_down"  # Close below lower band

        if prev_day_close > prev_day_sma:
            return "uptrend"
        else:
            return "downtrend"
            

    def get_nifty_weekly_movement(self):
        """
        Uses weekly closes (Wednesday 15:29) to compute 5-week SMA.
        Returns 'uptrend' if SMA < prev week close, 'downtrend' if prev week close < SMA.
        """
        df = self.get_all_ticks_by_symbol(self.underlying)
        df.index = pd.to_datetime(df.timestamp)  # Ensure datetime index

        # Filter to only up to 'now'
        df = df[df.index <= self.now]

        # weekly resample – Wednesday close
        weekly_close = df["c"].resample("W-WED").last()

        needed = self.window + 1                # e.g. 5-week SMA needs last 6 weeks
        if len(weekly_close) < needed:
            return None                    # not enough history yet

        last_k_plus_one = weekly_close.tail(needed)

        prev_week_close = last_k_plus_one.iloc[-2]   # week just before the most recent
        sma_k = last_k_plus_one.iloc[:-1].mean()     # SMA of the preceding k weeks

        return "uptrend" if prev_week_close > sma_k else "downtrend"

    def _enter_next_week(self, opt_type, entry_type,expiry_idx):
        """Tries to open ATM-50-25 PE 1-3-2 ratio in the next weekly expiry. Returns None if any symbol missing."""
        # atm_pe = self.find_symbol_by_moneyness(self.now, self.underlying, 1, opt_type, -2)
        atm_pe = self.find_symbol_by_moneyness(self.now, self.underlying, expiry_idx, opt_type, 0)
        if atm_pe is None:
            print("Symbol 1 Not found, will retry next bar")
            return None

        price_atm = self.get_tick(self.now, atm_pe)['c']
        otm_pe_3  = self.find_symbol_by_premium(self.now, self.underlying, expiry_idx, opt_type, price_atm * self.mult_1)
        if otm_pe_3 is None:
            print("Symbol 2 Not found, will retry next bar")
            return None

        price_3 = self.get_tick(self.now, otm_pe_3)['c']
        otm_pe_2 = self.find_symbol_by_premium(self.now, self.underlying, expiry_idx, opt_type, price_3 * self.mult_2)
        if otm_pe_2 is None:
            print("Symbol 3 Not found, will retry next bar")
            return None
        
        if otm_pe_2 == otm_pe_3 or otm_pe_3 == atm_pe or otm_pe_2 == atm_pe:
            print("Duplicate symbols found, will retry next bar")
            return None

        lot = self.get_lot_size(self.underlying)
        ok1, _ = self.place_trade(self.now, 'BUY',  lot,       atm_pe,note=f'ATM ENTRY next-exp {entry_type}')
        ok2, _ = self.place_trade(self.now, 'SELL', lot * 3,   otm_pe_3, note=f'OTM3 ENTRY next-exp {entry_type}')
        ok3, _ = self.place_trade(self.now, 'BUY',  lot * 2,   otm_pe_2, note=f'OTM2 ENTRY next-exp {entry_type}')

        if ok1 and ok2 and ok3:
            return {'atm': atm_pe, 'otm3': otm_pe_3, 'otm2': otm_pe_2}
        return None

    def _exit_position(self, pos):
        """Flattens a previously stored position dictionary."""
        lot = self.get_lot_size(self.underlying)    

        # self.place_trade(self.now, 'SELL', lot, pos['atm'],price=self.get_last_available_tick(pos['atm'])['c'], note=f'ATM PE EXIT current-exp_{self.get_last_available_tick(pos["atm"])["timestamp"]}')
        # self.place_trade(self.now, 'BUY',  lot * 3, pos['otm3'], price=self.get_last_available_tick(pos['otm3'])['c'], note=f'OTM3 PE EXIT current-exp_{self.get_last_available_tick(pos["otm3"])["timestamp"]}')
        # self.place_trade(self.now, 'SELL', lot * 2, pos['otm2'], price=self.get_last_available_tick(pos['otm2'])['c'], note=f'OTM2 PE EXIT current-exp_{self.get_last_available_tick(pos["otm2"])["timestamp"]}')
        
        s1 , _ = self.place_trade(self.now, 'SELL', lot, pos['atm'], note=f'ATM EXIT current-exp')
        s2 , _ = self.place_trade(self.now, 'BUY',  lot * 3, pos['mid'], note=f'MID EXIT current-exp')
        s3 , _ = self.place_trade(self.now, 'SELL', lot * 2, pos['far'], note=f'FAR EXIT current-exp')

        if s1 and s2 and s3:
            return True
        
        raise ValueError("Exit trades failed")

    def _enter_next_week_v2(self, opt_type, entry_type, expiry_idx):
        """
        Tries to open ATM–MID–FAR 1–3–2 ratio in the next weekly expiry.
        Objective: net credit spread.
        Returns None if unable to construct.
        """

        # ---- ATM leg ----
        atm = self.find_symbol_by_moneyness(self.now, self.underlying, expiry_idx, opt_type, 0)
        if atm is None:
            print("ATM symbol not found, will retry next bar")
            return None

        price_atm = self.get_tick(self.now, atm)['c']

        # ---- MID leg (3x SELL) ----
        mid = self.find_symbol_by_premium(self.now, self.underlying, expiry_idx, opt_type, price_atm * self.mult_1)
        if mid is None:
            print("MID symbol not found, will retry next bar")
            return None

        price_mid = self.get_tick(self.now, mid)['c']

        # ---- FAR leg (2x BUY) ----
        far = self.find_symbol_by_premium(self.now, self.underlying, expiry_idx, opt_type, price_mid * self.mult_2)
        if far is None:
            print("FAR symbol not found, will retry next bar")
            return None

        # ---- Strike info ----
        strike_atm = self.parse_strike_from_symbol(atm)
        strike_mid = self.parse_strike_from_symbol(mid)
        strike_far = self.parse_strike_from_symbol(far)
        strike_step = self.get_strike_diff(self.underlying)
        print("Strikes:", strike_atm, strike_mid, strike_far, self.now)
        # Reject if all strikes same
        if strike_mid == strike_far or strike_far == strike_atm or strike_mid == strike_atm:
            print("Duplicate symbols found, will retry next bar")
            return None

            # ---- Net premium helper ----
        def net_premium():
            p_atm = self.get_tick(self.now, atm)
            p_mid = self.get_tick(self.now, mid)
            p_far = self.get_tick(self.now, far)
            if p_atm is None or p_mid is None or p_far is None:
                return None
            return (-1 * p_atm['c']) + (3 * p_mid['c']) - (2 * p_far['c'])


        np = net_premium()
        if np is None:
            print("Missing tick data, retry next bar")
            return None

        if np < 0:
            print("Debit spread detected, attempting adjustments")

            # ---- Step 1: Walk MID toward ATM ----
            while True:
                strike_mid = self.parse_strike_from_symbol(mid)
                strike_atm = self.parse_strike_from_symbol(atm)

                # Determine step direction based on option type
                if opt_type.upper() == 'PE':
                    next_strike = strike_mid - strike_step  # PE moves down
                else:  # CE
                    next_strike = strike_mid + strike_step  # CE moves up

                new_mid = self.replace_strike_in_symbol(mid, next_strike)

                if not new_mid:
                    break

                mid = new_mid

                np = net_premium()
                if np is None:
                    print("Missing tick after MID adjustment, retry next bar")
                    return None

                if np > 0:
                    break

            np = net_premium()
            if np is None:
                print("Missing tick data, retry next bar")
                return None

            if np < 0:
                strike_far = self.parse_strike_from_symbol(far)

                for i in range(1, 6):
                    if opt_type.upper() == 'PE':
                        next_strike = strike_far - (i * strike_step)  # PE: move down
                    else:  # CE
                        next_strike = strike_far + (i * strike_step)  # CE: move up
                    
                    new_far = self.replace_strike_in_symbol(far, next_strike)
        
                    if not new_far:
                        continue
                    
                    far = new_far

                    np = net_premium()
                    if np is None:
                        print("Missing tick after FAR adjustment, retry next bar")
                        return None

                    if np > 0:
                        break

            np = net_premium()
            if np is None or np < 0:
                print("Unable to form credit spread, retry next bar")
                return None

        # ---- Place trades ----
        lot = self.get_lot_size(self.underlying)

        ok1, _ = self.place_trade(self.now, 'BUY', lot, atm,note=f'ATM ENTRY next-exp {entry_type}')
        ok2, _ = self.place_trade(self.now, 'SELL', lot * 3, mid,note=f'MID ENTRY next-exp {entry_type}')
        ok3, _ = self.place_trade(self.now, 'BUY', lot * 2, far,note=f'FAR ENTRY next-exp {entry_type}')

        if ok1 and ok2 and ok3:
            return {'atm': atm, 'mid': mid, 'far': far}

        return None



    def on_new_day(self):

        # One-time init
        if not hasattr(self, 'entry_done'):
            self.entry_done = False
            self.curr_pos = {}
            self.next_pos = {}

        # Detect expiry
        self.dte = self.get_nearest_expiry(self.now, self.underlying) - self.now.date()
        self.is_expiry_day = self.dte.days == 0

        # Roll positions ONCE on expiry day
        if self.is_expiry_day:
            self.curr_pos, self.next_pos = self.next_pos, {}
            self.entry_done = False   # allow fresh attempts


        # -------- Underlying → Spot symbol mapping --------
        if self.underlying == 'BANKNIFTY':
            self.mysymbol = 'BANKNIFTYSPOT'
        elif self.underlying == 'NIFTY':
            self.mysymbol = 'NIFTYSPOT'
        elif self.underlying == 'FINNIFTY':
            self.mysymbol = 'FINNIFTYSPOT'
        elif self.underlying == 'MIDCPNIFTY':
            self.mysymbol = "MIDCPNIFTYSPOT"
        elif self.underlying == 'SENSEX':
            self.mysymbol = "SENSEXSPOT"
        elif self.underlying == 'GOLDM':
            self.mysymbol = "GOLDMSPOT"
        else:
            raise ValueError(f'Unknown Underlying: {self.underlying}')

        # -------- Session timing --------
        self.lot_size = self.get_lot_size(self.underlying)
        self.session_vals = sessions_dict[self.session]

        self.start_time = self.session_vals['start_time']
        # self.stop_time = self.session_vals['stop_time']
        self.stop_time = datetime.time(22, 20)

        # Add delay to start time
        dtn = datetime.datetime.now()
        self.start_time = (
            datetime.datetime.combine(dtn.date(), self.start_time)
            + datetime.timedelta(minutes=self.delay)
        ).time()

        
        
    def on_event(self):
        pass

    def on_bar_complete(self):


        # ==================================================
        # 1. EXIT LOGIC — independent & highest priority
        # ==================================================
        if self.is_expiry_day and self.now.time() >= self.stop_time and self.curr_pos:
            self._exit_position(self.curr_pos)
            self.curr_pos = {}   # old position fully closed


        # ==================================================
        # 2. ENTRY LOGIC — retry until success
        # ==================================================
        if self.entry_done:
            return

        if self.now.time() < self.start_time:
            return

        # Always try entering NEXT expiry
        result = self._enter_next_week_v2(
            opt_type="CE",
            entry_type="CE",
            expiry_idx=1 if self.is_expiry_day else 0
        )

        if result:
            self.next_pos = result
            self.entry_done = True


###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################


# import datetime, time
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('dark_background')

# from utils.definitions import *
# from utils.sessions import *
# import direct_redis

# if REDIS:
#     from engine.ems import EventInterfacePositional, lot_size_dict, strike_diff_dict
# else:
#     from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

# # r = direct_redis.DirectRedis()

# import datetime, time
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('dark_background')
# # from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict
# from utils.definitions import *
# from utils.sessions import *
# # from xxx.strat_live.general import *

# # import direct_redis

# # r = direct_redis.DirectRedis()

# class PUTRCS(EventInterfacePositional):

#     def __init__(self):
#         super().__init__()
#         self.strat_id = self.__class__.__name__.lower()
#         self.signal = 0
#         self.last_expiry = None
#         self.futprice = None
#         self.old_p1 = None
#         self.curr_pos = {}   # holds legs expiring today
#         self.next_pos = {}   # holds legs expiring next week
#         self.pending_entry = None  # Track pending entry params

#     def get_random_uid(self):
#         # Select
#         self.session = 'x0'# np.random.choice(sessions)
#         self.underlying = np.random.choice(underlyings)
#         self.selector = 'PCT' # np.random.choice(selectors)
#         if self.selector == 'M':
#             self.selector_val = np.random.choice(moneynesses)
#         elif self.selector == 'P':
#             self.selector_val = np.random.choice(seek_prices)
#         elif self.selector == 'PCT':
#             self.selector_val = np.random.choice([0.0025, 0.0015, 0.00075, 0.0001, 0.0002, 0.00015, 0.0005, 0.0003, 0.00045])
#         self.sl_pct = round(.05 * round(np.random.choice(np.random.rand(10)*0.5).round(2)/.05), 2)
#         self.tgt_pct = round(.05 * round(np.random.choice(np.random.random(1)).round(2)/.05), 2) #np.random.choice(tgt_pcts)
#         self.delay = np.random.choice(range(0, 30, 5))
#         self.mult_1 = 0.5
#         self.mult_2 = 0.25
#         self.window  = 5
#         return self.get_uid_from_params()

#     def set_params_from_uid(self, uid):
#         # print(uid)
#         s = uid.split('_')
#         try:
#             assert s[0] == self.strat_id
#         except AssertionError:
#             raise ValueError(f'Invalid UID {uid} for strat ID {self.strat_id}')
#         s = s[1:]
#         self.session = s.pop(0)
#         self.delay = int(s.pop(0))#=='True'
#         self.underlying = s.pop(0)
#         self.selector = s.pop(0)
#         if self.selector == 'P' or self.selector == 'M':
#             self.selector_val = int(s.pop(0))
#         elif self.selector == 'PCT':
#             self.selector_val = float(s.pop(0))
#         self.sl_pct = float(s.pop(0))
#         self.tgt_pct = float(s.pop(0))
#         self.mult_1 = float(s.pop(0))
#         self.mult_2 = float(s.pop(0))
#         self.window = int(s.pop(0))
#         # CROSS CHECK
#         assert len(s)==0
#         self.gen_uid = self.get_uid_from_params()
#         #print(self.gen_uid)
#         assert uid == self.gen_uid
#         self.uid = uid
#         # print(self.uid)
    
#     def get_uid_from_params(self):
#         return f"""
#         {self.strat_id}_
#         {self.session}_
#         {self.delay}_
#         {self.underlying}_
#         {self.selector}_
#         {self.selector_val}_
#         {self.sl_pct}_
#         {self.tgt_pct}_
#         {self.mult_1}_
#         {self.mult_2}_
#         {self.window}_
#         """.replace('\n', '').replace(' ', '').strip('_')

#     def _enter_next_week(self, opt_type, entry_type):
#         """Tries to open ATM-50-25 PE 1-3-2 ratio in the next weekly expiry. Returns None if any symbol missing."""
#         atm_pe = self.find_symbol_by_moneyness(self.now, self.underlying, 1, opt_type, 0)
#         if atm_pe is None:
#             print("Symbol 1 Not found, will retry next bar")
#             return None

#         price_atm = self.get_tick(self.now, atm_pe)['c']
#         otm_pe_3  = self.find_symbol_by_premium(self.now, self.underlying, 1, opt_type, price_atm * self.mult_1)
#         if otm_pe_3 is None:
#             print("Symbol 2 Not found, will retry next bar")
#             return None

#         price_3 = self.get_tick(self.now, otm_pe_3)['c']
#         otm_pe_2 = self.find_symbol_by_premium(self.now, self.underlying, 1, opt_type, price_3 * self.mult_2)
#         if otm_pe_2 is None:
#             print("Symbol 3 Not found, will retry next bar")
#             return None

#         lot = self.get_lot_size(self.underlying)
#         ok1, _ = self.place_trade(self.now, 'BUY',  lot,       atm_pe,note=f'ATM ENTRY next-exp {entry_type}')
#         ok2, _ = self.place_trade(self.now, 'SELL', lot * 3,   otm_pe_3, note=f'OTM3 ENTRY next-exp {entry_type}')
#         ok3, _ = self.place_trade(self.now, 'BUY',  lot * 2,   otm_pe_2, note=f'OTM2 ENTRY next-exp {entry_type}')

#         if ok1 and ok2 and ok3:
#             return {'atm': atm_pe, 'otm3': otm_pe_3, 'otm2': otm_pe_2}
#         return None

#     def _exit_position(self, pos):
#         """Flattens a previously stored position dictionary."""
#         lot = self.get_lot_size(self.underlying)    

#         # self.place_trade(self.now, 'SELL', lot, pos['atm'],price=self.get_last_available_tick(pos['atm'])['c'], note=f'ATM PE EXIT current-exp_{self.get_last_available_tick(pos["atm"])["timestamp"]}')
#         # self.place_trade(self.now, 'BUY',  lot * 3, pos['otm3'], price=self.get_last_available_tick(pos['otm3'])['c'], note=f'OTM3 PE EXIT current-exp_{self.get_last_available_tick(pos["otm3"])["timestamp"]}')
#         # self.place_trade(self.now, 'SELL', lot * 2, pos['otm2'], price=self.get_last_available_tick(pos['otm2'])['c'], note=f'OTM2 PE EXIT current-exp_{self.get_last_available_tick(pos["otm2"])["timestamp"]}')
        
#         self.place_trade(self.now, 'SELL', lot, pos['atm'], note=f'ATM EXIT current-exp')
#         self.place_trade(self.now, 'BUY',  lot * 3, pos['otm3'], note=f'OTM3 EXIT current-exp')
#         self.place_trade(self.now, 'SELL', lot * 2, pos['otm2'], note=f'OTM2 EXIT current-exp')

#     def on_new_day(self):
#         # ...

#         # if self.now().date()==datetime.datetime.strptime("2023-03-29", "%Y-%m-%d").date():
#         self.dte = self.get_nearest_expiry(self.now,self.underlying) - self.now.date()
#         # print(f"Current DTE: {self.dte.days} for {self.underlying} on {self.now.date()}")
#         if self.dte.days != 0:
#             self.is_expiry_day = 0
#         else:
#             self.is_expiry_day = 1  
#             self.curr_pos, self.next_pos = self.next_pos, {}


#         if self.underlying == 'BANKNIFTY':
#             self.mysymbol = 'BANKNIFTYSPOT'
#         elif self.underlying == 'NIFTY':
#             self.mysymbol = 'NIFTYSPOT'
#         elif self.underlying == 'FINNIFTY':
#             self.mysymbol = 'FINNIFTYSPOT'
#         elif self.underlying == 'MIDCPNIFTY':
#             self.mysymbol = "MIDCPNIFTYSPOT" 
#         elif self.underlying == 'SENSEX':
#             self.mysymbol = "SENSEXSPOT"
#         else:
#             raise ValueError(f'Unknown Underlying: {self.underlying}')   
        
#         self.lot_size = self.get_lot_size(self.underlying) 
#         # ...
#         self.session_vals = sessions_dict[self.session]
#         self.start_time = self.session_vals['start_time']# #datetime.time(9, 15)#
#         self.stop_time = self.session_vals['stop_time']#datetime.time (15, 14)#
#         # ADD DELAY TO START TIME
#         dtn = datetime.datetime.now()
#         self.start_time = (datetime.datetime.combine(dtn.date(), self.start_time) + datetime.timedelta(minutes=self.delay)).time()
#         # self.curr_pos, self.next_pos = self.next_pos, {}        
#         # df = self.get_all_ticks_by_symbol(self.underlying)
        
        
#     def on_event(self):
#         pass

#     def on_bar_complete(self):
        
#         if not self.is_expiry_day:
#             return
        

#         # ---------- 15:29  – EXIT the position that expires TODAY ----------
#         if self.now.time() >= self.stop_time and self.curr_pos:
#             self._exit_position(self.curr_pos)     # helper, see below
#             self.curr_pos = {}                     # nothing left to hold

#         # ---------- 14:30  – E NTER next-week position ----------------------
#         # Try entry if next_pos is empty and either it's start_time or a pending entry exists
#         should_try_entry = (
#             (self.now.time() == self.start_time and not self.next_pos)
#             or (self.pending_entry and not self.next_pos)
#         )
#         if should_try_entry:
#             if not self.pending_entry:
#                 opt_type = "PE"
#                 note = "PE Spread"
#                 self.pending_entry = (opt_type, note)

#             # Try to enter the trade
#             result = self._enter_next_week(*self.pending_entry)
#             if result:
#                 self.next_pos = result
#                 self.pending_entry = None  # Clear pending if successful
#             # else: keep pending_entry for next bar


# class PUTRCS(EventInterfacePositional):

#     def __init__(self):
#         super().__init__()
#         self.strat_id = self.__class__.__name__.lower()
#         self.signal = 0
#         self.last_expiry = None
#         self.futprice = None
#         self.old_p1 = None
#         self.position = None  # Single active position
#         self.pending_entry = None  # Track pending entry params
#         self.otm3_entry_price = None
#         self.otm3_stoploss = None
#         self.otm3_reentry_pending = False
#         self.otm3_sold_lots = 0
#         self.last_exit_time = None  # Track last exit time for delayed entry

#     def get_random_uid(self):
#         # Select
#         self.session = 'x0'# np.random.choice(sessions)
#         self.underlying = np.random.choice(underlyings)
#         self.selector = 'PCT' # np.random.choice(selectors)
#         if self.selector == 'M':
#             self.selector_val = np.random.choice(moneynesses)
#         elif self.selector == 'P':
#             self.selector_val = np.random.choice(seek_prices)
#         elif self.selector == 'PCT':
#             self.selector_val = np.random.choice([0.0025, 0.0015, 0.00075, 0.0001, 0.0002, 0.00015, 0.0005, 0.0003, 0.00045])
#         self.sl_pct = round(np.random.choice(np.random.rand(10) * 0.5), 2)
#         self.tgt_pct = round(np.random.random(), 2) #np.random.choice(tgt_pcts)
#         self.delay = np.random.choice(range(0, 30, 5))
#         self.mult_1 = 0.5
#         self.mult_2 = 0.25
#         self.window  = 5
#         return self.get_uid_from_params()

#     def set_params_from_uid(self, uid):
#         # print(uid)
#         s = uid.split('_')
#         try:
#             assert s[0] == self.strat_id
#         except AssertionError:
#             raise ValueError(f'Invalid UID {uid} for strat ID {self.strat_id}')
#         s = s[1:]
#         self.session = s.pop(0)
#         self.delay = int(s.pop(0))#=='True'
#         self.underlying = s.pop(0)
#         self.selector = s.pop(0)
#         if self.selector == 'P' or self.selector == 'M':
#             self.selector_val = int(s.pop(0))
#         elif self.selector == 'PCT':
#             self.selector_val = float(s.pop(0))
#         self.sl_pct = float(s.pop(0))
#         self.tgt_pct = float(s.pop(0))
#         self.mult_1 = float(s.pop(0))
#         self.mult_2 = float(s.pop(0))
#         self.window = int(s.pop(0))
#         # CROSS CHECK
#         assert len(s)==0
#         self.gen_uid = self.get_uid_from_params()
#         #print(self.gen_uid)
#         assert uid == self.gen_uid
#         self.uid = uid
#         # print(self.uid)
    
#     def get_uid_from_params(self):
#         return f"""
#         {self.strat_id}_
#         {self.session}_
#         {self.delay}_
#         {self.underlying}_
#         {self.selector}_
#         {self.selector_val}_
#         {self.sl_pct}_
#         {self.tgt_pct}_
#         {self.mult_1}_
#         {self.mult_2}_
#         {self.window}_
#         """.replace('\n', '').replace(' ', '').strip('_')

#     def _enter_position(self, opt_type, entry_type):
#         """Tries to open ATM-50-25 PE 1-3-2 ratio in the next weekly expiry. Returns None if any symbol missing."""
#         atm_pe = self.find_symbol_by_moneyness(self.now, self.underlying, 1, opt_type, 0)
#         if atm_pe is None:
#             print("Symbol 1 Not found, will retry next bar")
#             return None

#         price_atm = self.get_tick(self.now, atm_pe)['c']
#         otm_pe_3  = self.find_symbol_by_premium(self.now, self.underlying, 1, opt_type, price_atm * self.mult_1)
#         if otm_pe_3 is None:
#             print("Symbol 2 Not found, will retry next bar")
#             return None

#         price_3 = self.get_tick(self.now, otm_pe_3)['c']
#         otm_pe_2 = self.find_symbol_by_premium(self.now, self.underlying, 1, opt_type, price_3 * self.mult_2)
#         if otm_pe_2 is None:
#             print("Symbol 3 Not found, will retry next bar")
#             return None

#         lot = self.get_lot_size(self.underlying)
#         ok1, _ = self.place_trade(self.now, 'BUY',  lot,       atm_pe,note=f'ATM ENTRY {entry_type}')
#         ok2, _ = self.place_trade(self.now, 'SELL', lot * 3,   otm_pe_3, note=f'OTM3 ENTRY {entry_type}')
#         ok3, _ = self.place_trade(self.now, 'BUY',  lot * 2,   otm_pe_2, note=f'OTM2 ENTRY {entry_type}')

#         if ok1 and ok2 and ok3:
#             self.otm3_entry_price = price_3
#             self.otm3_stoploss = price_3 * (1 + self.sl_pct)
#             self.otm3_reentry_pending = False
#             self.otm3_sold_lots = 3
#             return {'atm': atm_pe, 'otm3': otm_pe_3, 'otm2': otm_pe_2}
#         return None

#     def _exit_position(self, pos):
#         """Flattens a previously stored position dictionary."""
#         lot = self.get_lot_size(self.underlying)
#         self.place_trade(self.now, 'SELL', lot, pos['atm'], note=f'ATM EXIT')
#         self.place_trade(self.now, 'BUY',  lot * self.otm3_sold_lots, pos['otm3'], note=f'OTM3 EXIT')
#         self.place_trade(self.now, 'SELL', lot * 2, pos['otm2'], note=f'OTM2 EXIT')
#         self.otm3_sold_lots = 0
#         self.otm3_entry_price = None
#         self.otm3_stoploss = None
#         self.otm3_reentry_pending = False

#     def on_new_day(self):
#         self.dte = self.get_nearest_expiry(self.now,self.underlying) - self.now.date()
#         if self.dte.days != 0:
#             self.is_expiry_day = 0
#         else:
#             self.is_expiry_day = 1
#         if self.underlying == 'BANKNIFTY':
#             self.mysymbol = 'BANKNIFTYSPOT'
#         elif self.underlying == 'NIFTY':
#             self.mysymbol = 'NIFTYSPOT'
#         elif self.underlying == 'FINNIFTY':
#             self.mysymbol = 'FINNIFTYSPOT'
#         elif self.underlying == 'MIDCPNIFTY':
#             self.mysymbol = "MIDCPNIFTYSPOT"
#         elif self.underlying == 'SENSEX':
#             self.mysymbol = "SENSEXSPOT"
#         else:
#             raise ValueError(f'Unknown Underlying: {self.underlying}')
#         self.lot_size = self.get_lot_size(self.underlying)
#         self.session_vals = sessions_dict[self.session]
#         self.start_time = self.session_vals['start_time']
#         self.stop_time = self.session_vals['stop_time']
#         dtn = datetime.datetime.now()
#         self.roll_time = (datetime.datetime.combine(dtn.date(), self.start_time) + datetime.timedelta(minutes=self.delay)).time()
#         self.next_entry_time = (datetime.datetime.combine(dtn.date(), self.roll_time) + datetime.timedelta(minutes=5)).time()
#         # Reset last_exit_time at the start of each day
#         self.last_exit_time = None
        
        
#     def on_event(self):
#         pass

#     def on_bar_complete(self):
        
#         if not self.is_expiry_day:
#             # OTM3 SL/re-entry logic still applies on non-expiry days
#             pass
#         # ---------- ROLL LOGIC: EXIT at roll_time, ENTER at next_entry_time ----------
#         if self.is_expiry_day and self.now.time() == self.roll_time:
#             if self.position:
#                 self._exit_position(self.position)
#                 self.position = None
#                 self.last_exit_time = self.now
#         if self.is_expiry_day and self.now.time() == self.next_entry_time:
#             if self.position is None and (
#                 self.last_exit_time is None or (
#                     self.last_exit_time.date() == self.now.date() and self.last_exit_time.time() == self.roll_time
#                 )
#             ):
#                 opt_type = "PE"
#                 note = "PE Spread"
#                 result = self._enter_position(opt_type, note)
#                 if result:
#                     self.position = result
#         # ---------- OTM3 SL and re-entry logic ----------
#         if self.position and self.otm3_sold_lots > 1:
#             otm3_symbol = self.position['otm3']
#             current_price = self.get_tick(self.now, otm3_symbol)['c']
#             if current_price >= self.otm3_stoploss:
#                 lot = self.get_lot_size(self.underlying)
#                 self.place_trade(self.now, 'BUY', lot * 2, otm3_symbol, note='OTM3 SL HIT')
#                 self.otm3_sold_lots = 1
#                 self.otm3_reentry_pending = True
#         if self.position and self.otm3_reentry_pending and self.otm3_sold_lots == 1:
#             otm3_symbol = self.position['otm3']
#             current_price = self.get_tick(self.now, otm3_symbol)['c']
#             if current_price <= self.otm3_entry_price:
#                 lot = self.get_lot_size(self.underlying)
#                 self.place_trade(self.now, 'SELL', lot * 2, otm3_symbol, note='OTM3 RE-ENTRY')
#                 self.otm3_sold_lots = 3
#                 self.otm3_reentry_pending = False    
#                 # Update stop loss after re-entry
#                 self.otm3_entry_price = current_price
#                 self.otm3_stoploss = current_price * (1 + self.sl_pct)    