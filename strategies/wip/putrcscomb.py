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

# r = direct_redis.DirectRedis()

import datetime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict
from utils.definitions import *
from utils.sessions import *
# from xxx.strat_live.general import *

# import direct_redis

# r = direct_redis.DirectRedis()

class PUTRCSCOMB(EventInterfacePositional):

    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()
        self.signal = 0
        self.last_expiry = None
        self.futprice = None
        self.old_p1 = None
        # --- Cross expiry logic additions ---
        self.underlyings = ['NIFTY', 'SENSEX']
        self.curr_pos = {u: {} for u in self.underlyings}   # holds legs expiring today for each underlying
        self.next_pos = {u: {} for u in self.underlyings}   # holds legs expiring next week for each underlying
        self.is_expiry_day = {u: 0 for u in self.underlyings}
        self.dte = {u: datetime.timedelta(0) for u in self.underlyings}
        self.lot_size = {u: 0 for u in self.underlyings}
        self.mysymbol = {u: '' for u in self.underlyings}
        self.session_vals = None
        self.start_time = None
        self.stop_time = None
        self.delay = 0
        self.mult_1 = 0.5
        self.mult_2 = 0.25

    def get_random_uid(self):
        # Select
        self.session = 'x0'# np.random.choice(sessions)
        # self.underlying = np.random.choice(self.underlyings)
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
        # self.underlying = s.pop(0)
        self.selector = s.pop(0)
        if self.selector == 'P' or self.selector == 'M':
            self.selector_val = int(s.pop(0))
        elif self.selector == 'PCT':
            self.selector_val = float(s.pop(0))
        self.sl_pct = float(s.pop(0))
        self.tgt_pct = float(s.pop(0))
        self.mult_1 = float(s.pop(0))
        self.mult_2 = float(s.pop(0))
        
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
        {self.selector}_
        {self.selector_val}_
        {self.sl_pct}_
        {self.tgt_pct}_
        {self.mult_1}_
        {self.mult_2}_
        """.replace('\n', '').replace(' ', '').strip('_')

    def _enter_next_week(self, underlying):
        """Opens ATM-50-25 PE 1-3-2 ratio in the *next* weekly expiry for the given underlying."""
        self.underlying = underlying
        atm_pe = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'PE', 0)
        if atm_pe is None:
            return {}

        price_atm = self.get_tick(self.now, atm_pe)['c']
        otm_pe_3  = self.find_symbol_by_premium(self.now, self.underlying, 0, 'PE', price_atm * self.mult_1)
        if otm_pe_3 is None:
            return {}

        price_3 = self.get_tick(self.now, otm_pe_3)['c']
        otm_pe_2 = self.find_symbol_by_premium(self.now, self.underlying, 0, 'PE', price_3 * self.mult_2)
        if otm_pe_2 is None:
            return {}

        lot = self.get_lot_size(self.underlying)

        ok1, _ = self.place_trade(self.now, 'BUY',  lot,       atm_pe,note=f'{underlying} ATM PE ENTRY next-exp')
        ok2, _ = self.place_trade(self.now, 'SELL', lot * 3,   otm_pe_3, note=f'{underlying} OTM3 PE ENTRY next-exp')
        ok3, _ = self.place_trade(self.now, 'BUY',  lot * 2,   otm_pe_2, note=f'{underlying} OTM2 PE ENTRY next-exp')

        if ok1 and ok2 and ok3:
            return {'atm': atm_pe, 'otm3': otm_pe_3, 'otm2': otm_pe_2}
        return {}

    def _exit_position(self, pos, underlying):
        """Flattens a previously stored position dictionary for the given underlying."""
        self.underlying = underlying
        lot = self.get_lot_size(self.underlying)
        self.place_trade(self.now, 'SELL', lot, pos['atm'], note=f'{underlying} ATM PE EXIT current-exp')
        self.place_trade(self.now, 'BUY',  lot * 3, pos['otm3'], note=f'{underlying} OTM3 PE EXIT current-exp')
        self.place_trade(self.now, 'SELL', lot * 2, pos['otm2'], note=f'{underlying} OTM2 PE EXIT current-exp')

    def on_new_day(self):
        # For both underlyings, check expiry and update positions
        for underlying in self.underlyings:
            self.underlying = underlying
            nearest_expiry = self.get_nearest_expiry(self.now, underlying)
            if nearest_expiry is None:
                self.dte[underlying] = datetime.timedelta(0)
                self.is_expiry_day[underlying] = 0
                continue
            self.dte[underlying] = nearest_expiry - self.now.date()
            if self.dte[underlying].days != 0:
                self.is_expiry_day[underlying] = 0
            else:
                self.is_expiry_day[underlying] = 1
                self.curr_pos[underlying], self.next_pos[underlying] = self.next_pos[underlying], {}

            # Set symbol and lot size for each underlying
            if underlying == 'BANKNIFTY':
                self.mysymbol[underlying] = 'BANKNIFTYSPOT'
            elif underlying == 'NIFTY':
                self.mysymbol[underlying] = 'NIFTYSPOT'
            elif underlying == 'FINNIFTY':
                self.mysymbol[underlying] = 'FINNIFTYSPOT'
            elif underlying == 'MIDCPNIFTY':
                self.mysymbol[underlying] = 'MIDCPNIFTYSPOT'
            elif underlying == 'SENSEX':
                self.mysymbol[underlying] = 'SENSEXSPOT'
            else:
                raise ValueError(f'Unknown Underlying: {underlying}')
            self.lot_size[underlying] = self.get_lot_size(underlying)

        # Use the session values from the first underlying (NIFTY)
        self.session_vals = sessions_dict[self.session]
        self.start_time = self.session_vals['start_time']
        self.stop_time = self.session_vals['stop_time']
        dtn = datetime.datetime.now()
        self.start_time = (datetime.datetime.combine(dtn.date(), self.start_time) + datetime.timedelta(minutes=self.delay)).time()

    def on_event(self):
        pass

    def on_bar_complete(self):
        # For both underlyings, handle expiry and cross-entry logic
        now_time = self.now.time()
        # 1. Exit positions expiring today at stop_time
        for underlying in self.underlyings:
            if self.is_expiry_day[underlying] and now_time >= self.stop_time and self.curr_pos[underlying]:
                self._exit_position(self.curr_pos[underlying], underlying)
                self.curr_pos[underlying] = {}

        # 2. Cross-entry logic: On NIFTY expiry, enter SENSEX next week; on SENSEX expiry, enter NIFTY next week
        if now_time == self.start_time:
            # On NIFTY expiry day, enter SENSEX next week
            if self.is_expiry_day['NIFTY'] and not self.next_pos['SENSEX']:
                self.next_pos['SENSEX'] = self._enter_next_week('SENSEX')
            # On SENSEX expiry day, enter NIFTY next week
            if self.is_expiry_day['SENSEX'] and not self.next_pos['NIFTY']:
                self.next_pos['NIFTY'] = self._enter_next_week('NIFTY')

