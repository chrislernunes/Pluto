import datetime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from utils.definitions import *
from utils.sessions import *
import math

if REDIS:
    from engine.ems import EventInterfacePositional, lot_size_dict, strike_diff_dict
else:
    from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

import datetime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from utils.definitions import *
from utils.sessions import *

class GOLDCC(EventInterfacePositional):

    def __init__(self):
        super().__init__()
        self.strat_id = self.__class__.__name__.lower()
        self.signal = 0
        self.last_expiry = None
        self.futprice = None
        self.old_p1 = None
        self.position_ce = 0
        
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

    def on_new_day(self):

        # Detect expiry
        self.dte = self.get_nearest_expiry(self.now, self.underlying) - self.now.date()
        self.is_expiry_day = self.dte.days == 0

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
        self.strike_step = strike_diff_dict[self.underlying]

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

        if self.now.time() < self.start_time or self.now.time() > self.stop_time:
            return
        
        # ENTRY

        if self.now.time() >= self.start_time and self.now.date().day >= 1 and self.now.date().day <= 10  and self.position_ce == 0:
            
            self.spot = self.get_tick(self.now, self.mysymbol)['c']
            target_strike = self.spot * self.selector_val/100
            print(target_strike, self.selector_val)
            idx = math.ceil(target_strike / self.strike_step)

            self.symbol_ce = self.find_symbol_by_moneyness(
                        self.now, self.underlying, 0, 'CE', idx
                    )
            
            if self.symbol_ce is not None:
                    # sell CE
                    self.success_ce, self.entry_price_ce = self.place_trade(self.now, 'SELL', self.lot_size, self.symbol_ce,note="ENTRY CE")
                    if self.success_ce:
                        self.position_ce = 1
                        self.expiry_ce = self.parse_date_from_symbol(self.symbol_ce)
            
        # EXIT  

        if self.position_ce == 1:
            # Exit on expiry day
            if self.now.date() >= self.expiry_ce and self.now.time() >= self.stop_time:
                self.success_exit_ce, self.exit_price_ce = self.place_trade(self.now, 'BUY', self.lot_size, self.symbol_ce,note="EXIT CE EXPIRY")
                if self.success_exit_ce:
                    self.position_ce = 0