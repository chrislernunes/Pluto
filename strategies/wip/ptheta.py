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

class PTHETA(EventInterfacePositional):

    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()
        self.symbol_ce = None
        self.symbol_pe = None
        self.position_ce = 0  # 0: no position, 1: open, -1: exited
        self.position_pe = 0
        self.reset_count_ce = 0
        self.reset_count_pe = 0
        self.entry_price_ce = None
        self.entry_price_pe = None
        self.sl_price_ce = None
        self.sl_price_pe = None
        self.success_ce = False
        self.success_pe = False
        self.reason_ce = None
        self.reason_pe = None
        self.main_exit_happened = False  # Track if main exit has occurred

    def get_random_uid(self):
        # Select
        self.active_weekday = np.random.choice([0])
        self.session = 'x0'#np.random.choice(sessions)
        self.dte_entry = np.random.choice([4])
        self.underlying = np.random.choice(['NIFTY', 'SENSEX'])
        selectors = ['P']
        self.selector = np.random.choice(selectors)
        self.sl_pct = min(0.8, max(0.3, round(.05 * round(np.random.choice(np.random.random(1)).round(2)/.05), 2))) # np.random.rand(10)*0.5
        self.tgt_pct = max(0.5, round(.05 * round(np.random.choice(np.random.random(1)).round(2)/.05), 2))
        self.max_reset = np.random.choice([0, 1, 2, 3, 4])
        self.trail_on = np.random.choice([False])
        self.Target = np.random.choice([True, False])
        self.delay = np.random.choice(range(0, 60, 10))
        self.ps = np.random.choice([0.005, 0.007, 0.01, 0.015, 0.02])

        # ...
        return self.get_uid_from_params()

    def set_params_from_uid(self, uid):
        s = uid.split('_')
        try:
            assert s[0] == self.strat_id
        except AssertionError:
            raise ValueError(f'Invalid UID {uid} for strat ID {self.strat_id}')
        s = s[1:]
        self.active_weekday = int(s.pop(0))
        self.session = s.pop(0)
        self.delay = int(s.pop(0))#=='True'
        self.dte_entry = int(s.pop(0))
        self.underlying = s.pop(0)
        self.selector = s.pop(0)
        self.sl_pct = float(s.pop(0))
        self.tgt_pct = float(s.pop(0))
        self.max_reset = int(s.pop(0))
        self.trail_on = s.pop(0)=='True'
        self.Target = s.pop(0)=='True'
        self.ps = float(s.pop(0))
        # CROSS CHECK
        assert len(s)==0
        self.gen_uid = self.get_uid_from_params()
        assert uid == self.gen_uid
        self.uid = uid
    
    def get_uid_from_params(self):
        return f"""
        {self.strat_id}_
        {self.active_weekday}_
        {self.session}_
        {self.delay}_
        {self.dte_entry}_
        {self.underlying}_
        {self.selector}_
        {self.sl_pct}_
        {self.tgt_pct}_
        {self.max_reset}_
        {self.trail_on}_
        {self.Target}_
        {self.ps}_
        
        """.replace('\n', '').replace(' ', '').strip('_')   

    def calculate_margin(self):
        spot_price = float(self.get_tick(self.now, f'{self.underlying}SPOT')['c'])
        if self.underlying.startswith('NIFTY'):
            self.margin=(spot_price*self.lot_size*2)/14
        else:
            self.margin=(spot_price*self.lot_size*2)/13
    
    def calculate_mn_premium(self):
        self.mn_premium=self.margin*(self.ps)/self.lot_size

    def get_trading_days_to_expiry(self, current_date, expiry_date):
        """Calculate trading days between current date and expiry date (excluding weekends)"""
        trading_days = 0
        current = current_date
        while current < expiry_date:
            if current.weekday() < 5:  # Monday = 0, Friday = 4, Saturday = 5, Sunday = 6
                trading_days += 1
            current += datetime.timedelta(days=1)
        return trading_days

    def on_new_day(self):
        # expiry_date = self.get_nearest_expiry(self.now, self.underlying)
        # self.dte = self.get_trading_days_to_expiry(self.now.date(), expiry_date)
        self.dte = self.get_dte_by_underlying(self.now, self.underlying)
        self.is_expiry_day = 1 if self.dte == 0 else 0
        self.lot_size = self.get_lot_size(self.underlying)
        self.session_vals = sessions_dict[self.session]
        self.start_time = self.session_vals['start_time']
        self.stop_time = self.session_vals['stop_time']
        dtn = datetime.datetime.now()
        self.start_time = (datetime.datetime.combine(dtn.date(), self.start_time) + datetime.timedelta(minutes=self.delay)).time()
        self.stop_time = (datetime.datetime.combine(dtn.date(), self.stop_time) - datetime.timedelta(minutes=14)).time()
        self.calculate_margin()
        self.calculate_mn_premium()
        # Reset re-entry counts and positions after expiry exit
        if self.dte == self.dte_entry:
            self.reset_count_ce = 0
            self.reset_count_pe = 0
            self.position_ce = 0
            self.position_pe = 0
            self.reason_ce = None
            self.reason_pe = None
            self.symbol_ce = None
            self.symbol_pe = None
            self.entry_price_ce = None
            self.entry_price_pe = None
            self.sl_price_ce = None
            self.sl_price_pe = None
            self.main_exit_happened = False  # Reset main exit flag for new cycle

    def try_main_entry(self):
        # Only enter if both legs are flat and it's the right day/time
        if (
            self.position_ce == 0 and self.position_pe == 0 and
            self.dte == self.dte_entry and
            self.now.time() >= self.start_time and self.now.time() < self.stop_time
        ):
            self.symbol_ce = self.find_symbol_by_premium(self.now, self.underlying, 0, 'CE', self.mn_premium)
            # self.symbol_ce,_ = self.find_symbol_by_itm_percent_v2(self.now, self.underlying, 0, 'CE', 0.3)
            self.symbol_pe = self.find_symbol_by_premium(self.now, self.underlying, 0, 'PE', self.mn_premium)
            # self.symbol_pe,_ = self.find_symbol_by_itm_percent_v2(self.now, self.underlying, 0, 'PE', 0.3)
            if self.symbol_ce:
                self.success_ce, self.entry_price_ce = self.place_trade(self.now, 'SELL', self.lot_size, self.symbol_ce, note='CE ENTRY')
                if self.success_ce:
                    self.position_ce = 1
                    self.sl_price_ce = self.entry_price_ce * (1 + self.sl_pct)
            if self.symbol_pe:
                self.success_pe, self.entry_price_pe = self.place_trade(self.now, 'SELL', self.lot_size, self.symbol_pe, note='PE ENTRY')
                if self.success_pe:
                    self.position_pe = 1
                    self.sl_price_pe = self.entry_price_pe * (1 + self.sl_pct)

    def try_main_exit(self):
        # Exit both legs at stop_time on expiry day
        if self.is_expiry_day and self.now.time() >= self.stop_time:
            if self.position_ce == 1:
                self.success_ce, _ = self.place_trade(self.now, 'BUY', self.lot_size, self.symbol_ce, note='CE EXIT TIME')
                if self.success_ce:
                    self.position_ce = 0
                    self.reset_count_ce = 0
                    self.reason_ce = 'TIME'
            if self.position_pe == 1:
                self.success_pe, _ = self.place_trade(self.now, 'BUY', self.lot_size, self.symbol_pe, note='PE EXIT TIME')
                if self.success_pe:
                    self.position_pe = 0
                    self.reset_count_pe = 0
                    self.reason_pe = 'TIME'
            # Set main exit flag when we reach exit time, regardless of whether positions were exited
            self.main_exit_happened = True

    def check_stoploss_and_reentry(self):
        # CE SL and re-entry
        if self.position_ce == 1:
            curr_price_ce = float(self.get_tick(self.now, self.symbol_ce)['c'])
            if curr_price_ce >= self.sl_price_ce:
                self.success_ce, _ = self.place_trade(self.now, 'BUY', self.lot_size, self.symbol_ce, note='CE SL EXIT')
                if self.success_ce:
                    self.position_ce = -1
                    self.reason_ce = 'SL'
        if self.position_ce == -1 and self.reset_count_ce < self.max_reset and self.reason_ce == 'SL':
            # Only allow re-entry if main exit hasn't happened yet
            if not self.main_exit_happened:
                curr_price_ce = float(self.get_tick(self.now, self.symbol_ce)['c'])
                if curr_price_ce <= self.entry_price_ce:
                    self.success_ce, self.entry_price_ce = self.place_trade(self.now, 'SELL', self.lot_size, self.symbol_ce, note=f'CE RE-ENTRY {self.reset_count_ce+1}')
                    if self.success_ce:
                        self.position_ce = 1
                        self.sl_price_ce = self.entry_price_ce * (1 + self.sl_pct)
                        self.reset_count_ce += 1
                        self.reason_ce = None
        # PE SL and re-entry
        if self.position_pe == 1:
            curr_price_pe = float(self.get_tick(self.now, self.symbol_pe)['c'])
            if curr_price_pe >= self.sl_price_pe:
                self.success_pe, _ = self.place_trade(self.now, 'BUY', self.lot_size, self.symbol_pe, note='PE SL EXIT')
                if self.success_pe:
                    self.position_pe = -1
                    self.reason_pe = 'SL'
        if self.position_pe == -1 and self.reset_count_pe < self.max_reset and self.reason_pe == 'SL':
            # Only allow re-entry if main exit hasn't happened yet
            if not self.main_exit_happened:
                curr_price_pe = float(self.get_tick(self.now, self.symbol_pe)['c'])
                if curr_price_pe <= self.entry_price_pe:
                    self.success_pe, self.entry_price_pe = self.place_trade(self.now, 'SELL', self.lot_size, self.symbol_pe, note=f'PE RE-ENTRY {self.reset_count_pe+1}')
                    if self.success_pe:
                        self.position_pe = 1
                        self.sl_price_pe = self.entry_price_pe * (1 + self.sl_pct)
                        self.reset_count_pe += 1
                        self.reason_pe = None

    def on_bar_complete(self):
        self.try_main_entry()
        self.check_stoploss_and_reentry()
        self.try_main_exit()         