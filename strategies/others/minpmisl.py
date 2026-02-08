import datetime, time
import pandas as pd
import numpy as np

from utils.definitions import *
from utils.sessions import *
from utils.utility import *

if REDIS:
    from engine.ems import EventInterface
else:
    from engine.ems_db import EventInterface
    
import direct_redis

r = direct_redis.DirectRedis()

# mn premium individual stop loss
class MINPMISL(EventInterface):
    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()

    def get_random_uid(self):
        # Select
        self.active_weekday = np.random.choice([0])
        self.session = 'x0'#np.random.choice(sessions)
        self.timeframe = np.random.choice([1])
        self.underlying = np.random.choice(['NIFTY', 'SENSEX'])
        selectors = ['P']
        self.selector = np.random.choice(selectors)
        if self.selector == 'PCT':
            values = [0.0025, 0.0015, 0.00075, 0.0002, 0.00015, 0.0005, 0.0003, 0.00045] 
            self.selector_val = np.random.choice(values)
        elif self.selector == 'P':
            self.selector_val = np.random.choice([75])
        elif self.selector == 'R':
            self.selector_val = round(np.random.choice(np.arange(0.3,02.0, 0.1)),2)
        elif self.selector == 'M':
            self.selector_val = np.random.choice([0, 1, 2])

        self.hedge_shift = np.random.choice([99])
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
        self.timeframe = int(s.pop(0))
        self.underlying = s.pop(0)
        self.selector = s.pop(0)
        if self.selector == 'P':
            self.selector_val = int(s.pop(0))
        elif self.selector in ['PCT', 'R']:
            self.selector_val = float(s.pop(0))
        self.hedge_shift = int(s.pop(0))
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
        # print(self.uid)
    
    def get_uid_from_params(self):
        return f"""
        {self.strat_id}_
        {self.active_weekday}_
        {self.session}_
        {self.delay}_
        {self.timeframe}_
        {self.underlying}_
        {self.selector}_
        {self.selector_val}_
        {self.hedge_shift}_
        {self.sl_pct}_
        {self.tgt_pct}_
        {self.max_reset}_
        {self.trail_on}_
        {self.Target}_
        {self.ps}_
        
        """.replace('\n', '').replace(' ', '').strip('_')
    
    
    def on_new_day(self):
        if self.underlying == 'BANKNIFTY':
            self.mysymbol = "BANKNIFTYSPOT" 
        elif self.underlying == 'NIFTY':
            self.mysymbol = "NIFTYSPOT" 
        elif self.underlying == 'FINNIFTY':
            self.mysymbol = "FINNIFTYSPOT" 
        elif self.underlying == 'MIDCPNIFTY':
            self.mysymbol = "MIDCPNIFTYSPOT" 
        elif self.underlying == 'SENSEX':
            self.mysymbol = "SENSEXSPOT"

        # ...
        self.lot_size = self.get_lot_size(self.underlying)
        # ...
        self.symbol_ce = None
        self.symbol_pe = None
        # ...
        self.position_ce = 0
        self.position_pe = 0
        # ...
        self.reset_count_ce = 0
        self.reset_count_pe = 0

        self.premium_for_day_selection = True
        self.i=0
        # ...
        self.session_vals = sessions_dict[self.session]
        self.start_time = self.session_vals['start_time']#datetime.time(9, 15)
        self.stop_time = self.session_vals['stop_time']#datetime.time (11, 15)
        # ADD DELAY TO START TIME
        dtn = datetime.datetime.now()
        
        self.start_time = (datetime.datetime.combine(dtn.date(), self.start_time) + datetime.timedelta(minutes=self.delay)).time()
        self.stop_time = (datetime.datetime.combine(dtn.date(), self.stop_time) - datetime.timedelta(minutes=self.timeframe)).time()

        # self.start_time = (datetime.datetime.combine(dtn.date(), self.start_time) + datetime.timedelta(minutes=(self.delay))).time()
        self.find_dte = True
              
    def calculate_margin(self):
        spot_price = float(self.get_tick(self.now, f'{self.underlying}SPOT')['c'])
        if self.underlying.startswith('NIFTY'):
            self.margin=(spot_price*self.lot_size*2)/13
        else:   
            self.margin=(spot_price*self.lot_size*2)/13
    
    def calculate_mn_premium(self):
        self.mn_premium=self.margin*(self.ps)/self.lot_size

    def on_bar_complete(self):  
        
        if self.find_dte:
            # atm_ce = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'CE', 0)
            # self.dte = self.get_dte(self.now, atm_ce)
            self.dte = self.get_dte_by_underlying(self.now, self.underlying)
            self.find_dte = False
        # ...
        if self.dte != self.active_weekday and self.active_weekday != 99:
            return
        
        if self.now.time() < self.start_time or self.now.time() > self.stop_time:
            return

        if self.i==0:
            self.calculate_margin()
            self.calculate_mn_premium()
            self.i=1
            if self.selector == 'P':
                self.symbol_ce = self.find_symbol_by_premium(
                    self.now, self.underlying, 0, 'CE',  self.mn_premium
                )
                self.symbol_pe = self.find_symbol_by_premium(
                    self.now, self.underlying, 0, 'PE',  self.mn_premium
                )

        if self.position_ce == 0 and self.now.time() < datetime.time(15, 0):
            if self.symbol_ce:
                self.symbol_ce_hedge = self.find_symbol_by_premium(self.now, self.underlying, 0, "CE", 2, perform_rms_checks=False)
                self.success_ce, self.entry_price_ce, self.entry_price_ce_hedge = self.place_spread_trade(
                        self.now, 'SELL', self.lot_size, self.symbol_ce, self.symbol_ce_hedge, note='ENTRY'
                    )
                if self.success_ce:
                    self.position_ce = 1
                    self.sl_price_ce = self.entry_price_ce * (1 + self.sl_pct)

        if self.position_pe == 0 and self.now.time() < datetime.time(15, 0):
            if self.symbol_pe:
                self.symbol_pe_hedge = self.find_symbol_by_premium(self.now, self.underlying, 0, "PE", 2, perform_rms_checks=False)   
                self.success_pe, self.entry_price_pe, self.entry_price_pe_hedge = self.place_spread_trade(
                        self.now, 'SELL', self.lot_size, self.symbol_pe, self.symbol_pe_hedge, note='ENTRY'
                    )
                if self.success_pe:
                    self.position_pe = 1
                    self.sl_price_pe = self.entry_price_pe * (1 + self.sl_pct)
        
        # EXIT
        if self.position_ce == 1 :
            self.current_price_ce = float(self.get_tick(self.now, self.symbol_ce)['c'])
            self.to_exit = False

            # SL
            if self.current_price_ce  >= self.sl_price_ce:
                self.to_exit = True
                self.reason_ce = 'SL'
            # TIME
            if self.now.time() >= self.stop_time:
                self.to_exit = True
                self.reason_ce = 'TIME'

            if self.to_exit:
                self.success_ce, _, _ = self.place_spread_trade(
                    self.now, 'BUY', self.lot_size, self.symbol_ce, self.symbol_ce_hedge, note=self.reason_ce
                )
                if self.success_ce :
                    self.position_ce = -1

        if self.position_pe == 1 :
            self.current_price_pe = float(self.get_tick(self.now, self.symbol_pe)['c'])
            self.to_exit = False

            # SL[]
            if self.current_price_pe >= self.sl_price_pe:
                self.to_exit = True
                self.reason_pe = 'SL'
            # TIME
            if self.now.time() >= self.stop_time:
                self.to_exit = True
                self.reason_pe = 'TIME'

            if self.to_exit:
                self.success_pe, _, _ = self.place_spread_trade(
                    self.now, 'BUY', self.lot_size, self.symbol_pe, self.symbol_pe_hedge, note=self.reason_pe
                )
                if self.success_pe :
                    self.position_pe = -1

        # RE-ENTRY
        if self.position_ce == -1 and self.reset_count_ce < self.max_reset and self.now.time() < datetime.time(15, 0):
            self.curr_tick_ce = float(self.get_tick(self.now, self.symbol_ce)['c'])
            if self.curr_tick_ce  <= self.entry_price_ce :
                self.position_ce = 0
                self.reset_count_ce += 1
        
        if self.position_pe == -1 and self.reset_count_pe < self.max_reset and self.now.time() < datetime.time(15, 0):
            self.curr_tick_pe = float(self.get_tick(self.now, self.symbol_pe)['c'])
            if self.curr_tick_pe  <= self.entry_price_pe :
                self.position_pe = 0
                self.reset_count_pe += 1