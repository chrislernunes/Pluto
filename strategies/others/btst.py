"""
BTSTDIR Strategy (Buy Today, Sell Tomorrow — Directional Intraday Rollable Strategy)
------------------------------------------------------------------------------------

Overview:
---------
The BTSTDIR strategy is a positional options trading strategy designed to enter 
directional trades (CALL or PUT) based on ORB (Opening Range Breakout) signals and 
rollover logic. It supports entry/exit logic, optional stop loss trailing, 
and expiry rollover for both call and put legs. It is structured to operate 
within a specified session and exit at defined square-off times or upon trigger conditions.

Main Features:
--------------
1. **ORB-Based Entry**:
   - Entry is based on a breakout beyond a threshold (`breakout_factor * price[0]`) 
     from the spot price at start of the day.
   - Uses either premium-based (`selector='P'`) or moneyness-based (`selector='M'`) strike selection.

2. **Rollover Support**:
   - On expiry day (`DTE == 0`), supports two types of expiry behavior:
     - `'r'` (rollover): Switch to next expiry when conditions are met.
     - `'n'` (non-rollover): Enter directly in next expiry on the day of expiry.

3. **Stop Loss / Target / Trailing**:
   - Applies fixed stop loss and target (`sl_pct`, `tgt_pct`) after entry.
   - If trailing is enabled (`trail_on=True`), adjusts SL and target dynamically as price moves in favor.

4. **Session-Aware Execution**:
   - Uses defined `session` (e.g., `'x0'`) to control trading time range.
   - Entry only starts after `start_time + delay`, exits by `sq_off_time - delay_exit`.

5. **Trade State and Tracking**:
   - Tracks CE and PE positions independently (`position_ce`, `position_pe`).
   - Maintains entry/exit reasons for each leg.
   - Auto-reset logic per day to clear any residual positions.

6. **Robust UID Parsing**:
   - Supports reproducible backtesting through UID-based strategy encoding.
   - Every configuration can be uniquely represented and parsed using a UID string.

Edge Case Handling:
-------------------
- Safe checks for DTE mismatch, missing ticks, and invalid session keys.
- Includes logic to avoid multiple entries or exits on the same bar.
- Rollover only triggers once per expiry and resets entry state post-rollover.

Typical Use Case:
-----------------
- Backtest or run live intraday positional strategies with expiry awareness and 
  directional bias using NIFTY, BANKNIFTY, SENSEX, or MIDCPNIFTY underlyings.

"""

import datetime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from utils.definitions import *
from utils.sessions import *
import direct_redis, math

if REDIS:
    from engine.ems import EventInterfacePositional
else:
    from engine.ems_db import EventInterfacePositional

r = direct_redis.DirectRedis()


class BTSTDIR(EventInterfacePositional):
    
    def __init__(self):
        super().__init__()
        self.strat_id = self.__class__.__name__.lower()

        self.position_ce = 0
        self.position_pe = 0

        self.symbol_ce = None
        self.prices_ce = []

        self.symbol_pe = None
        self.prices_pe = []

        self.symbol_ce_hedge = None
        self.symbol_pe_hedge = None
        self.last_active_date = None

        self.sl_updated_ce = False
        self.sl_updated_pe = False

    def get_random_uid(self):
        # Select
        self.active_weekday = 99#np.random.choice(weekdays)
        self.session = np.random.choice(['x0'])
        self.timeframe = 1 #np.random.choice(timeframes)
        self.underlying = np.random.choice(['MIDCPNIFTY']) # 'NIFTY', 'FINNIFTY', 'BANKNIFTY', 'SENSEX', 
        self.selector = 'P' # np.random.choice(selectors)
        if self.selector == 'M':
            self.selector_val = np.random.choice(moneynesses)
        elif self.selector == 'P':
            self.selector_val = np.random.choice(range(5, 20, 5)) # np.random.choice([15, 25, 50, 75, 100])
        # self.hedge_shift = np.random.choice(hedge_shifts)

        self.sl_pct = round(np.random.choice(np.arange(0.3, 0.5, 0.05)), 2) #round(.05 * round(np.random.choice(np.random.rand(10)*0.5).round(2)/.05), 2)
        self.tgt_pct = round(np.random.choice(np.arange(0.6, 0.9, 0.05)), 2) #round(.05 * round(np.random.choice(np.random.random(1)).round(2)/.05), 2) #np.random.choice(tgt_pcts)
        self.max_reset = np.random.choice([0,1])
        self.trail_on = np.random.choice([True])
        self.delay = np.random.choice(range(0, 120, 30))
        # ...
        if self.session in ['x0', 'x1', 'x2', 'y0', 't1']:
            orb_sizes = [15, 30, 45, 60, 75, 90]
        else:
            orb_sizes = [5, 10, 15, 20, 25, 30]
        self.orb_size = np.random.choice(orb_sizes)
        self.breakout_factor = round(np.random.choice(np.arange(1.0, 1.5, 0.05)), 2)
        
        self.ohlc = np.random.choice(['o', 'c'])
        self.delay_exit = np.random.choice(range(0, 10, 1))
        self.strat_type = np.random.choice(['r', 'n']) # r - Roll over at EOD , n - directly enter next expiry
        self.trail_pct = np.random.choice([0.05, 0.025, 0.01])
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
        self.selector_val = int(s.pop(0))
        # self.hedge_shift = int(s.pop(0))
        self.sl_pct = float(s.pop(0))
        self.tgt_pct = float(s.pop(0))
        self.max_reset = int(s.pop(0))
        self.trail_on = s.pop(0)=='True'
        # ...
        self.orb_size = int(s.pop(0))
        self.breakout_factor = float(s.pop(0))
        self.ohlc = s.pop(0)
        self.delay_exit = int(s.pop(0))
        self.strat_type = s.pop(0)
        self.trail_pct = float(s.pop(0))
        self.roll_or_no=s.pop(0)
#  'btstdir_99_o3_0_1_NIFTY_M_0_9990.5_1.99_0_False_90_0.003_c_12_r_0.4_False',

        # CROSS CHECK
        assert len(s)==0
        self.gen_uid = self.get_uid_from_params()
        assert uid == self.gen_uid
        self.uid = uid
        print(self.uid)
    
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
        {self.sl_pct}_
        {self.tgt_pct}_
        {self.max_reset}_
        {self.trail_on}_
        {self.orb_size}_
        {self.breakout_factor}_
        {self.ohlc}_
        {self.delay_exit}_
        {self.strat_type}_
        {self.trail_pct}_
        {self.roll_or_no}
        """.replace('\n', '').replace(' ', '').strip('_')

    def on_new_day(self):
        # ...
        self.lot_size = self.get_lot_size(self.underlying)
        # ...
        self.session_vals = sessions_dict[self.session]
        self.start_time = self.session_vals['start_time']#datetime.time(9, 15)
        self.stop_time = self.session_vals['stop_time']#datetime.time (11, 15)
        self.sq_off_time = self.session_vals['sq_off_time']#datetime.time (11, 15)

        #...
        self.find_dte =True   
        self.roll_over = False
        self.spot_price = []  
        self.reason_ce = None 
        self.reason_pe = None 

        if self.position_ce == -1:    self.position_ce = 0
        if self.position_pe == -1:    self.position_pe = 0
        self.sl_updated=False
        self.sl_updated_pe=False

        # ADD DELAY TO START TIME
        dtn = datetime.datetime.now()

        #---->
        self.start_time = (datetime.datetime.combine(dtn.date(), self.start_time) + datetime.timedelta(minutes=self.delay)).time()
        self.sq_off_time = (datetime.datetime.combine(dtn.date(), self.sq_off_time) - datetime.timedelta(minutes=self.delay_exit)).time()
        if self.strat_type=='r':
            if self.ohlc=='o':
                if self.roll_or_no=='True' or self.roll_or_no==True:
                    self.haste = 3
                elif self.roll_or_no=='False' or self.roll_or_no==False:
                    self.haste = 2
            if self.ohlc=='c':
                if self.roll_or_no=='True' or self.roll_or_no==True:
                    self.haste = 4
                elif self.roll_or_no=='False' or self.roll_or_no==False:
                    self.haste = 1

        self.roll_over_time = datetime.time(15, 1)

        
    def on_event(self):
        pass


    def on_bar_complete(self):
        # self.bool_pe=False
        # self.bool=False


        # time.sleep(0.1)
        # # ...
        # if self.find_dte:
        #     atm_ce = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'CE', 0)
        #     self.dte = self.get_dte(self.now, atm_ce)
        #     if atm_ce:
        #         self.find_dte = False

        if self.find_dte:
            self.dte = self.get_dte_by_underlying(self.now, self.underlying)
            self.find_dte = False
            
        # # ...
        if self.dte != self.active_weekday and self.active_weekday != 99:
            return
        
        expiry_idx = 0
        # ROLLOVER LOGIC BASED ON STRAT TYPE
        if self.strat_type == 'r' and self.roll_or_no==True: 
            if self.dte == 0 and self.position_ce==0 and self.position_pe==0 and self.now.time() >= (datetime.datetime.combine(datetime.datetime.now().date(), self.roll_over_time)+ datetime.timedelta(minutes=self.haste) ).time():
                return
            elif self.dte == 0 and self.now.time() == (datetime.datetime.combine(datetime.datetime.now().date(), self.roll_over_time) + datetime.timedelta(minutes=self.haste)).time():
                self.roll_over = True
                expiry_idx = 1
            elif self.dte == 0 and self.now.time() > (datetime.datetime.combine(datetime.datetime.now().date(), self.roll_over_time)+ datetime.timedelta(minutes=self.haste) ).time():
                return
            else:
                self.roll_over = False
                expiry_idx = 0
                
        elif self.strat_type == 'r' and self.roll_or_no==False: 
            if self.dte == 0 and self.now.time() > (datetime.datetime.combine(datetime.datetime.now().date(), self.roll_over_time)+ datetime.timedelta(minutes=self.haste) ).time():
                return
            elif self.dte == 0 and self.position_ce==0 and self.position_pe==0 and self.now.time() == (datetime.datetime.combine(datetime.datetime.now().date(), self.roll_over_time)+ datetime.timedelta(minutes=self.haste) ).time():
                return
            elif self.dte == 0 and self.now.time() == (datetime.datetime.combine(datetime.datetime.now().date(), self.roll_over_time)+ datetime.timedelta(minutes=self.haste) ).time():
                self.roll_over = True
            else:
                expiry_idx = 0

        elif self.strat_type == 'n':
            if self.dte == 0:
                expiry_idx = 1
            else:
                expiry_idx = 0
        
        # EXIT   
        if self.position_ce == 1:
            self.current_price_ce = float(self.get_tick(self.now, self.symbol_ce)['c'])
            # print(f'CE {self.now} {self.entry_price_ce} {self.current_price_ce} {self.tgt_price_ce} {self.sl_price_ce}')
            self.to_exit = False
            if self.trail_on:
                if self.current_price_ce > self.tgt_price_ce :
                    self.new_tgt_ce = self.current_price_ce * (1+self.trail_pct)
                    self.tgt_price_ce = self.new_tgt_ce
                    self.new_sl_price_ce = self.tgt_price_ce * (1-self.trail_pct)
                    self.sl_updated_ce = True 
                    self.sl_price_ce = self.new_sl_price_ce
                    
   
            # SL
            if self.now.time() >= datetime.time(9,25) and self.now.time() <= datetime.time(15, 29):
                if self.current_price_ce <= self.sl_price_ce:
                        self.to_exit = True
                        if self.sl_updated_ce:
                            self.reason_ce='SL TRAILED'
                        else:
                            self.reason_ce = 'SL'
            
            # ROLL-OVER
            if self.roll_over:
                self.to_exit = True
                self.reason_ce = 'ROLL-OVER'

            # TIME
            if self.now.time() == self.sq_off_time:
                self.to_exit = True
                self.reason_ce = 'TIME'

            if self.to_exit:
                self.success_ce, _= self.place_trade(self.now, 'SELL', self.lot_size, self.symbol_ce, note=self.reason_ce, signal_number=self.last_active_date)
                if self.success_ce:
                    self.position_ce = -1
                    self.sl_updated_ce = False

        if self.position_pe == 1:
            self.current_price_pe = float(self.get_tick(self.now, self.symbol_pe)['c'])
            # print(f'PE {self.now} {self.entry_price_pe} {self.current_price_pe} {self.tgt_price_pe} {self.sl_price_pe}')
            self.to_exit = False 
            if self.trail_on:
                if self.current_price_pe > self.tgt_price_pe :
                    self.new_tgt_pe = self.current_price_pe * (1+self.trail_pct) 
                    self.tgt_price_pe = self.new_tgt_pe
                    self.new_sl_price_pe = self.tgt_price_pe * (1-self.trail_pct)
                    self.sl_updated_pe = True    
                    self.sl_price_pe = self.new_sl_price_pe

            # SL
            if self.now.time() >= datetime.time(9,25) and self.now.time() <= datetime.time(15, 29):

                if self.current_price_pe <= self.sl_price_pe:
                        self.to_exit = True
                        if self.sl_updated_pe:
                            self.reason_pe='SL TRAILED'
                        else:
                            self.reason_pe = 'SL'
            
            # ROLL-OVER
            if self.roll_over:
                self.to_exit = True
                self.reason_pe = 'ROLL-OVER'
            
            # if self.bool_pe==True:
            #     self.reason_pe='TRAILED'
            #     self.bool_pe=False
            # TGT
            # if self.current_price_pe >= self.tgt_price_pe and self.Target:
            #     self.to_exit = True
            #     self.reason_pe = 'TGT'
            # TIME
            if self.now.time() == self.sq_off_time:
                self.to_exit = True
                self.reason_pe = 'TIME'
            if self.to_exit:
                self.success_pe, _= self.place_trade(self.now, 'SELL', self.lot_size, self.symbol_pe, note=self.reason_pe, signal_number=self.last_active_date)
                if self.success_pe:
                    self.position_pe = -1
                    self.sl_updated_pe = False
        
        if self.now.time() < self.start_time or self.now.time() > self.stop_time:
            return
        
        # RESET POSITION IF NEW DAY
        if self.last_active_date != str(self.now.date()):
            if self.position_ce == -1:
                self.position_ce = 0
            if self.position_pe == -1:
                self.position_pe = 0

        self.current_tick = self.get_tick(self.now, f'{self.underlying}SPOT')
        if len(self.spot_price) == 0:
            self.spot_price.append(float(self.current_tick[self.ohlc]))
        else:  
            self.spot_price.append(float(self.current_tick['c']))
            
        if self.roll_or_no==True or self.roll_or_no=='True':
            # ROLL-OVER ENTRY
            if self.reason_ce == 'ROLL-OVER' and self.position_ce == -1:
                self.position_ce = 0
            if self.reason_pe == 'ROLL-OVER' and self.position_pe == -1:
                self.position_pe = 0
        else:
            pass
        #     if self.reason_ce == 'ROLL-OVER' and self.position_ce == 0:
        #         return
        #     if self.reason_pe == 'ROLL-OVER' and self.position_pe == 0:
        #         return

        # ENTRY
        if self.now.time() >= self.start_time and self.now.time() < self.stop_time and len(self.spot_price) > self.orb_size:
            if (self.position_ce == 0 and self.position_pe == 0):
                if ((self.spot_price[-1] - self.spot_price[0]) > self.spot_price[0] * self.breakout_factor) or self.reason_ce == 'ROLL-OVER':
                    print('CALL ENTRY')
                    if self.selector == 'P':
                        self.symbol_ce = self.find_symbol_by_premium(
                            self.now, self.underlying, expiry_idx, 'CE', self.selector_val, 
                        )
                    elif self.selector == 'M':
                        self.symbol_ce = self.find_symbol_by_moneyness(
                            self.now, self.underlying, expiry_idx, 'CE', self.selector_val
                        )
                    if self.symbol_ce is not None:                     
                        self.success_ce, self.entry_price_ce = self.place_trade(self.now, 'BUY', self.lot_size, self.symbol_ce, note='CE ENTRY', signal_number=str(self.now.date()))
                        if self.success_ce:
                            self.sl_price_ce = float(self.entry_price_ce)*(1-self.sl_pct)
                            self.tgt_price_ce = float(self.entry_price_ce)*(1+self.tgt_pct)
                            self.position_ce = 1
                            self.last_active_date = str(self.now.date())
                         
                if ((self.spot_price[-1] - self.spot_price[0]) < -1 * self.spot_price[0] * self.breakout_factor) or self.reason_pe == 'ROLL-OVER': 
                    print('PUT ENTRY')
                    if self.selector == 'P':
                        self.symbol_pe = self.find_symbol_by_premium(
                            self.now, self.underlying, expiry_idx, 'PE', self.selector_val, 
                        )
                    elif self.selector == 'M':
                        self.symbol_pe = self.find_symbol_by_moneyness(
                            self.now, self.underlying, expiry_idx, 'PE', self.selector_val
                        )
                    if self.symbol_pe is not None:  
                        self.success_pe, self.entry_price_pe = self.place_trade(self.now, 'BUY', self.lot_size, self.symbol_pe, note='PE ENTRY', signal_number=str(self.now.date()))
                        if self.success_pe:
                            self.sl_price_pe = float(self.entry_price_pe)*(1-self.sl_pct)
                            self.tgt_price_pe = float(self.entry_price_pe)*(1+self.tgt_pct)
                            self.position_pe = 1
                            self.last_active_date = str(self.now.date())
    
        # r.hset(f'mtm_{self.uid}', str(self.now), self.get_mtm(self.now))