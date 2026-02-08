"""
Strategy Name: SPIKESELL 

Overview:
---------
SPIKESELL is a momentum-based intraday options selling strategy that triggers short entries on CE and PE options
when a sharp price spike is followed by a defined retracement. It avoids re-entries and is designed to strictly
manage risk using configurable stop-loss, target, trailing SL, and mark-to-market (MTM) based exits. Optionally,
the strategy includes profit-locking behavior to tighten risk exposure after a certain profit is achieved.

Key Features:
-------------
1. **Spike-Based Entry Logic**:
   - Monitors price over a rolling `lookback` window.
   - A spike is detected when the current price exceeds `lookback` price by `spike_thresh` percent.
   - Entry is triggered when price subsequently retraces by at least `retrace_thresh` percent.

2. **Time-Controlled Execution**:
   - Strategy operates within a session window (`session`, `delay`, `haste`) and runs at intervals defined by `timeframe` and `offset`.

3. **Strike Selection (`selector`)**:
   - 'M': Moneyness-based selection (ATM ± offset).
   - 'P', 'PCT', and 'R' are supported in logic but not used in this default configuration.

4. **Risk Management**:
   - Stop-loss (`sl_type`, `sl_val`) is either a percentage or absolute value.
   - Targets are defined via `tgt_pct`.
   - Optional trailing stop-loss (`trail_on`).

5. **MTM and Profit Locking**:
   - Uses `max_loss` as MTM floor for forced exit.
   - If `lock_profit` is enabled:
     - Locks in profits once `lock_profit_trigger` is hit.
     - Adjusts `max_loss` dynamically as MTM increases using `subsequent_profit_step` and `subsequent_profit_amt`.

6. **No Reentry**:
   - Once a position is closed (SL/TGT/MTM/TIME), it is not reopened. Clean and deterministic logic without loops.

Usage:
------
- Parameters can be randomly generated using `get_random_uid()` or set from a UID string using `set_params_from_uid(uid)`.
- To execute, integrate with the EMS backtest/simulation framework providing ticks, symbol finding, and trade simulation.

Example UID:
------------
spikesell_NIFTY_0_x0_0_0_1_0_M_1_5_15_0.2_0.05_PCT_0.5_True_0.75_True_-2000_500_200_100_50

Breakdown:
----------
- Underlying: NIFTY
- DTE: 0 (same-day)
- Session: x0, Delay: 0, Haste: 0
- Timeframe: 1-min, Offset: 0
- Selector: Moneyness with value +1 strike from ATM
- Hedge shift: 5 strikes OTM
- Lookback: 15 bars, Spike Thresh: +20%, Retrace Thresh: -5%
- SL Type: PCT, SL Val: +50%
- Trailing SL: True
- Target: 25%
- Profit Locking: Enabled
  - Max Loss: ₹-2000
  - Trigger Lock at ₹500, lock to ₹200
  - Add ₹50 to max loss every ₹100 gain thereafter

Notes:
------
- Designed for expiry or near-expiry days to capture option volatility.
- Entry and exit actions are mutually exclusive and strategy is well-suited for backtesting.
"""

import datetime
import pandas as pd
import numpy as np
from utils.definitions import *
from utils.sessions import *
from utils.utility import *

if REDIS:
    from engine.ems import EventInterface
else:
    from engine.ems_db import EventInterface


#################

class SPIKESELL(EventInterface):
    
    def __init__(self):
        super().__init__()
        self.strat_id = self.__class__.__name__.lower()

    def get_random_uid(self):
        # Select
        self.underlying = np.random.choice(['SENSEX', 'NIFTY'])
        self.active_dte = np.random.choice([0, 1, 2, 3, 4])
        # dte_options = dte_dict[self.underlying]
        # self.active_dte = np.random.choice(dte_options)
        self.session = np.random.choice(['x0'])
        self.delay = np.random.choice([0])
        self.haste = np.random.choice([0])
        self.timeframe = np.random.choice([1])
        if self.timeframe > 1:
            self.offset = np.random.choice(range(0, self.timeframe))
        else:
            self.offset = 0
        # STRIKE SELECTION
        self.selector = np.random.choice(['M'])
        if self.selector == 'M':
            self.selector_val = int(np.random.choice(np.arange(-2,5,1)))
        elif self.selector == 'P':
            self.selector_val = int(np.random.choice(np.arange(5,120,5)))
        elif self.selector == 'R':
            self.selector_val = round(np.random.choice(np.arange(0.4,1.5,0.1)),4)
        elif self.selector == 'PCT':
            self.selector_val = round(np.random.choice(np.arange(0.0005,0.004,0.0005)),4)
        self.hedge_shift = np.random.choice([3,5,7,10])
        # ...
        self.lookback = np.random.choice([5,10,15,30,45,60,120,180,240])
        self.spike_thresh = np.random.choice([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.75,0.9,1.0,1.5,2.0])
        self.retrace_thresh = np.random.choice([0,0.01,0.02,0.03,0.04,0.05,0.1,0.15,0.2])
        # SL
        self.sl_type = np.random.choice(['PCT', 'ABS'])# PCT, ABS 
        if self.sl_type == 'PCT':
            self.sl_val = np.random.choice([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.25,2.5,2.75,3.0])
        elif self.sl_type == 'ABS':
            self.sl_val = np.random.choice(range(5,251,50))
        # TGT
        self.tgt_pct = np.random.choice([0.1,0.25,0.5,0.75,0.8,0.9,0.99])
        # TSL
        self.trail_on = np.random.choice([True, False])
        # MTM LOCKING
        self.lock_profit = np.random.choice([True,False])
        if self.lock_profit:
            self.max_loss = np.random.choice(np.arange(-5000,-500,250))
            self.lock_profit_trigger = np.random.choice(np.arange(100,2000,250))
            self.lock_profit_to = int(self.lock_profit_trigger*np.random.choice(np.arange(0.3,0.99,0.1)))#locks 10%, to 100% 
            self.subsequent_profit_step = np.random.choice(np.arange(100,1000,250))
            self.subsequent_profit_amt = int(self.subsequent_profit_step*np.random.choice(np.arange(0.3,0.99,0.2)))
        else:
            #self.max_loss = -10000
            self.max_loss = np.random.choice([99])
            self.lock_profit_trigger = 0
            self.lock_profit_to = 0
            self.subsequent_profit_step = 0
            self.subsequent_profit_amt = 0
        return self.get_uid_from_params()

    def set_params_from_uid(self, uid):
        s = uid.split('_')
        try:
            assert s[0] == self.strat_id
        except AssertionError:
            raise ValueError(f'Invalid UID {uid} for strat ID {self.strat_id}')
        s = s[1:]
        #####
        #####
        self.underlying = s.pop(0)
        self.active_dte = int(s.pop(0))
        self.session = s.pop(0)
        self.delay = int(s.pop(0))#=='True'
        self.haste = int(s.pop(0))#=='True'
        self.timeframe = int(s.pop(0))
        self.offset = int(s.pop(0))
        self.selector = s.pop(0)
        if self.selector=='M' or self.selector=='P':
            self.selector_val = int(s.pop(0))
        else:
            self.selector_val = float(s.pop(0))
        self.hedge_shift = int(s.pop(0))
        self.lookback = int(s.pop(0))
        self.spike_thresh = float(s.pop(0))
        self.retrace_thresh = float(s.pop(0))
        #self.sl_pct = float(s.pop(0))
        self.sl_type = s.pop(0)
        if self.sl_type == 'PCT':
            self.sl_val = float(s.pop(0))
        elif self.sl_type == 'ABS':
            self.sl_val = int(s.pop(0))
        self.trail_on = s.pop(0)=='True'
        self.tgt_pct = float(s.pop(0))
        self.lock_profit = s.pop(0)=='True'
        self.max_loss = int(s.pop(0))
        self.lock_profit_trigger = int(s.pop(0))
        self.lock_profit_to = int(s.pop(0))
        self.subsequent_profit_step = int(s.pop(0))
        self.subsequent_profit_amt = int(s.pop(0))
        # CROSS CHECK
        print(s)
        
        assert len(s)==0
        self.gen_uid = self.get_uid_from_params()
        print(uid, len(uid))
        print(self.gen_uid, len(self.gen_uid))
        assert uid == self.gen_uid
        self.uid = uid
        print(self.uid)
    
    def get_uid_from_params(self):
        return f"""
        {self.strat_id}_
        {self.underlying}_
        {self.active_dte}_
        {self.session}_
        {self.delay}_
        {self.haste}_
        {self.timeframe}_
        {self.offset}_
        {self.selector}_
        {self.selector_val}_
        {self.hedge_shift}_
        {self.lookback}_
        {self.spike_thresh}_
        {self.retrace_thresh}_
        {self.sl_type}_
        {self.sl_val}_
        {self.trail_on}_
        {self.tgt_pct}_
        {self.lock_profit}_
        {self.max_loss}_
        {self.lock_profit_trigger}_
        {self.lock_profit_to}_
        {self.subsequent_profit_step}_
        {self.subsequent_profit_amt}_
        """.replace('\n', '').replace(' ', '').strip('_')

    def on_new_day(self):
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

        else:
            raise ValueError(f'Unknown Underlying: {self.underlying}')
        # ...
        self.lot_size = self.get_lot_size(self.underlying)
        # ...
        self.symbol_ce = None
        self.symbol_pe = None
        # ...
        self.trade_triggered = False
        self.prices_ce = []#False
        self.prices_pe = []#False
        # ...
        self.triggered_ce = False
        self.triggered_pe = False
        self.spiked_ce = False
        self.spiked_pe = False
        # ...
        self.position_ce = 0
        self.position_pe = 0
        # ...
        self.reset_count_ce = 0
        self.reset_count_pe = 0
        #
        self.ce_high = 0
        self.pe_high = 0
        # ...
        self.mtm = 0
        self.profit_locked_at = 0
        self.profit_locked = False
        #if self.selector != 'M':
        #    self.premium_for_day_selection = False 
        #else:
        self.premium_for_day_selection = True
        # ...
        self.session_vals = sessions_dict[self.session]
        self.start_time = self.session_vals['start_time']#datetime.time(9, 15)
        self.stop_time = self.session_vals['stop_time']#datetime.time (11, 15)
        # ADD DELAY / HASTE TO START / STOP TIME
        dtn = datetime.datetime.now()
        self.start_time = (datetime.datetime.combine(dtn.date(), self.start_time) + datetime.timedelta(minutes=self.delay)).time()
        self.stop_time = (datetime.datetime.combine(dtn.date(), self.stop_time) - datetime.timedelta(minutes=self.haste)).time()
        # ...

    def on_event(self):
        pass

    def on_bar_complete(self):
        # print(self.now)
        #######################################################
        if self.premium_for_day_selection:
            if self.selector == 'P':
                atm_ce = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'CE', 0)
                if atm_ce != None:
                    self.premium_for_day = self.selector_val
                    self.dte = self.get_dte(self.now, atm_ce)
                    self.premium_for_day_selection = False
            if self.selector == 'M':
                self.premium_for_day_selection = False 
                atm_ce = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'CE', 0)
                if atm_ce != None:
                    self.dte = self.get_dte(self.now, atm_ce)
            ##############################
            if self.selector == 'R':
                atm_ce = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'CE', 0)
                atm_pe = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'PE', 0)
                if atm_ce != None and atm_pe != None:
                    self.dte = self.get_dte(self.now, atm_ce)
                    atm_ce_price = self.get_tick(self.now, atm_ce)['c']
                    atm_pe_price = self.get_tick(self.now, atm_pe)['c']
                    atm_cp = (atm_ce_price + atm_pe_price)/2
                    self.premium_for_day = (atm_cp/(1+(0.8*self.dte)))*self.selector_val
                    print(self.premium_for_day, "PREMIUM FOR DAY")
                    self.premium_for_day_selection = False
            if self.selector == 'PCT':
                atm_ce = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'CE', 0)
                if atm_ce != None:
                    self.dte = self.get_dte(self.now, atm_ce)
                    self.premium_for_day = self.selector_val*self.get_tick(self.now, self.mysymbol)['c']
                    self.premium_for_day_selection = False
                    print(self.premium_for_day, "PREMIUM FOR DAY", self.dte)
        #######################################################
        # DTE
        if self.dte != self.active_dte and self.active_dte != 99:
            return
        #######################################################
        #MTM CALCULATION
        self.mtm = self.get_mtm()
        if self.mtm < 0 and self.max_loss < 0 and self.mtm < self.max_loss*2:
            #raise ValueError('MTM loss exceeds 2x of max loss !!!')
            pass
        if self.lock_profit:
            if (self.mtm > self.lock_profit_trigger) and not self.profit_locked:
                self.max_loss = self.lock_profit_to
                self.profit_locked_at = self.mtm
                self.profit_locked = True
            if self.profit_locked:
                if (self.mtm - self.profit_locked_at) > self.subsequent_profit_step:
                    self.max_loss += self.subsequent_profit_amt
                    self.profit_locked_at = self.mtm
        #######################################################
        # EXIT
        if self.position_ce == 1:
            self.current_price_ce = float(self.get_tick(self.now, self.symbol_ce)['c'])
            if self.trail_on:
                if self.sl_type == 'PCT':
                    self.new_sl_ce = self.current_price_ce * (1+self.sl_val)
                elif self.sl_type == 'ABS':
                    self.new_sl_ce = self.current_price_ce +self.sl_val
                if self.new_sl_ce < self.sl_price_ce:
                    self.sl_price_ce = self.new_sl_ce
            self.to_exit = False
            # SL
            if self.current_price_ce >= self.sl_price_ce:
                self.to_exit = True
                self.reason_ce = 'SL'
            # TGT
            if self.current_price_ce <= self.tgt_price_ce:
                self.to_exit = True
                self.reason_ce = 'TGT'
            #MTM
            if self.mtm < self.max_loss:
                self.to_exit = True
                self.reason_ce = 'MTM'
            # TIME
            if self.now.time() >= self.stop_time:
                self.to_exit = True
                self.reason_ce = 'TIME'
            if self.to_exit:
                self.success_ce, _, _ = self.place_spread_trade(
                    self.now, 'BUY', self.lot_size, self.symbol_ce, self.symbol_ce_hedge, note=self.reason_ce
                )
                if self.success_ce:
                    self.position_ce = -1
                    if self.reason_ce == 'TGT': 
                        self.position_ce = 0
        if self.position_pe == 1:
            self.current_price_pe = float(self.get_tick(self.now, self.symbol_pe)['c'])
            if self.trail_on:
                if self.sl_type == 'PCT':
                    self.new_sl_pe = self.current_price_pe*(1+self.sl_val)
                elif self.sl_type == 'ABS':
                    self.new_sl_pe = self.current_price_pe +self.sl_val
                if self.new_sl_pe < self.sl_price_pe:
                    self.sl_price_pe = self.new_sl_pe
            self.to_exit = False
            # SL
            if self.current_price_pe >= self.sl_price_pe:
                self.to_exit = True
                self.reason_pe = 'SL'
            # TGT
            if self.current_price_pe <= self.tgt_price_pe:
                self.to_exit = True
                self.reason_pe = 'TGT'
            #MTM
            if self.mtm < self.max_loss:
                self.to_exit = True
                self.reason_pe = 'MTM'
            # TIME
            if self.now.time() >= self.stop_time:
                self.to_exit = True
                self.reason_pe = 'TIME'
            if self.to_exit:
                self.success_pe, _, _ = self.place_spread_trade(
                    self.now, 'BUY', self.lot_size, self.symbol_pe, self.symbol_pe_hedge, note=self.reason_pe
                )
                if self.success_pe:
                    self.position_pe = -1
                    if self.reason_pe == 'TGT':
                        self.position_pe = 0
        #######################################################
        # TIMEFRAME
        if self.now.time().minute % self.timeframe != self.offset and self.now.time() < self.stop_time:
            return
        #######################################################
        # ENTRY
        if self.now.time() >= self.start_time and self.now.time() < self.stop_time:
            if self.position_ce == 0:
                # select symbol CE
                if self.symbol_ce is None:
                    if self.selector in ['P', 'R', 'PCT']:
                        self.symbol_ce = self.find_symbol_by_premium(
                            self.now, self.underlying, 0, 'CE', self.premium_for_day, force_atm=False, perform_rms_checks=False
                        )
                    else:
                        self.symbol_ce = self.find_symbol_by_moneyness(
                            self.now, self.underlying, 0, 'CE', self.selector_val
                        )
                if self.symbol_ce is not None:
                    # SPIKE
                    if not self.triggered_ce:
                        self.current_price_ce = self.get_tick(self.now, self.symbol_ce)['c']
                        self.prices_ce.append(self.current_price_ce)
                        if len(self.prices_ce)>=self.lookback:
                            if not self.spiked_ce:
                                self.compare_price_ce = self.prices_ce[-self.lookback]
                                self.spike_price_ce = self.compare_price_ce*(1+self.spike_thresh)
                                if self.current_price_ce>=self.spike_price_ce:
                                    self.spiked_ce = True
                                    self.retrace_price_ce = self.spike_price_ce*(1-self.retrace_thresh) 
                            if self.spiked_ce:
                                if self.current_price_ce<=self.retrace_price_ce:
                                    self.triggered_ce = True
                    if self.triggered_ce:
                        if self.hedge_shift == 0:
                            self.symbol_ce_hedge = None
                        else:
                            self.symbol_ce_hedge = self.shift_strike_in_symbol(self.symbol_ce, self.hedge_shift)
                            # HEDGE SANITY CHECK
                            price = self.get_tick(self.now, self.symbol_ce_hedge)['c']
                            if np.isnan(price) or price == 0:
                                self.symbol_ce_hedge = self.find_symbol_by_premium(self.now, self.underlying, 0, "CE", 1, perform_rms_checks=False)                    
                        # sell CE
                        self.success_ce, self.entry_price_ce, self.entry_price_ce_hedge = self.place_spread_trade(
                            self.now, 'SELL', self.lot_size, self.symbol_ce, self.symbol_ce_hedge, note='ENTRY'
                        )
                        if self.success_ce:
                            if self.sl_type == 'PCT':
                                self.sl_price_ce = float(self.entry_price_ce)*(1+self.sl_val)
                            elif self.sl_type == 'ABS':
                                self.sl_price_ce = float(self.entry_price_ce)+self.sl_val
                            self.tgt_price_ce = float(self.entry_price_ce)*(1-self.tgt_pct)
                            self.position_ce = 1
                            self.success_ce = False
                            self.ce_high = 0
            if self.position_pe == 0:
                # select symbol P1
                if self.symbol_pe is None:
                    if self.selector in ['P', 'R', 'PCT']:
                        self.symbol_pe = self.find_symbol_by_premium(
                            self.now, self.underlying, 0, 'PE', self.premium_for_day, force_atm=False, perform_rms_checks=False
                        )
                    else:
                        self.symbol_pe = self.find_symbol_by_moneyness(
                            self.now, self.underlying, 0, 'PE', self.selector_val
                        )
                if self.symbol_pe is not None:
                    # SPIKE
                    if not self.triggered_pe:
                        self.current_price_pe = self.get_tick(self.now, self.symbol_pe)['c']
                        self.prices_pe.append(self.current_price_pe)
                        if len(self.prices_pe)>=self.lookback:
                            if not self.spiked_pe:
                                self.compare_price_pe = self.prices_pe[-self.lookback]
                                self.spike_price_pe = self.compare_price_pe*(1+self.spike_thresh)
                                if self.current_price_pe>=self.spike_price_pe:
                                    self.spiked_pe = True
                                    self.retrace_price_pe = self.spike_price_pe*(1-self.retrace_thresh) 
                            if self.spiked_pe:
                                if self.current_price_pe<=self.retrace_price_pe:
                                    self.triggered_pe = True
                    if self.triggered_pe:
                        if self.hedge_shift == 0:
                            self.symbol_pe_hedge = None
                        else:
                            self.symbol_pe_hedge = self.shift_strike_in_symbol(self.symbol_pe, self.hedge_shift)
                            # HEDGE SANITY CHECK
                            price = self.get_tick(self.now, self.symbol_pe_hedge)['c']
                            if np.isnan(price) or price == 0:
                                self.symbol_pe_hedge = self.find_symbol_by_premium(self.now, self.underlying, 0, "PE", 1, perform_rms_checks=False)
                        # sell PE
                        self.success_pe, self.entry_price_pe, self.entry_price_pe_hedge = self.place_spread_trade(
                            self.now, 'SELL', self.lot_size, self.symbol_pe, self.symbol_pe_hedge, note='ENTRY'
                        )
                        if self.success_pe:
                            if self.sl_type == 'PCT':
                                self.sl_price_pe = float(self.entry_price_pe)*(1+self.sl_val)
                            elif self.sl_type == 'ABS':
                                self.sl_price_pe = float(self.entry_price_pe)+self.sl_val
                            self.tgt_price_pe = float(self.entry_price_pe)*(1-self.tgt_pct)
                            self.position_pe = 1
                            self.success_pe = False
                            self.pe_high = 0

