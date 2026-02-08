"""
TBS Strategy (Time-Based Strangle with Hedge and Exit Logic WITHOUT REENTRY)
------------------------------------------------------------

Strategy Idea:
--------------
The TBS (Time-Based Strangle) strategy sells a CE and a PE option (strangle) on a selected underlying 
(e.g., NIFTY) within a defined time window during the trading day. Each leg is hedged with a further 
OTM option to control risk. Exit is based on stop-loss, target, or end-of-session time.

Key Features:
-------------
- Time-based entry controlled by session and delay
- Entry leg selection based on premium or moneyness
- Trade execution happens only when the DTE of the ATM CE equals `active_dte`
- Each leg is hedged using an OTM premium-based hedge
- Exit triggered by SL, target, or session stop
- Optional trailing SL
- Re-entry is disabled in this version

Parameters:
-----------
- `active_dte` (int): Days to expiry required to activate the strategy (e.g., 0 = expiry day, 99 = all DTEs allowed)
- `session` (str): Session identifier, e.g., 'x0', 'x1' (mapped in sessions_dict)
- `delay` (int): Delay in minutes added to session start time
- `timeframe` (int): Bar interval in minutes
- `underlying` (str): Instrument to trade (e.g., 'NIFTY')
- `selector` (str): 'P' = premium-based entry, 'M' = moneyness-based entry
- `selector_val` (float): Premium (if 'P') or moneyness (if 'M')
- `hedge_shift` (int): Strike gap from entry leg to hedge leg (not used directly)
- `sl_pct` (float): Stop loss % over entry price (e.g., 0.25 = 25%)
- `tgt_pct` (float): Target % below entry price (e.g., 0.5 = 50%)
- `trail_on` (bool): Whether to apply trailing SL logic
- `Target` (bool): Whether to enforce target-based exits

Sample UID:
-----------
tbs_0_x0_0_1_NIFTY_P_20_10_0.25_0.5_True_True

Meaning:
- Execute when ATM CE has DTE = 0 (e.g., expiry day)
- Use session 'x0' (with no delay)
- 1-minute bars
- Entry strike selection by ₹20 premium
- SL = 25%, TGT = 50%, no resets
- Trailing SL and TGT both enabled

Assumptions:
------------
- Uses `place_spread_trade()` for hedged entries and exits
- Re-entry is disabled
"""


import datetime
import pandas as pd
import numpy as np
from utils.definitions import *

if REDIS:
    from engine.ems import EventInterface
else:
    from engine.ems_db import EventInterface

from utils.sessions import *

class TBS(EventInterface):
    def __init__(self):
        super().__init__()
        self.strat_id = self.__class__.__name__.lower()
        self.premium_for_day_selection = True

    def get_random_uid(self):
        self.active_dte = np.random.choice([0, 1, 2])
        self.session = 'x0'
        self.timeframe = np.random.choice(timeframes)
        self.underlying = np.random.choice(['NIFTY', 'SENSEX'])
        self.selector = np.random.choice(selectors)

        self.selector_val = (
            np.random.choice(moneynesses) if self.selector == 'M'
            else np.random.choice(seek_prices)
        )

        self.hedge_shift = np.random.choice(hedge_shifts)
        self.sl_pct = np.random.choice(sl_pcts)
        self.tgt_pct = np.random.choice(tgt_pcts)
        self.trail_on = np.random.choice([True, False])
        self.delay = np.random.choice(delays)
        self.Target = np.random.choice([True, False])
        return self.get_uid_from_params()

    def set_params_from_uid(self, uid):
        s = uid.split('_')
        assert s[0] == self.strat_id
        s = s[1:]

        self.active_dte = int(s.pop(0))
        self.session = s.pop(0)
        self.delay = int(s.pop(0))
        self.timeframe = int(s.pop(0))
        self.underlying = s.pop(0)
        self.selector = s.pop(0)
        self.selector_val = int(s.pop(0))
        self.hedge_shift = int(s.pop(0))
        self.sl_pct = float(s.pop(0))
        self.tgt_pct = float(s.pop(0))
        self.trail_on = s.pop(0) == 'True'
        self.Target = s.pop(0) == 'True'

        assert len(s) == 0
        self.gen_uid = self.get_uid_from_params()
        assert uid == self.gen_uid
        self.uid = uid

    def get_uid_from_params(self):
        return f"{self.strat_id}_{self.active_dte}_{self.session}_{self.delay}_{self.timeframe}_{self.underlying}_" \
               f"{self.selector}_{self.selector_val}_{self.hedge_shift}_{self.sl_pct}_{self.tgt_pct}_" \
               f"{self.trail_on}_{self.Target}".strip('_')

    def on_new_day(self):
        self.lot_size = self.get_lot_size(self.underlying)
        self.symbol_ce = None
        self.symbol_pe = None
        self.position_ce = 0
        self.position_pe = 0
        self.dte = None
        self.session_vals = sessions_dict[self.session]
        self.start_time = (datetime.datetime.combine(datetime.date.today(), self.session_vals['start_time']) +
                           datetime.timedelta(minutes=self.delay)).time()
        self.stop_time = self.session_vals['stop_time']

    def on_event(self):
        pass

    def on_bar_complete(self):
        # ...
        if self.premium_for_day_selection:
            atm_ce = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'CE', 0)
            atm_pe = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'PE', 0)

            if atm_ce != None and atm_pe != None:
                self.dte = self.get_dte(self.now, atm_ce)
                self.premium_for_day_selection = False
        
        if self.dte != self.active_dte and self.active_dte != 99:
            return
        
        if self.now.time().minute % self.timeframe != 0 and self.now.time() < self.stop_time:
            return
        # EXIT
        if self.position_ce == 1:
            self.current_price_ce = float(self.get_tick(self.now, self.symbol_ce)['c'])
            if self.trail_on:
                self.new_sl_ce = self.current_price_ce * (1+self.sl_pct)
                if self.new_sl_ce < self.sl_price_ce:
                    self.sl_price_ce = self.new_sl_ce
            self.to_exit = False
            # SL
            if self.current_price_ce >= self.sl_price_ce:
                self.to_exit = True
                self.reason_ce = 'SL'
            # TGT
            if self.current_price_ce <= self.tgt_price_ce and self.Target:
                self.to_exit = True
                self.reason_ce = 'TGT'
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
        if self.position_pe == 1:
            self.current_price_pe = float(self.get_tick(self.now, self.symbol_pe)['c'])
            if self.trail_on:
                self.new_sl_pe = self.current_price_pe * (1+self.sl_pct)
                if self.new_sl_pe < self.sl_price_pe:
                    self.sl_price_pe = self.new_sl_pe
            self.to_exit = False
            # SL
            if self.current_price_pe >= self.sl_price_pe:
                self.to_exit = True
                self.reason_pe = 'SL'
            # TGT
            if self.current_price_pe <= self.tgt_price_pe and self.Target:
                self.to_exit = True
                self.reason_pe = 'TGT'
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
        # ENTRY
        if self.now.time() >= self.start_time and self.now.time() < self.stop_time:
            if self.position_ce == 0:
                # select symbol CE
                if self.selector == 'P':
                    self.symbol_ce = self.find_symbol_by_premium(
                        self.now, self.underlying, 0, 'CE', self.selector_val
                    )
                elif self.selector == 'M':
                    self.symbol_ce = self.find_symbol_by_moneyness(
                        self.now, self.underlying, 0, 'CE', self.selector_val
                    )
                if self.symbol_ce is not None:
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
                        self.sl_price_ce = float(self.entry_price_ce)*(1+self.sl_pct)
                        self.tgt_price_ce = float(self.entry_price_ce)*(1-self.tgt_pct)
                        self.position_ce = 1
            if self.position_pe == 0:
                # select symbol P1
                if self.selector == 'P':
                    self.symbol_pe = self.find_symbol_by_premium(
                        self.now, self.underlying, 0, 'PE', self.selector_val
                    )
                elif self.selector == 'M':
                    self.symbol_pe = self.find_symbol_by_moneyness(
                        self.now, self.underlying, 0, 'PE', self.selector_val
                    )
                if self.symbol_pe is not None:
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
                        self.sl_price_pe = float(self.entry_price_pe)*(1+self.sl_pct)
                        self.tgt_price_pe = float(self.entry_price_pe)*(1-self.tgt_pct)
                        self.position_pe = 1
