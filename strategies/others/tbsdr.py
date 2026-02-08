"""
Strategy Name: TBSDR (Time-Based Straddle with Directional Reentry and Rolling)
----------------------------------------------------------------------------------

Overview:
---------
TBSDR is a directional and reactive enhancement of the time-based straddle strategy (TBS).
It sells a straddle (CE + PE) at a scheduled time and manages exits individually per leg
based on stop-loss, target, or session-end. Upon stop-loss, the strategy optionally rolls 
the opposite leg to a new strike directionally, adapting to market movement. This enables 
position rebalancing in response to breakout behavior.

Core Features:
--------------
1. **Time-Based Entry:**
   - Entry is governed by `session` and `delay`, allowing precise control of when positions are opened.

2. **Premium or Moneyness-Based Strike Selection:**
   - Uses `selector` and `selector_val` to identify CE/PE strikes based on desired premium or distance from ATM.

3. **Hedging:**
   - Each leg is protected using a hedge strike determined by a fixed premium (`hedge_prices_dict`).

4. **Exit Conditions:**
   - Exits triggered by:
     - Stop-loss (`sl_pct` above entry),
     - Target profit (`tgt_pct` below entry),
     - End of session (`stop_time`).

5. **Directional Rolling Logic:**
   - If one leg hits SL and the other is still active, the strategy rolls the surviving leg to a new strike,
     directionally selected using `selector_dir` and `selector_val_dir`.

6. **Reentry Logic:**
   - If a leg hits SL and later reverts back to its original entry price or below, it re-enters the same strike
     (up to `max_reset` times).

7. **Trailing Stop Option:**
   - Optional trailing SL logic can reduce risk if `trail_on` is enabled.

Use Case:
---------
Designed for intraday expiry trading with potential breakout conditions. It suits traders seeking 
reactivity to directional movement while still operating within a risk-defined straddle framework.

Parameters:
-----------
- `active_weekday`: Active DTE for strategy (e.g. 0 for expiry day, 99 for all).
- `session`, `delay`, `timeframe`: Define when entry and monitoring should occur.
- `selector` / `selector_val`: Primary CE/PE strike selection logic.
- `hedge_shift`: Determines how far hedges are from main positions.
- `sl_pct`, `tgt_pct`: Stop-loss and target levels.
- `max_reset`: Max reentry attempts after SL.
- `trail_on`: Enables trailing SL mechanism.
- `selector_dir`, `selector_val_dir`: Directional reentry parameters after SL roll.

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

#################

class TBSDR(EventInterface):
    
    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()

    def get_random_uid(self):
        # Select
        self.active_weekday = 99#np.random.choice(weekdays)
        self.session = np.random.choice(sessions)
        self.timeframe = np.random.choice(timeframes)
        self.underlying = np.random.choice(underlyings)
        self.selector = 'P' #np.random.choice(selectors)
        if self.selector == 'M':
            self.selector_val = np.random.choice(moneynesses)
        elif self.selector == 'P':
            self.selector_val = np.random.choice(seek_prices)
        self.hedge_shift = np.random.choice(hedge_shifts)
        self.sl_pct = np.random.choice(sl_pcts)
        self.tgt_pct = np.random.choice(tgt_pcts)
        self.max_reset = np.random.choice(resets)
        self.trail_on = np.random.choice([True, False])
        self.delay = np.random.choice(delays)
        self.selector_dir = np.random.choice(selectors)
        if self.selector_dir == 'M':
            self.selector_val_dir = np.random.choice(moneynesses)
        elif self.selector_dir == 'P':
            self.selector_val_dir = np.random.choice([15, 25, 50, 75, 100, 150])
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
        self.hedge_shift = int(s.pop(0))
        self.sl_pct = float(s.pop(0))
        self.tgt_pct = float(s.pop(0))
        self.max_reset = int(s.pop(0))
        self.trail_on = s.pop(0)=='True'
        self.selector_dir = s.pop(0)
        self.selector_val_dir = int(s.pop(0))
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
        {self.hedge_shift}_
        {self.sl_pct}_
        {self.tgt_pct}_
        {self.max_reset}_
        {self.trail_on}_
        {self.selector_dir}_
        {self.selector_val_dir}_
        """.replace('\n', '').replace(' ', '').strip('_')

    def on_new_day(self):
        # ...
        self.lot_size = self.get_lot_size(self.underlying)
        # ...
        self.symbol_ce = None
        self.symbol_pe = None
        # ...
        self.trade_triggered = False
        # ...
        self.position_ce = 0
        self.position_pe = 0
        # ...
        self.reset_count_ce = 0
        self.reset_count_pe = 0
        # ...
        self.session_vals = sessions_dict[self.session]
        self.start_time = self.session_vals['start_time']#datetime.time(9, 15)
        self.stop_time = self.session_vals['stop_time']#datetime.time (11, 15)
        # ADD DELAY TO START TIME
        dtn = datetime.datetime.now()
        self.start_time = (datetime.datetime.combine(dtn.date(), self.start_time) + datetime.timedelta(minutes=self.delay)).time()

    def on_event(self):
        pass

    def on_bar_complete(self):
        # ...
        if self.now.weekday() != self.active_weekday and self.active_weekday != 99:
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
            if self.current_price_ce <= self.tgt_price_ce:
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
                    # ROLL UP
                    if self.reason_ce == 'SL' and self.position_pe == 1:
                        self.success_pe, _ = self.place_trade(
                            self.now, 'BUY', self.lot_size, self.symbol_pe, note="ROLLUP_EXIT"
                        )
                        if self.success_pe:
                            self.position_pe = 0
                            self.symbol_pe = self.find_symbol_by_premium(
                                self.now, self.underlying, 0, 'PE', self.selector_val_dir
                            )
                            self.success_pe, self.entry_price_pe = self.place_trade(
                                self.now, 'SELL', self.lot_size, self.symbol_pe, note="ROLLUP_ENTRY"
                            )
                            if self.success_pe:
                                self.sl_price_pe = float(self.entry_price_pe)*(1+self.sl_pct)
                                self.tgt_price_pe = float(self.entry_price_pe)*(1-self.tgt_pct)
                                self.position_pe = 1
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
            if self.current_price_pe <= self.tgt_price_pe:
                self.to_exit = True
                self.reason_pe = 'TGT'
            # TIME
            if self.now.time() >= self.stop_time:
                self.to_exit = True
                self.reason_pe = 'TIME'
            if self.to_exit:
                #place_trade(timestamp, 'BUY', 25, symbol_pe, note='EXIT_'+reason)
                #place_trade(timestamp, 'SELL', 25, symbol_pe_hedge, note='EXIT_HEDGE_'+reason)
                self.success_pe, _, _ = self.place_spread_trade(
                    self.now, 'BUY', self.lot_size, self.symbol_pe, self.symbol_pe_hedge, note=self.reason_pe
                )
                if self.success_pe:
                    self.position_pe = -1
                    # ROLL UP
                    if self.reason_pe == 'SL' and self.position_ce == 1:
                        self.success_ce, _ = self.place_trade(
                            self.now, 'BUY', self.lot_size, self.symbol_ce, note="ROLLUP_EXIT"
                        )
                        if self.success_ce:
                            self.position_ce = 0
                            self.symbol_ce = self.find_symbol_by_premium(
                                self.now, self.underlying, 0, 'CE', self.selector_val_dir
                            )
                            self.success_ce, self.entry_price_ce = self.place_trade(
                                self.now, 'SELL', self.lot_size, self.symbol_ce, note="ROLLUP_ENTRY"
                            )
                            if self.success_ce:
                                self.sl_price_ce = float(self.entry_price_ce)*(1+self.sl_pct)
                                self.tgt_price_ce = float(self.entry_price_ce)*(1-self.tgt_pct)
                                self.position_ce = 1
        # RE-ENTRY
        if self.position_ce == -1:
            self.current_price_ce = float(self.get_tick(self.now, self.symbol_ce)['c'])
            #print(self.now, self.current_price_ce)
            if self.reason_ce == 'SL' and self.reset_count_ce < self.max_reset and self.current_price_ce <= float(self.entry_price_ce):
                self.position_ce = 0
                self.reset_count_ce += 1
        if self.position_pe == -1:
            self.current_price_pe = float(self.get_tick(self.now, self.symbol_pe)['c'])
            #print(self.now, self.current_price_pe)
            if self.reason_pe == 'SL' and self.reset_count_pe < self.max_reset and self.current_price_pe <= float(self.entry_price_pe):
                self.position_pe = 0
                self.reset_count_pe += 1
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
                    #self.symbol_ce_hedge = self.shift_strike_in_symbol(self.symbol_ce, self.hedge_shift)
                    self.symbol_ce_hedge = self.find_symbol_by_premium(
                        self.now, self.underlying, 0, 'CE', hedge_prices_dict[self.underlying]#self.selector_val
                        , perform_rms_checks=False
                    )
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
                    #self.symbol_pe_hedge = self.shift_strike_in_symbol(self.symbol_pe, self.hedge_shift)
                    self.symbol_pe_hedge = self.find_symbol_by_premium(
                        self.now, self.underlying, 0, 'PE', hedge_prices_dict[self.underlying]#self.selector_val
                        , perform_rms_checks=False
                    )
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

