"""
Strategy Name: UT

Overview:
---------
The UT strategy is an intraday **breakout entry** options selling strategy on a selected underlying (e.g., BANKNIFTY).
It monitors combined premium (CE + PE) of ATM/near ATM strangle and enters short positions on **only one leg** (CE or PE) 
once the combined premium crosses a predefined threshold. It includes optional trailing stop loss and target-based exits, 
along with controlled re-entries upon SL.

Key Idea:
---------
- Track CE + PE premiums (strangle) after a defined start time.
- If the combined premium exceeds a certain threshold (`trigger_pct` over base strangle value), choose to sell either CE or PE,
  depending on which has a **lower premium**.
- Apply SL and Target on the sold option leg. If hit, optional re-entry is triggered based on `max_reset`.

Parameters:
-----------
- `active_weekday` (int): Day of the week to activate strategy (0=Mon, ..., 6=Sun). `99` = all weekdays.
- `session` (str): Session key from `sessions_dict` (e.g., 'x0').
- `delay` (int): Delay (in minutes) from session start before entry can begin.
- `timeframe` (int): Strategy runs on bar completion every `timeframe` minutes.
- `underlying` (str): Underlying instrument (e.g., 'BANKNIFTY').
- `selector` (str): Strike selection method ('M' = moneyness based, 'P' = premium based).
- `selector_val` (int): Value for strike selection (e.g., moneyness level or desired premium).
- `hedge_shift` (int): Number of strikes away for hedging (used to create spread).
- `sl_pct` (float): Stop loss percentage above entry premium (e.g., 0.35 = 35%).
- `tgt_pct` (float): Target percentage below entry premium.
- `max_reset` (int): Maximum number of SL-based re-entries allowed.
- `trail_on` (bool): Whether to enable trailing stop-loss/target updates.
- `trigger_pct` (float): Threshold over initial straddle price to trigger entry (e.g., 0.15 = 15% increase).
- `trail_pct` (float): Percentage for trailing buffer (if `trail_on` is True).

Entry Logic:
------------
- Wait until `start_time + delay`.
- Monitor CE + PE premiums selected via `selector` and `selector_val`.
- If combined premium ≥ (base premium × (1 + `trigger_pct`)), entry is triggered.
- Trade only one leg (whichever has **lower** premium between CE and PE), hedge with shifted strike.

Exit Logic:
-----------
- Exit position if:
  - SL is hit: premium increases by `sl_pct`.
  - TGT is hit: premium drops by `tgt_pct`.
  - End of session.
- Optional trailing logic adjusts SL and TGT downwards once TGT is hit.
- Re-entry is allowed on SL up to `max_reset` times per leg.

Notes:
------
- The strategy avoids taking both CE and PE positions — only one leg is active at a time.
- Strike symbols and hedge checks are robust to missing tick data.
- Can be extended to support different underlyings, additional filters, or directionally biased filters.

Example UID:
------------
ut_99_x0_30_1_NIFTY_M_0_10_0.35_0.5_1_False_0.2_0.1

This corresponds to:
- All weekdays (`99`), session 'x0', 30 min delay, 1-minute timeframe
- BANKNIFTY, moneyness 0, hedge 10 strikes away
- SL: 35%, Target: 50%, 1 re-entry allowed, no trailing
- Trigger when straddle increases by 20%, trailing buffer: 10%
"""

import datetime
import pandas as pd
import numpy as np

from utils.sessions import *
from utils.utility import *
from utils.definitions import *

if REDIS:
    from engine.ems import EventInterface
else:
    from engine.ems_db import EventInterface

class UT(EventInterface):  
    def __init__(self):
        super().__init__()
        self.strat_id = self.__class__.__name__.lower()

    def get_random_uid(self):
        # Select
        self.active_weekday = 99#np.random.choice(weekdays)
        self.session = 'x0' # np.random.choice(sessions)
        self.timeframe = np.random.choice([1])
        self.underlying = np.random.choice(['BANKNIFTY'])
        self.selector = np.random.choice(['M'])
        if self.selector == 'M':
            self.selector_val = np.random.choice([0])
        elif self.selector == 'P':
            if self.underlying == 'NIFTY':
                self.selector_val = np.random.choice(seek_prices)
            elif self.underlying == 'SENSEX':
                self.selector_val = np.random.choice([x*2 for x in seek_prices if x != 25])
        self.hedge_shift = np.random.choice([7, 10])
        self.sl_pct = min(0.8, max(0.3, round(.05 * round(np.random.choice(np.random.random(1)).round(2)/.05), 2))) # np.random.rand(10)*0.5
        self.tgt_pct = max(0.5, round(.05 * round(np.random.choice(np.random.random(1)).round(2)/.05), 2))
        self.max_reset = np.random.choice([0, 1, 2])
        self.trail_on = np.random.choice([False])
        self.delay = np.random.choice(range(0, 120, 10))
        # ...
        trigger_pcts = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        self.trigger_pct = np.random.choice(trigger_pcts)
        self.trail_pct = np.random.choice(trigger_pcts)
        # self.selector_val_str = np.random.choice(seek_prices)
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
        # ...
        self.trigger_pct = float(s.pop(0))
        self.trail_pct = float(s.pop(0))
        # self.selector_val_str = int(s.pop(0))
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
        {self.trigger_pct}_
        {self.trail_pct}_
        """.replace('\n', '').replace(' ', '').strip('_')

    def on_new_day(self):
        # ...
        self.lot_size = self.get_lot_size(self.underlying)
        # ...
        self.symbol_ce = None
        self.symbol_pe = None
        # ...
        self.symbols_selected = False
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

        if self.underlying in ['BANKNIFTY', 'MIDCPNIFTY', 'FINNIFTY']:
            self.expiry_idx = 'm'
        else:
            self.expiry_idx = 0
        
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
                # RUPEE SL TRAILING
                # self.new_sl_ce = self.current_price_ce * (1+self.sl_pct)
                # if self.new_sl_ce < self.sl_price_ce:
                #     self.sl_price_ce = self.new_sl_ce

                # BUFFER TARGET TRAILING
                if self.current_price_ce <= self.tgt_price_ce:
                    self.tgt_price_ce = self.current_price_ce * ( 1 - self.trail_pct )
                    self.sl_price_ce = self.current_price_ce * ( 1 + self.trail_pct )

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


        if self.position_pe == 1:
            self.current_price_pe = float(self.get_tick(self.now, self.symbol_pe)['c'])
            if self.trail_on:
                # RUPEE SL TRAILING
                # self.new_sl_pe = self.current_price_pe * (1+self.sl_pct)
                # if self.new_sl_pe < self.sl_price_pe:
                #     self.sl_price_pe = self.new_sl_pe
                
                # BUFFER TARGET TRAILING
                if self.current_price_pe <= self.tgt_price_pe:
                    self.tgt_price_pe = self.current_price_pe * ( 1 - self.trail_pct )
                    self.sl_price_pe = self.current_price_pe * ( 1 + self.trail_pct )

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
        # RE-ENTRY
        if self.position_ce == -1:
            self.current_price_ce = float(self.get_tick(self.now, self.symbol_ce)['c'])
            #print(self.now, self.current_price_ce)
            if self.reason_ce == 'SL' and self.reset_count_ce < self.max_reset: #and self.current_price_ce <= float(self.entry_price_ce):
                self.symbol_ce = None
                self.position_ce = 0
                self.reset_count_ce += 1
                self.symbol_ce = None
                self.symbol_pe = None
                self.symbols_selected = False
                self.trade_triggered = False
        if self.position_pe == -1:
            self.current_price_pe = float(self.get_tick(self.now, self.symbol_pe)['c'])
            #print(self.now, self.current_price_pe)
            if self.reason_pe == 'SL' and self.reset_count_pe < self.max_reset: #and self.current_price_pe <= float(self.entry_price_pe):
                self.symbol_pe = None
                self.position_pe = 0
                self.reset_count_pe += 1
                self.symbol_ce = None
                self.symbol_pe = None
                self.symbols_selected = False
                self.trade_triggered = False
        # ENTRY
        if self.now.time() >= self.start_time and self.now.time() < self.stop_time:
            # select STR
            if not self.symbols_selected:
                if self.selector == 'P':
                    self.symbol_ce = self.find_symbol_by_premium(
                        self.now, self.underlying, self.expiry_idx, 'CE', self.selector_val
                    )
                    self.symbol_pe = self.find_symbol_by_premium(
                        self.now, self.underlying, self.expiry_idx, 'PE', self.selector_val
                    )
                elif self.selector == 'M':
                    self.symbol_ce = self.find_symbol_by_moneyness(
                        self.now, self.underlying, self.expiry_idx, 'CE', self.selector_val
                    )
                    self.symbol_pe = self.find_symbol_by_moneyness(
                        self.now, self.underlying, self.expiry_idx, 'PE', self.selector_val
                    )

                if self.symbol_ce is not None and self.symbol_pe is not None:
                    self.symbols_selected = True
                    self.price_ce = self.get_tick(self.now, self.symbol_ce)['c']
                    self.price_pe = self.get_tick(self.now, self.symbol_pe)['c']
                    self.price_cp = float(self.price_ce) + float(self.price_pe)
                    self.trigger_price = self.price_cp*(1+self.trigger_pct)
            # track STR for trigger
            if self.symbols_selected and not self.trade_triggered:
                self.price_ce = self.get_tick(self.now, self.symbol_ce)['c']
                self.price_pe = self.get_tick(self.now, self.symbol_pe)['c']
                self.price_cp = float(self.price_ce) + float(self.price_pe)
                if self.price_cp >= self.trigger_price:
                    self.trade_triggered = True
                    if float(self.price_ce) >= float(self.price_pe):
                        self.tradeopt = 'PE'
                        self.symbol_ce = None
                        self.symbol_pe = None
                    else:
                        self.tradeopt = 'CE'
                        self.symbol_ce = None
                        self.symbol_pe = None
            # on trigger...
            if self.trade_triggered:
                if self.position_ce == 0 and self.tradeopt == 'CE':
                    # select symbol CE
                    if self.symbol_ce is None:
                        if self.selector == 'P':
                            self.symbol_ce = self.find_symbol_by_premium(
                                self.now, self.underlying, self.expiry_idx, 'CE', self.selector_val
                            )
                        elif self.selector == 'M':
                            self.symbol_ce = self.find_symbol_by_moneyness(
                                self.now, self.underlying, self.expiry_idx, 'CE', self.selector_val
                            )
                    if self.symbol_ce is not None: #and self.current_price_ce < self.trigger_price_ce:
                        self.symbol_ce_hedge = self.shift_strike_in_symbol(self.symbol_ce, self.hedge_shift)
                        # HEDGE SANITY CHECK
                        price = self.get_tick(self.now, self.symbol_ce_hedge)['c']
                        if np.isnan(price) or price == 0:
                            self.symbol_ce_hedge = self.find_symbol_by_premium(self.now, self.underlying, self.expiry_idx, "CE", 1, perform_rms_checks=False, force_atm=False)
                        # self.symbol_ce_hedge = self.find_symbol_by_premium(
                        #     self.now, self.underlying, 0, 'CE', hedge_prices_dict[self.underlying]#self.selector_val
                        #     , perform_rms_checks=False
                        # )
                        # sell CE
                        self.success_ce, self.entry_price_ce, self.entry_price_ce_hedge = self.place_spread_trade(
                            self.now, 'SELL', self.lot_size, self.symbol_ce, self.symbol_ce_hedge, note='ENTRY'
                        )
                        if self.success_ce:
                            self.sl_price_ce = float(self.entry_price_ce)*(1+self.sl_pct)
                            self.tgt_price_ce = float(self.entry_price_ce)*(1-self.tgt_pct)
                            self.position_ce = 1
                if self.position_pe == 0 and self.tradeopt == 'PE':
                    # select symbol PE
                    if self.symbol_pe is None:
                        if self.selector == 'P':
                            self.symbol_pe = self.find_symbol_by_premium(
                                self.now, self.underlying, self.expiry_idx, 'PE', self.selector_val
                            )
                        elif self.selector == 'M':
                            self.symbol_pe = self.find_symbol_by_moneyness(
                                self.now, self.underlying, self.expiry_idx, 'PE', self.selector_val
                            )
                    if self.symbol_pe is not None: #and self.current_price_pe < self.trigger_price_pe:
                        self.symbol_pe_hedge = self.shift_strike_in_symbol(self.symbol_pe, self.hedge_shift)
                        # HEDGE SANITY CHECK
                        price = self.get_tick(self.now, self.symbol_pe_hedge)['c']
                        if np.isnan(price) or price == 0:
                            self.symbol_pe_hedge = self.find_symbol_by_premium(self.now, self.underlying, self.expiry_idx, "PE", 1, perform_rms_checks=False, force_atm=False)
                        # self.symbol_pe_hedge = self.find_symbol_by_premium(
                        #     self.now, self.underlying, 0, 'PE', hedge_prices_dict[self.underlying]#self.selector_val
                        #     , perform_rms_checks=False
                        # )
                        # sell PE
                        self.success_pe, self.entry_price_pe, self.entry_price_pe_hedge = self.place_spread_trade(
                            self.now, 'SELL', self.lot_size, self.symbol_pe, self.symbol_pe_hedge, note='ENTRY'
                        )
                        if self.success_pe:
                            self.sl_price_pe = float(self.entry_price_pe)*(1+self.sl_pct)
                            self.tgt_price_pe = float(self.entry_price_pe)*(1-self.tgt_pct)
                            self.position_pe = 1