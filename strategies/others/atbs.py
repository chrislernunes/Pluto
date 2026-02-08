"""
ATBS Strategy (Advanced Time-Based Strangle with Adaptive Entry and Hedging)
----------------------------------------------------------------------------

Strategy Idea:
--------------
ATBS builds on the basic Time-Based Strangle (TBS) framework by introducing adaptive mechanisms:
- Entry legs (CE/PE) are selected based on fixed moneyness, premium, or a percentage of spot.
- Legs are hedged dynamically based on % of the total straddle premium.
- Exit can happen due to SL, TGT, TIME, or a spot-based SL (optional).
- Allows controlled re-entry after SL or TGT based exits.

Key Features:
-------------
- Selector types: 'M' (moneyness), 'P' (premium), 'PCTA' (percent of spot)
- Entry and hedge leg premiums are selected based on percentage of straddle price
- SL and TGT based on net premium (entry - hedge)
- Optional trailing SL and SL reset to cost
- Controlled re-entry via `reentry_type`
- Dynamically infers underlying spot symbol

Parameters:
-----------
- `active_dte` (int): Days to expiry required to activate the strategy (e.g., 0 = expiry day, 99 = run always)
- `session` (str): Session ID from `sessions_dict`
- `delay` (int): Minutes to delay from session start
- `timeframe` (int): Bar frequency
- `underlying` (str): e.g., 'BANKNIFTY'
- `selector` (str): 'M', 'P', or 'PCTA'
- `selector_val` (float or int): Strike distance, premium, or % of spot
- `hedge_shift` (int): Placeholder, unused
- `sl_pct` (float): SL % on net premium
- `tgt_pct` (float): TGT % on net premium
- `max_reset` (int): Allowed re-entries
- `trail_on` (bool): Enable trailing SL
- `sl_to_cost` (bool): Reset opposite leg SL to cost if one leg hits SL
- `sell_prem_perc` (float): % of straddle premium to use for entry leg selection
- `hedge_prem_perc` (float): % of straddle premium to use for hedge leg selection
- `reentry_type` (str): Either 'sl' or 'tgt'

Sample UIDs:
-------------
# PCTA (percent of spot)
atbs_0_x0_1_1_NIFTY_PCTA_0.5_2_0.2_0.5_1_True_True_0.2_0.05_sl

# P (premium-based)
atbs_0_x0_1_1_NIFTY_P_50_2_0.2_0.5_1_True_True_0.2_0.05_tgt

# M (moneyness-based)
atbs_0_x0_1_1_NIFTY_M_3_2_0.2_0.5_1_True_True_0.2_0.05_sl
"""

import datetime
import numpy as np
from utils.sessions import *
from utils.definitions import *

if REDIS:
    from engine.ems import EventInterface
else:
    from engine.ems_db import EventInterface

class ATBS(EventInterface):
    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()

    def get_random_uid(self):
        self.active_dte = np.random.choice([0, 1, 2, 99])
        self.session = 'x0'
        self.delay = np.random.choice(delays)
        self.timeframe = np.random.choice(timeframes)
        self.underlying = 'BANKNIFTY'
        self.selector = np.random.choice(['M', 'P', 'PCTA'])
        if self.selector == 'M':
            self.selector_val = np.random.choice(moneynesses)
        elif self.selector == 'P':
            self.selector_val = np.random.choice(seek_prices)
        elif self.selector == 'PCTA':
            self.selector_val = np.random.choice([0.25, 0.5, 0.75, 1.0])
        self.hedge_shift = 0
        self.sl_pct = np.random.choice(sl_pcts)
        self.tgt_pct = np.random.choice(tgt_pcts)
        self.max_reset = np.random.choice(resets)
        self.trail_on = np.random.choice([True, False])
        self.sl_to_cost = np.random.choice([True, False])
        self.sell_prem_perc = np.random.choice([0.2, 0.3, 0.5])
        self.hedge_prem_perc = np.random.choice([0.02, 0.05])
        self.reentry_type = np.random.choice(['sl', 'tgt'])
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
        if self.selector == 'PCTA':
            self.selector_val = float(s.pop(0))
        else:
            self.selector_val = int(s.pop(0))
        self.hedge_shift = int(s.pop(0))
        self.sl_pct = float(s.pop(0))
        self.tgt_pct = float(s.pop(0))
        self.max_reset = int(s.pop(0))
        self.trail_on = s.pop(0) == 'True'
        self.sl_to_cost = s.pop(0) == 'True'
        self.sell_prem_perc = float(s.pop(0))
        self.hedge_prem_perc = float(s.pop(0))
        self.reentry_type = s.pop(0)
        assert len(s) == 0
        self.gen_uid = self.get_uid_from_params()
        assert uid == self.gen_uid
        self.uid = uid

    def get_uid_from_params(self):
        return f"{self.strat_id}_{self.active_dte}_{self.session}_{self.delay}_{self.timeframe}_" \
               f"{self.underlying}_{self.selector}_{self.selector_val}_{self.hedge_shift}_{self.sl_pct}_" \
               f"{self.tgt_pct}_{self.max_reset}_{self.trail_on}_{self.sl_to_cost}_" \
               f"{self.sell_prem_perc}_{self.hedge_prem_perc}_{self.reentry_type}".strip('_')

    def on_new_day(self):
        self.lot_size = self.get_lot_size(self.underlying)
        self.strike_shift = self.get_strike_diff(self.underlying)
        self.mysymbol = f"{self.underlying}SPOT"
        self.position_ce = 0
        self.position_pe = 0
        self.reset_count_ce = 0
        self.reset_count_pe = 0
        self.symbol_selection = True
        self.session_vals = sessions_dict[self.session]
        self.start_time = (datetime.datetime.combine(datetime.date.today(), self.session_vals['start_time']) +
                           datetime.timedelta(minutes=self.delay)).time()
        self.stop_time = self.session_vals['stop_time']

    def on_bar_complete(self):
        # ----------- Symbol Selection (once per day) -------------
        if self.symbol_selection:
            spot_price = float(self.get_tick(self.now, self.mysymbol)['c'])
            atm_strike = round(spot_price / self.strike_shift) * self.strike_shift
            if self.selector == 'PCTA':
                moneyness_delta = round(((self.selector_val * spot_price) / 100) / self.strike_shift) * self.strike_shift
                self.symbol_ce_strike = round((spot_price + moneyness_delta) / self.strike_shift) * self.strike_shift
                self.symbol_pe_strike = round((spot_price - moneyness_delta) / self.strike_shift) * self.strike_shift
                self.selector_val_ce_ = int((self.symbol_ce_strike - atm_strike) / self.strike_shift)
                self.selector_val_pe_ = int((atm_strike - self.symbol_pe_strike) / self.strike_shift)
            else:
                self.selector_val_ce_ = self.selector_val_pe_ = self.selector_val
            self.symbol_selection = False

        # ----------- DTE Check -------------
        atm_ce = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'CE', 0)
        if not atm_ce or (self.get_dte(self.now, atm_ce) != self.active_dte and self.active_dte != 99):
            return

        # ----------- Time Check -------------
        if self.now.time().minute % self.timeframe != 0 and self.now.time() < self.stop_time:
            return

        # ----------- Entry Logic -------------
        if self.position_ce == 0:
            self.symbol_ce = self._select_leg_symbol('CE', self.selector_val_ce_)
            self.symbol_ce_hedge = self._select_hedge_symbol('CE')
            if self.symbol_ce and self.symbol_ce_hedge:
                success, entry, hedge_entry = self.place_spread_trade(self.now, 'SELL', self.lot_size,
                                                                      self.symbol_ce, self.symbol_ce_hedge, note='ENTRY')
                if success:
                    self.entry_price_ce = entry - hedge_entry
                    self.sl_price_ce = self.entry_price_ce * (1 + self.sl_pct)
                    self.tgt_price_ce = self.entry_price_ce * (1 - self.tgt_pct)
                    self.position_ce = 1

        if self.position_pe == 0:
            self.symbol_pe = self._select_leg_symbol('PE', self.selector_val_pe_)
            self.symbol_pe_hedge = self._select_hedge_symbol('PE')
            if self.symbol_pe and self.symbol_pe_hedge:
                success, entry, hedge_entry = self.place_spread_trade(self.now, 'SELL', self.lot_size,
                                                                      self.symbol_pe, self.symbol_pe_hedge, note='ENTRY')
                if success:
                    self.entry_price_pe = entry - hedge_entry
                    self.sl_price_pe = self.entry_price_pe * (1 + self.sl_pct)
                    self.tgt_price_pe = self.entry_price_pe * (1 - self.tgt_pct)
                    self.position_pe = 1

        # ----------- Exit Logic -------------
        self._handle_exit('CE')
        self._handle_exit('PE')

        # ----------- Re-entry Logic -------------
        self._handle_reentry('CE')
        self._handle_reentry('PE')

    # ----------- Helper Functions -------------
    def _select_leg_symbol(self, option_type, val):
        if self.selector == 'P':
            atm_ce = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'CE', 0)
            atm_pe = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'PE', 0)
            if atm_ce and atm_pe:
                atm_price = float(self.get_tick(self.now, atm_ce)['c']) + float(self.get_tick(self.now, atm_pe)['c'])
                return self.find_symbol_by_premium(self.now, self.underlying, 0, option_type,
                                                   atm_price * self.sell_prem_perc, seek_type='lt')
            return None
        else:
            return self.find_symbol_by_moneyness(self.now, self.underlying, 0, option_type, val)

    def _select_hedge_symbol(self, option_type):
        atm_ce = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'CE', 0)
        atm_pe = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'PE', 0)
        if atm_ce and atm_pe:
            atm_price = float(self.get_tick(self.now, atm_ce)['c']) + float(self.get_tick(self.now, atm_pe)['c'])
            return self.find_symbol_by_premium(self.now, self.underlying, 0, option_type,
                                               atm_price * self.hedge_prem_perc, seek_type='lt', perform_rms_checks=False)
        return None

    def _handle_exit(self, leg):
        if getattr(self, f'position_{leg.lower()}') != 1:
            return
        sym = getattr(self, f'symbol_{leg.lower()}')
        hedge = getattr(self, f'symbol_{leg.lower()}_hedge')
        price = float(self.get_tick(self.now, sym)['c']) - float(self.get_tick(self.now, hedge)['c'])
        sl_price = getattr(self, f'sl_price_{leg.lower()}')
        tgt_price = getattr(self, f'tgt_price_{leg.lower()}')
        to_exit = False

        if self.trail_on:
            new_sl = price * (1 + self.sl_pct)
            setattr(self, f'sl_price_{leg.lower()}', min(sl_price, new_sl))

        if price >= sl_price:
            to_exit = True
            setattr(self, f'reason_{leg.lower()}', 'SL')
            if self.sl_to_cost:
                opp = 'pe' if leg.lower() == 'ce' else 'ce'
                setattr(self, f'sl_price_{opp}', getattr(self, f'entry_price_{opp}'))
        elif price <= tgt_price:
            to_exit = True
            setattr(self, f'reason_{leg.lower()}', 'TGT')
        elif self.now.time() >= self.stop_time:
            to_exit = True
            setattr(self, f'reason_{leg.lower()}', 'TIME')

        if to_exit:
            success, _, _ = self.place_spread_trade(self.now, 'BUY', self.lot_size,
                                                    sym, hedge, note=getattr(self, f'reason_{leg.lower()}'))
            if success:
                setattr(self, f'position_{leg.lower()}', -1)

    def _handle_reentry(self, leg):
        pos = getattr(self, f'position_{leg.lower()}')
        reason = getattr(self, f'reason_{leg.lower()}', '')
        reset_count = getattr(self, f'reset_count_{leg.lower()}')
        if pos != -1 or reason != self.reentry_type or reset_count >= self.max_reset:
            return
        setattr(self, f'position_{leg.lower()}', 0)
        setattr(self, f'reset_count_{leg.lower()}', reset_count + 1)
