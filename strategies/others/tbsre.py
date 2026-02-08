"""
Strategy Name: TBS-RE (tbsre) — Time-Based Strangle with Reentry

Overview:
---------
TBS-RE is an options strangle strategy framework designed for fine-grained control over strike selection,
hedging, risk management, and re-entry mechanisms. It provides a lot of customization on reentries, 
risk management, and strike selection compared to vanila TBS.

Core Features:
--------------
1. **Time Control**:
   - `session`, `delay`, `haste`, and `timeframe` define when entries and exits are valid.

2. **Strike Selection (`selector`)**:
   - `'M'`: Moneyness-based (strikes away from ATM)
   - `'P'`: Absolute premium-based
   - `'PCT'`: Strike with a premium that is a % of spot price
   - `'R'`: Ratio of ATM combo divided by DTE (volatility-adjusted)

3. **Hedging**:
   - `hedge_shift`: Defines how far the hedge is from the main position in strikes.
   - Fallbacks to fixed hedge if target premium not available.

4. **Stop Loss & Target (`sl_type`)**:
   - `'PCT'`: Stop loss defined as % of entry premium.
   - `'ABS'`: Absolute point-based stop loss.
   - `trail_on`: Optional trailing stop mechanism.
   - `tgt_pct`: Target as % of entry premium.

5. **Reentry Logic (`reentry_type`)**:
   Defines how the strategy can re-enter after stop-loss.
   - `0`: COST — Reenter only if price retraces back to or below entry.
   - `1`: RETRACE — Reenter when price retraces by a set % from its peak post-SL.
   - `2`: ASAP — Reenter immediately on next valid bar after SL.
   - `3`: BOTH — If one leg exits via SL, exit the other and reenter both.
   - `4`: HOLD — Reenter only if **both** legs exit via SL.
   - Additional controls: `reentry_limit` (max re-entries), `reentry_val` (retrace %)

6. **MTM Risk Control**:
   - `max_loss`: Absolute MTM stopout for the full strategy.
   - `lock_profit`: If enabled, once a certain profit is hit (`lock_profit_trigger`), a new tighter stop is activated (`lock_profit_to`), and trails further profits using:
     - `subsequent_profit_step`
     - `subsequent_profit_amt`

Parameters:
-----------
tbsre_
 1. underlying               -> e.g., 'NIFTY', 'BANKNIFTY'
 2. active_dte               -> Expiry DTE (e.g., 0 = current day expiry)
 3. session                  -> Session ID, e.g., 'x0', 'x1'
 4. delay                    -> Minutes to delay entry from session start
 5. haste                    -> Minutes to advance exit before session end
 6. timeframe                -> Candlestick/bar timeframe (in minutes)
 7. offset                   -> Minute offset within timeframe
 8. selector                 -> 'M', 'P', 'PCT', or 'R'
 9. selector_val             -> Strike distance or premium depending on selector
10. hedge_shift              -> Number of strikes away for hedge leg (0 = no hedge)
11. sl_type                  -> 'PCT' or 'ABS'
12. sl_val                   -> SL value (percentage or absolute depending on sl_type)
13. trail_on                 -> Whether trailing SL is enabled (True/False)
14. reentry_type             -> 0=COST, 1=RETRACE, 2=ASAP, 3=BOTH, 4=HOLD
15. reentry_limit            -> Max number of re-entries allowed
16. reentry_val              -> Retracement % for reentry (only for type=1)
17. tgt_pct                  -> Target as % of entry
18. reenter_on_tgt           -> If True, reentry is allowed even after a Target exit
19. lock_profit             -> Whether MTM profit-locking and trailing is enabled (True/False)
20. max_loss_pct            -> Initial max allowable MTM loss as % of margin (e.g., 0.02 = 2% loss)
21. lock_profit_trigger_pct -> Profit threshold as % of margin to start trailing (e.g., 0.005 = 0.5% profit)
22. lock_trail_pct          -> Fraction of peak profit to trail as dynamic max_loss (e.g., 0.2 = 20%)

Example UID:
------------
tbsre_NIFTY_0_x0_15_15_1_0_P_100_10_PCT_0.75_True_1_2_0.1_0.5_False_True_0.02_0.02_0.2
"""

import datetime
import pandas as pd
import numpy as np

from utils.definitions import *
from utils.utility import *
from utils.sessions import *

if REDIS:
    from engine.ems import EventInterface
else:
    from engine.ems_db import EventInterface

#################

class TBSRE(EventInterface):
    
    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()

    def get_random_uid(self):
        # Select
        self.underlying = np.random.choice(['NIFTY', 'SENSEX'])
        self.active_dte = np.random.choice([0])
        self.session = 'x0'#np.random.choice(sessions)
        self.delay = np.random.choice([0, 5, 15, 30, 45, 60, 90, 120, 240])
        self.haste = 0#np.random.choice([0,5,10,20,30,45,60])
        self.timeframe = 1#np.random.choice([1,2,3,4,5])
        if self.timeframe > 1:
            self.offset = np.random.choice(range(0, self.timeframe))
        else:
            self.offset = 0
        # STRIKE SELECTION
        self.selector = 'P'#np.random.choice(['P', 'M', 'PCT', 'R'])
        if self.selector == 'M':
            self.selector_val = int(np.random.choice(np.arange(-2,11,1)))
        elif self.selector == 'P':
            self.selector_val = int(np.random.choice([10,25,50,75,100,150,200,250,300]))
        elif self.selector == 'R':
            self.selector_val = round(np.random.choice(np.arange(0.4,1.5,0.1)),4)
        elif self.selector == 'PCT':
            self.selector_val = round(np.random.choice(np.arange(0.0005,0.004,0.0005)),4)
        elif self.selector == 'PS':
            self.selector_val = np.random.choice([0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
        self.hedge_shift = 0#np.random.choice([3,5,10])
        # SL
        self.sl_type = 'PCT' #np.random.choice(['PCT', 'ABS']) 
        if self.sl_type == 'PCT':
            self.sl_val = np.random.choice([0.2,0.5,0.75,1.0,1.5,2.0])
        elif self.sl_type == 'ABS':
            self.sl_val = np.random.choice(range(5,251,5))
        # TGT
        self.tgt_pct = np.random.choice([0.1,0.25,0.5,0.75,0.8,0.9,0.99])
        # TSL
        self.trail_on = np.random.choice([True, False])
        # REENTRY 
        # 0=COST, 1=RETRACE, 2=ASAP, 3=BOTH, 4=HOLD
        self.reentry_type = 0#np.random.choice([0,1,2,3,4])
        self.reentry_limit = 0#np.random.choice([0,1,2,3,4,5,99])
        if self.reentry_type == 1:
            self.reentry_val = np.random.choice([0.05,0.1,0.15,0.2])
        else:
            self.reentry_val = 0.0
        # RENTER ON TGT
        self.reenter_on_tgt = np.random.choice([True,False])
        # === MTM LOCKING (SIMPLIFIED, MARGIN-PERCENTAGE BASED) ===
        margin = self.calculate_margin()
        # Decide whether to enable profit-locking
        self.lock_profit = np.random.choice([True, False])
        if self.lock_profit:
            # Randomly select max loss as % of margin (e.g., 1% to 5%)
            self.max_loss_pct = np.random.choice([0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
            # Randomly select profit trigger as % of margin (e.g., 1% to 5%)
            self.lock_profit_trigger_pct = np.random.choice([0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
            # Randomly select trailing percentage (e.g., 10% to 50%)
            self.lock_trail_pct = np.random.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
        else:
            # Disable profit-locking parameters
            self.max_loss_pct = 0.02  # default 2% of margin
            self.lock_profit_trigger_pct = 0.0
            self.lock_trail_pct = 0.0
        # Compute actual margin-scaled values for runtime
        self.max_loss = -self.max_loss_pct * margin   # negative for loss
        self.lock_profit_trigger = self.lock_profit_trigger_pct * margin
        # self.lock_trail_pct is used as-is during trailing calculation
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
        self.sl_type = s.pop(0)
        if self.sl_type == 'PCT':
            self.sl_val = float(s.pop(0))
        elif self.sl_type == 'ABS':
            self.sl_val = int(s.pop(0))
        self.trail_on = s.pop(0)=='True'
        self.reentry_type = int(s.pop(0))
        self.reentry_limit = int(s.pop(0))
        self.reentry_val = float(s.pop(0))
        self.tgt_pct = float(s.pop(0))
        self.reenter_on_tgt = s.pop(0)=='True'
        self.lock_profit = s.pop(0)=='True'
        # === MTM LOCKING PARSE FROM UID (SIMPLIFIED) ===
        margin = MARGIN
        # Pop margin-percentage values from UID
        self.max_loss_pct = float(s.pop(0))                 # e.g., 0.02
        self.lock_profit_trigger_pct = float(s.pop(0))      # e.g., 0.02
        self.lock_trail_pct = float(s.pop(0))               # e.g., 0.2
        # Compute actual rupee values using margin
        self.max_loss = -self.max_loss_pct * margin         # negative for loss
        self.lock_profit_trigger = self.lock_profit_trigger_pct * margin
        # CROSS CHECK
        assert len(s)==0
        self.gen_uid = self.get_uid_from_params()
        assert uid == self.gen_uid
        self.uid = uid
        print(self.uid)
    
    def get_uid_from_params(self):
        return (
            f"{self.strat_id}_"
            f"{self.underlying}_"
            f"{self.active_dte}_"
            f"{self.session}_"
            f"{self.delay}_"
            f"{self.haste}_"
            f"{self.timeframe}_"
            f"{self.offset}_"
            f"{self.selector}_"
            f"{self.selector_val}_"
            f"{self.hedge_shift}_"
            f"{self.sl_type}_"
            f"{self.sl_val}_"
            f"{self.trail_on}_"
            f"{self.reentry_type}_"
            f"{self.reentry_limit}_"
            f"{self.reentry_val}_"
            f"{self.tgt_pct}_"
            f"{self.reenter_on_tgt}_"
            f"{self.lock_profit}_"
            f"{round(self.max_loss_pct, 6)}_"               # store as clean % of margin
            f"{round(self.lock_profit_trigger_pct, 6)}_"    # store as clean % of margin
            f"{round(self.lock_trail_pct, 6)}"              # store as clean fraction
        )

    def on_new_day(self):
        # === Symbol Resolution ===
        underlying_map = {
            'BANKNIFTY': 'BANKNIFTY-I',
            'NIFTY': 'NIFTY-I',
            'FINNIFTY': 'FINNIFTYSPOT',
            'MIDCPNIFTY': 'MIDCPNIFTY-I',
            'SENSEX': 'SENSEX-I'
        }
        try:
            self.mysymbol = underlying_map[self.underlying]
        except KeyError:
            raise ValueError(f'Unknown Underlying: {self.underlying}')
        # === Lot size ===
        self.lot_size = self.get_lot_size(self.underlying)
        # === Initialize Symbols and Positions ===
        self.symbol_ce = None
        self.symbol_pe = None
        self.position_ce = 0
        self.position_pe = 0
        self.reset_count_ce = 0
        self.reset_count_pe = 0
        self.ce_high = 0
        self.pe_high = 0
        # === MTM and Profit Locking ===
        self.mtm = 0
        self.profit_locked_at = 0
        self.profit_locked = False
        self.max_profit_seen = 0
        self.initial_max_loss = self.max_loss
        # === DTE ===
        self.dte = None
        # === Premium Selection Flag ===
        self.premium_for_day_selection = True  # or adjust as noted if needed
        # === Session Handling with Delay and Haste ===
        self.session_vals = sessions_dict[self.session]
        dtn = datetime.datetime.now()
        self.start_time = (datetime.datetime.combine(dtn.date(), self.session_vals['start_time']) + datetime.timedelta(minutes=self.delay)).time()
        self.stop_time = (datetime.datetime.combine(dtn.date(), self.session_vals['stop_time']) - datetime.timedelta(minutes=self.haste)).time()
        # === Reentry Limit ===
        self.max_reset = self.reentry_limit
        self.reason_ce = ''
        self.reason_pe = ''

    def on_event(self):
        pass

    def calculate_margin(self, now):
        spot_price = float(self.get_tick(now, f'{self.underlying}SPOT')['c'])
        if self.underlying.startswith('NIFTY'):
            margin=(spot_price*self.lot_size*2)/14
        else:
            margin=(spot_price*self.lot_size*2)/13
        return margin
    
    def calculate_mn_premium(self, margin, ps):
        mn_premium = margin*ps/self.lot_size
        return mn_premium


    def on_bar_complete(self):
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
            if self.selector == 'PS':
                atm_ce = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'CE', 0)
                if atm_ce != None:
                    self.dte = self.get_dte(self.now, atm_ce)
                    self.premium_for_day_selection = False
                    margin = self.calculate_margin(self.now)
                    mn_premium = self.calculate_mn_premium(margin, self.selector_val)
                    self.premium_for_day = mn_premium

        #######################################################
        # DTE
        if self.dte != self.active_dte and self.active_dte != 99:
            return
        
        #######################################################
        # MTM CALCULATION
        self.mtm = self.get_mtm()
        # Optional: sanity check for catastrophic loss (disabled by default)
        if self.mtm < 0 and self.max_loss < 0 and self.mtm < self.max_loss * 2:
            # raise ValueError(f"MTM loss exceeds 2x max loss: MTM={self.mtm}, Max Loss={self.max_loss}")
            pass
        # Simplified Trailing Profit Locking
        if self.lock_profit:
            if self.mtm > self.lock_profit_trigger:
                # Update the max profit seen for trailing calculation
                self.max_profit_seen = max(self.max_profit_seen, self.mtm)
                # Trailing formula:
                # max_loss = initial_max_loss + lock_trail_pct * max_profit_seen
                self.max_loss = self.initial_max_loss + self.lock_trail_pct * self.max_profit_seen

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
                #_, _ = self.place_trade(
                #        self.now, 'SELL', self.lot_size, self.symbol_ce_hedge, note='EXTRAHEDGE')
                if self.success_ce:
                    self.position_ce = -1
                    if self.reason_ce == 'TGT' and self.reenter_on_tgt: 
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
            # MTM
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
                #_, _ = self.place_trade(
                #        self.now, 'SELL', self.lot_size, self.symbol_pe_hedge, note='EXTRAHEDGE')
                if self.success_pe:
                    self.position_pe = -1
                    if self.reason_pe == 'TGT' and self.reenter_on_tgt:
                        self.position_pe = 0
        #######################################################
        # TIMEFRAME
        if self.now.time().minute % self.timeframe != self.offset and self.now.time() < self.stop_time:
            return
        #######################################################
        # if this strat session has elapsed X%, activate TSL...
        #######################################################
        # ...
        # RE-ENTRY
        # 0=COST, 1=RETRACE, 2=ASAP, 3=BOTH, 4=HOLD
        # 0=COST
        if self.reentry_type == 0: 
            if self.position_ce == -1 and self.reason_ce == 'SL':
                self.current_price_ce = float(self.get_tick(self.now, self.symbol_ce)['c'])
                #print(self.now, self.current_price_ce)
                if self.reset_count_ce < self.max_reset and self.current_price_ce <= float(self.entry_price_ce):
                    self.position_ce = 0
                    self.reset_count_ce += 1
            if self.position_pe == -1 and self.reason_pe == 'SL':
                self.current_price_pe = float(self.get_tick(self.now, self.symbol_pe)['c'])
                #print(self.now, self.current_price_pe)
                if self.reset_count_pe < self.max_reset and self.current_price_pe <= float(self.entry_price_pe):
                    self.position_pe = 0
                    self.reset_count_pe += 1
        # 1=RETRACE
        elif self.reentry_type == 1:
            self.retrace_factor = self.reentry_val# = np.random.choice([0.05,0.1,0.15,0.2])
            if self.position_ce == -1 and self.reason_ce == 'SL':
                self.current_price_ce = self.get_tick(self.now, self.symbol_ce)
                if float(self.current_price_ce['h']) > self.ce_high: 
                    self.ce_high = float(self.current_price_ce['h'])
                if self.reset_count_ce < self.max_reset and self.current_price_ce['c'] <= float(self.ce_high * (1-self.retrace_factor)) and self.current_price_ce['c']<=self.entry_price_ce*2:
                    self.position_ce = 0
                    self.reset_count_ce += 1
            if self.position_pe == -1 and self.reason_pe == 'SL':
                self.current_price_pe = self.get_tick(self.now, self.symbol_pe)
                if float(self.current_price_pe['h']) > self.pe_high:
                    self.pe_high = float(self.current_price_pe['h'])
                if self.reset_count_pe < self.max_reset and self.current_price_pe['c'] <= float(self.pe_high * (1-self.retrace_factor)) and self.current_price_pe['c']<=self.entry_price_pe*2:
                    self.position_pe = 0
                    self.reset_count_pe += 1
        # 2=ASAP
        elif self.reentry_type == 2:
            if self.position_ce == -1:
                if self.reason_ce == 'SL' and self.reset_count_ce < self.max_reset:
                    self.position_ce = 0
                    self.reset_count_ce += 1
                    self.symbol_ce = None
            if self.position_pe == -1:
                if self.reason_pe == 'SL' and self.reset_count_pe < self.max_reset:
                    self.position_pe = 0
                    self.reset_count_pe += 1
                    self.symbol_pe = None
        # 3=BOTHEXIT
        elif self.reentry_type == 3:
            if self.position_ce == -1 and self.position_pe == 1:
                self.reason_pe = 'FBE'
                self.success_pe, _, _ = self.place_spread_trade(
                    self.now, 'BUY', self.lot_size, self.symbol_pe, self.symbol_pe_hedge, note=self.reason_pe
                )
                if self.success_pe:
                    self.position_pe = -1
            if self.position_pe == -1 and self.position_ce == 1:
                self.reason_ce = 'FBE'
                self.success_ce, _, _ = self.place_spread_trade(
                    self.now, 'BUY', self.lot_size, self.symbol_ce, self.symbol_ce_hedge, note=self.reason_ce
                )
                if self.success_ce:
                    self.position_ce = -1
            # if both are exited
            if self.position_ce == -1 and self.position_pe == -1:
                # and reason for exit is SL + forced by reentry
                if (self.reason_ce == 'SL' and self.reason_pe == 'FBE') or (self.reason_ce == 'FBE' and self.reason_pe == 'SL'): 
                    # and reset count less than max resets
                    if self.reset_count_ce < self.max_reset and self.reset_count_pe < self.max_reset:
                        # then reset for another entry
                        self.position_ce = 0
                        self.position_pe = 0
                        self.reset_count_ce += 1
                        self.reset_count_pe += 1
                        self.symbol_ce = None
                        self.symbol_pe = None
        # 4=BOTHSL // HOLD
        elif self.reentry_type == 4:
            if self.position_ce == -1 and self.position_pe == -1:
                if self.reason_ce == 'SL' and self.reason_pe == 'SL' and self.reset_count_ce < self.max_reset and self.reset_count_pe < self.max_reset:
                    self.position_ce = 0
                    self.position_pe = 0
                    self.reset_count_ce += 1
                    self.reset_count_pe += 1
                    self.symbol_ce = None
                    self.symbol_pe = None
        else:
            raise ValueError(f'Unkown re-entry type: {self.reentry_type}')
        #######################################################
        # ENTRY
        if self.now.time() >= self.start_time and self.now.time() < self.stop_time:
            if self.position_ce == 0:
                # select symbol CE
                if self.symbol_ce is None:
                    if self.selector in ['P', 'R', 'PCT', 'PS']:
                        self.symbol_ce = self.find_symbol_by_premium(
                            self.now, self.underlying, 0, 'CE', self.premium_for_day, force_atm=False, perform_rms_checks=False
                        )
                    else:
                        self.symbol_ce = self.find_symbol_by_moneyness(
                            self.now, self.underlying, 0, 'CE', self.selector_val
                        )
                if self.symbol_ce is not None:
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
                    #_, _ = self.place_trade(
                    #        self.now, 'BUY', self.lot_size, self.symbol_ce_hedge, note='EXTRAHEDGE')
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
                # select symbol PE
                if self.symbol_pe is None:
                    if self.selector in ['P', 'R', 'PCT', 'PS']:
                        self.symbol_pe = self.find_symbol_by_premium(
                            self.now, self.underlying, 0, 'PE', self.premium_for_day, force_atm=False, perform_rms_checks=False
                        )
                    else:
                        self.symbol_pe = self.find_symbol_by_moneyness(
                            self.now, self.underlying, 0, 'PE', self.selector_val
                        )
                if self.symbol_pe is not None:
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
                    #_, _ = self.place_trade(
                    #        self.now, 'BUY', self.lot_size, self.symbol_pe_hedge, note='EXTRAHEDGE')
                    if self.success_pe:
                        if self.sl_type == 'PCT':
                            self.sl_price_pe = float(self.entry_price_pe)*(1+self.sl_val)
                        elif self.sl_type == 'ABS':
                            self.sl_price_pe = float(self.entry_price_pe)+self.sl_val
                        self.tgt_price_pe = float(self.entry_price_pe)*(1-self.tgt_pct)
                        self.position_pe = 1
                        self.success_pe = False
                        self.pe_high = 0

