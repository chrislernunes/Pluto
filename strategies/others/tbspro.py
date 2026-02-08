"""
Strategy Name: TBS-PRO (tbspro) — Time-Based Strangle Professional

Overview:
---------
TBS-PRO is an advanced options strangle strategy framework designed for fine-grained control over strike selection,
hedging, risk management, and re-entry mechanisms. It is intended for professional traders and strategy researchers
looking to run large-scale basket backtests with deep customization.

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

4. **KIKO Logic (`kiko_type`)**:
   KIKO (Kick-In/Kick-Out) is used to delay entry until price conditions are met.
   - `0` (None): No conditions, enter immediately when time condition is met.
   - `1` (KO): Entry only if price falls below a certain % (Kick-Out).
   - `2` (KI): Entry only if price rises above a certain % (Kick-In).
   - `3` (KIKO): First Kick-In (rise), then Kick-Out (fall).
   - `4` (KOKI): First Kick-Out (fall), then Kick-In (rise).
   - `ko_val`, `ki_val`: % thresholds from initial price to trigger KO or KI.

5. **Stop Loss & Target (`sl_type`)**:
   - `'PCT'`: Stop loss defined as % of entry premium.
   - `'ABS'`: Absolute point-based stop loss.
   - `trail_on`: Optional trailing stop mechanism.
   - `tgt_pct`: Target as % of entry premium.

6. **Reentry Limit**:
`reentry_limit` (max re-entries)

7. **MTM Risk Control**:
   - `max_loss`: Absolute MTM stopout for the full strategy.
   - `lock_profit`: If enabled, once a certain profit is hit (`lock_profit_trigger`), a new tighter stop is activated (`lock_profit_to`), and trails further profits using:
     - `subsequent_profit_step`
     - `subsequent_profit_amt`

Parameters:
-----------
tbspro_
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
11. kiko_type                -> 0=None, 1=KO, 2=KI, 3=KIKO, 4=KOKI
12. ki_val                   -> % trigger for knock-in
13. ko_val                   -> % trigger for knock-out
14. sl_type                  -> 'PCT' or 'ABS'
15. sl_val                   -> SL value (percentage or absolute depending on sl_type)
16. trail_on                 -> Whether trailing SL is enabled (True/False)
17. reentry_limit            -> Max number of re-entries allowed
18. tgt_pct                  -> Target as % of entry
19. reenter_on_tgt           -> If True, reentry is allowed even after a Target exit
20. lock_profit              -> Whether MTM profit-locking is enabled (True/False)
21. max_loss                 -> Max allowable MTM loss to trigger hard stop
22. lock_profit_trigger      -> Profit threshold after which max_loss is tightened
23. lock_profit_to           -> New max_loss after profit-locking is activated
24. subsequent_profit_step   -> How often to trail max_loss further (step)
25. subsequent_profit_amt    -> Amount by which max_loss increases after each step

Example UID:
------------
tbspro_NIFTY_0_x0_15_15_1_0_P_100_3_3_0.05_0.1_PCT_0.75_True_5_0.5_True_True_-3000_500_250_200_100
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
from utils.utility import *
#################

class TBSPRO(EventInterface):
    
    def __init__(self, conn=None):
        super().__init__(conn)
        self.strat_id = self.__class__.__name__.lower()

    def get_random_uid(self):
        # Select
        self.underlying = np.random.choice(['NIFTY', 'SENSEX'])
        #self.active_dte = np.random.choice([0])
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
        self.hedge_shift = 0#np.random.choice([3,5,10])
        # KIKO
        # 0=None, 1=KO, 2=KI, 3=KIKO, 4=KOKI
        self.kiko_type = 0#np.random.choice([0,1,2,3,4])
        if self.kiko_type == 0:
            self.ki_val = 0.0
            self.ko_val = 0.0
        elif self.kiko_type == 1:
            self.ki_val = 0.0
            self.ko_val = np.random.choice([0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.25,0.3,0.35,0.4])
        elif self.kiko_type == 2:
            self.ki_val = np.random.choice([0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.25,0.3,0.35,0.4])
            self.ko_val = 0.0
        elif self.kiko_type in [3,4]:
            self.ki_val = np.random.choice([0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.25,0.3,0.35,0.4])
            self.ko_val = np.random.choice([0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.25,0.3,0.35,0.4])
        # SL
        self.sl_type = 'PCT' #np.random.choice(['PCT', 'ABS']) 
        if self.sl_type == 'PCT':
            self.sl_val = np.random.choice([0.2,0.5,0.75,1.0,1.5,2.0])
        elif self.sl_type == 'ABS':
            self.sl_val = np.random.choice(range(5,251,5))
        # TGT
        self.tgt_pct = np.random.choice([0.1,0.25,0.5,0.75,0.8,0.9,0.99])
        # TSL
        self.trail_on = False#np.random.choice([True, False])
        # REENTRY 
        self.reentry_limit = 0#np.random.choice([0,1,2,3,4,5,99])
        # RENTER ON TGT
        self.reenter_on_tgt = False#np.random.choice([True,False])
        # ...
        # MTM LOCKING
        self.lock_profit = False#np.random.choice([True,False])
        if self.lock_profit:
            self.max_loss = np.random.choice(np.arange(-5000,-500,100))
            self.lock_profit_trigger = np.random.choice(np.arange(100,2000,100))
            self.lock_profit_to = int(self.lock_profit_trigger*np.random.choice(np.arange(0.3,0.99,0.1)))#locks 10%, to 100% 
            self.subsequent_profit_step = np.random.choice(np.arange(100,1000,100))
            self.subsequent_profit_amt = int(self.subsequent_profit_step*np.random.choice(np.arange(0.3,0.99,0.2)))
        else:
            #self.max_loss = -10000
            self.max_loss = -10000#np.random.choice(np.arange(-5000,-500,100))
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
        self.kiko_type = int(s.pop(0))
        self.ki_val = float(s.pop(0))
        self.ko_val = float(s.pop(0))
        self.sl_type = s.pop(0)
        if self.sl_type == 'PCT':
            self.sl_val = float(s.pop(0))
        elif self.sl_type == 'ABS':
            self.sl_val = int(s.pop(0))
        self.trail_on = s.pop(0)=='True'
        self.reentry_limit = int(s.pop(0))
        self.tgt_pct = float(s.pop(0))
        self.reenter_on_tgt = s.pop(0)=='True'
        self.lock_profit = s.pop(0)=='True'
        self.max_loss = int(s.pop(0))
        self.lock_profit_trigger = int(s.pop(0))
        self.lock_profit_to = int(s.pop(0))
        self.subsequent_profit_step = int(s.pop(0))
        self.subsequent_profit_amt = int(s.pop(0))
        # CROSS CHECK
        assert len(s)==0
        self.gen_uid = self.get_uid_from_params()
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
        {self.kiko_type}_
        {self.ki_val}_
        {self.ko_val}_
        {self.sl_type}_
        {self.sl_val}_
        {self.trail_on}_
        {self.reentry_limit}_
        {self.tgt_pct}_
        {self.reenter_on_tgt}_
        {self.lock_profit}_
        {self.max_loss}_
        {self.lock_profit_trigger}_
        {self.lock_profit_to}_
        {self.subsequent_profit_step}_
        {self.subsequent_profit_amt}_
        """.replace('\n', '').replace(' ', '').strip('_')

    def on_new_day(self):
        if self.underlying == 'BANKNIFTY':
            self.mysymbol = 'BANKNIFTY-I'
        elif self.underlying == 'NIFTY':
            self.mysymbol = 'NIFTY-I'
        elif self.underlying == 'FINNIFTY':
            self.mysymbol = 'FINNIFTYSPOT'
        elif self.underlying == 'MIDCPNIFTY':
            self.mysymbol = "MIDCPNIFTY-I"
        elif self.underlying == 'SENSEX':
            self.mysymbol = "SENSEX-I" 

        else:
            raise ValueError(f'Unknown Underlying: {self.underlying}')
        # ...
        self.lot_size = self.get_lot_size(self.underlying)
        # ...
        self.symbol_ce = None
        self.symbol_pe = None
        # ...
        self.trade_triggered = False
        # ...
        self.ki_price_ce = None 
        self.ko_price_ce = None
        # ...
        self.ki_price_pe = None
        self.ko_price_pe = None
        # ...
        self.ki_done_ce = False
        self.ko_done_ce = False
        # ...
        self.ki_done_pe = False
        self.ko_done_pe = False
        # ...
        self.triggered_ce = False
        self.triggered_pe = False
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
        self.max_reset = self.reentry_limit #= np.random.choice([0,1,2,3,4,5,99])

    def on_event(self):
        pass
    
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
                #_, _ = self.place_trade(
                #        self.now, 'SELL', self.lot_size, self.symbol_ce_hedge, note='EXTRAHEDGE'
                #)
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
                #_, _ = self.place_trade(
                #        self.now, 'SELL', self.lot_size, self.symbol_pe_hedge, note='EXTRAHEDGE'
                #)
                if self.success_pe:
                    self.position_pe = -1
                    if self.reason_pe == 'TGT' and self.reenter_on_tgt:
                        self.position_pe = 0
        #######################################################
        # TIMEFRAME
        if self.now.time().minute % self.timeframe != self.offset and self.now.time() < self.stop_time:
            return
        #######################################################
        # =======================
        # RE-ENTRY (KIKO-based)
        # =======================
        # Reentry for CE
        if self.position_ce == -1 and self.reset_count_ce < self.max_reset:
            self.current_price_ce = self.get_tick(self.now, self.symbol_ce)['c']
            
            if not self.triggered_ce:
                if self.kiko_type == 0:
                    self.triggered_ce = True
                elif self.kiko_type == 1:
                    if self.ko_price_ce is None and not np.isnan(self.current_price_ce):
                        self.ko_price_ce = self.current_price_ce * (1 - self.ko_val)
                    elif self.current_price_ce <= self.ko_price_ce:
                        self.triggered_ce = True
                elif self.kiko_type == 2:
                    if self.ki_price_ce is None and not np.isnan(self.current_price_ce):
                        self.ki_price_ce = self.current_price_ce * (1 + self.ki_val)
                    elif self.current_price_ce >= self.ki_price_ce:
                        self.triggered_ce = True
                elif self.kiko_type == 3:
                    if self.ki_price_ce is None and not np.isnan(self.current_price_ce):
                        self.ki_price_ce = self.current_price_ce * (1 + self.ki_val)
                    elif self.current_price_ce >= self.ki_price_ce:
                        self.ki_done_ce = True
                    if self.ki_done_ce:
                        if self.ko_price_ce is None and not np.isnan(self.current_price_ce):
                            self.ko_price_ce = self.current_price_ce * (1 - self.ko_val)
                        elif self.current_price_ce <= self.ko_price_ce:
                            self.triggered_ce = True
                elif self.kiko_type == 4:
                    if self.ko_price_ce is None and not np.isnan(self.current_price_ce):
                        self.ko_price_ce = self.current_price_ce * (1 - self.ko_val)
                    elif self.current_price_ce <= self.ko_price_ce:
                        self.ko_done_ce = True
                    if self.ko_done_ce:
                        if self.ki_price_ce is None and not np.isnan(self.current_price_ce):
                            self.ki_price_ce = self.current_price_ce * (1 + self.ki_val)
                        elif self.current_price_ce >= self.ki_price_ce:
                            self.triggered_ce = True

            if self.triggered_ce:
                if self.hedge_shift == 0:
                    self.symbol_ce_hedge = None
                else:
                    self.symbol_ce_hedge = self.shift_strike_in_symbol(self.symbol_ce, self.hedge_shift)
                    price = self.get_tick(self.now, self.symbol_ce_hedge)['c']
                    if np.isnan(price) or price == 0:
                        self.symbol_ce_hedge = self.find_symbol_by_premium(self.now, self.underlying, 0, "CE", 1, perform_rms_checks=False)

                self.success_ce, self.entry_price_ce, self.entry_price_ce_hedge = self.place_spread_trade(
                    self.now, 'SELL', self.lot_size, self.symbol_ce, self.symbol_ce_hedge, note='REENTRY'
                )
                #_, _ = self.place_trade(self.now, 'BUY', self.lot_size, self.symbol_ce_hedge, note='EXTRAHEDGE')

                if self.success_ce:
                    self.reset_count_ce += 1
                    if self.sl_type == 'PCT':
                        self.sl_price_ce = float(self.entry_price_ce) * (1 + self.sl_val)
                    elif self.sl_type == 'ABS':
                        self.sl_price_ce = float(self.entry_price_ce) + self.sl_val
                    self.tgt_price_ce = float(self.entry_price_ce) * (1 - self.tgt_pct)
                    self.position_ce = 1
                    self.ce_high = 0
                    self.ki_price_ce = None
                    self.ko_price_ce = None
                    self.ki_done_ce = False
                    self.ko_done_ce = False
                    self.triggered_ce = False

        # Reentry for PE
        if self.position_pe == -1 and self.reset_count_pe < self.max_reset:
            self.current_price_pe = self.get_tick(self.now, self.symbol_pe)['c']
            
            if not self.triggered_pe:
                if self.kiko_type == 0:
                    self.triggered_pe = True
                elif self.kiko_type == 1:
                    if self.ko_price_pe is None and not np.isnan(self.current_price_pe):
                        self.ko_price_pe = self.current_price_pe * (1 - self.ko_val)
                    elif self.current_price_pe <= self.ko_price_pe:
                        self.triggered_pe = True
                elif self.kiko_type == 2:
                    if self.ki_price_pe is None and not np.isnan(self.current_price_pe):
                        self.ki_price_pe = self.current_price_pe * (1 + self.ki_val)
                    elif self.current_price_pe >= self.ki_price_pe:
                        self.triggered_pe = True
                elif self.kiko_type == 3:
                    if self.ki_price_pe is None and not np.isnan(self.current_price_pe):
                        self.ki_price_pe = self.current_price_pe * (1 + self.ki_val)
                    elif self.current_price_pe >= self.ki_price_pe:
                        self.ki_done_pe = True
                    if self.ki_done_pe:
                        if self.ko_price_pe is None and not np.isnan(self.current_price_pe):
                            self.ko_price_pe = self.current_price_pe * (1 - self.ko_val)
                        elif self.current_price_pe <= self.ko_price_pe:
                            self.triggered_pe = True
                elif self.kiko_type == 4:
                    if self.ko_price_pe is None and not np.isnan(self.current_price_pe):
                        self.ko_price_pe = self.current_price_pe * (1 - self.ko_val)
                    elif self.current_price_pe <= self.ko_price_pe:
                        self.ko_done_pe = True
                    if self.ko_done_pe:
                        if self.ki_price_pe is None and not np.isnan(self.current_price_pe):
                            self.ki_price_pe = self.current_price_pe * (1 + self.ki_val)
                        elif self.current_price_pe >= self.ki_price_pe:
                            self.triggered_pe = True

            if self.triggered_pe:
                if self.hedge_shift == 0:
                    self.symbol_pe_hedge = None
                else:
                    self.symbol_pe_hedge = self.shift_strike_in_symbol(self.symbol_pe, self.hedge_shift)
                    price = self.get_tick(self.now, self.symbol_pe_hedge)['c']
                    if np.isnan(price) or price == 0:
                        self.symbol_pe_hedge = self.find_symbol_by_premium(self.now, self.underlying, 0, "PE", 1, perform_rms_checks=False)

                self.success_pe, self.entry_price_pe, self.entry_price_pe_hedge = self.place_spread_trade(
                    self.now, 'SELL', self.lot_size, self.symbol_pe, self.symbol_pe_hedge, note='REENTRY'
                )
                #_, _ = self.place_trade(self.now, 'BUY', self.lot_size, self.symbol_pe_hedge, note='EXTRAHEDGE')

                if self.success_pe:
                    self.reset_count_pe += 1
                    if self.sl_type == 'PCT':
                        self.sl_price_pe = float(self.entry_price_pe) * (1 + self.sl_val)
                    elif self.sl_type == 'ABS':
                        self.sl_price_pe = float(self.entry_price_pe) + self.sl_val
                    self.tgt_price_pe = float(self.entry_price_pe) * (1 - self.tgt_pct)
                    self.position_pe = 1
                    self.pe_high = 0
                    self.ki_price_pe = None
                    self.ko_price_pe = None
                    self.ki_done_pe = False
                    self.ko_done_pe = False
                    self.triggered_pe = False

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
                    # KIKO CONDITIONS
                    # 0=None, 1=KO, 2=KI, 3=KIKO, 4=KOKI
                    if not self.triggered_ce:
                        if self.kiko_type == 0:
                            self.triggered_ce = True
                        elif self.kiko_type == 1:
                            self.current_price_ce = self.get_tick(self.now, self.symbol_ce)['c']
                            if self.ko_price_ce is None:
                                if not np.isnan(self.current_price_ce):
                                    self.ko_price_ce = self.current_price_ce * (1-self.ko_val) 
                            else:
                                if self.current_price_ce <= self.ko_price_ce:
                                    self.triggered_ce = True
                        elif self.kiko_type == 2:
                            self.current_price_ce = self.get_tick(self.now, self.symbol_ce)['c']
                            if self.ki_price_ce is None:
                                if not np.isnan(self.current_price_ce):
                                    self.ki_price_ce = self.current_price_ce * (1+self.ki_val) 
                            else:
                                if self.current_price_ce >= self.ki_price_ce:
                                    self.triggered_ce = True
                        elif self.kiko_type == 3:
                            self.current_price_ce = self.get_tick(self.now, self.symbol_ce)['c']
                            if self.ki_price_ce is None:
                                if not np.isnan(self.current_price_ce):
                                    self.ki_price_ce = self.current_price_ce * (1+self.ki_val) 
                            else:
                                if self.current_price_ce >= self.ki_price_ce:
                                    self.ki_done_ce = True
                            if self.ki_done_ce:
                                if self.ko_price_ce is None:
                                    if not np.isnan(self.current_price_ce):
                                        self.ko_price_ce = self.current_price_ce * (1-self.ko_val) 
                                else:
                                    if self.current_price_ce <= self.ko_price_ce:
                                        self.triggered_ce = True
                        elif self.kiko_type == 4:
                            self.current_price_ce = self.get_tick(self.now, self.symbol_ce)['c']
                            if self.ko_price_ce is None:
                                if not np.isnan(self.current_price_ce):
                                    self.ko_price_ce = self.current_price_ce * (1-self.ko_val) 
                            else:
                                if self.current_price_ce <= self.ko_price_ce:
                                    self.ko_done_ce = True
                            if self.ko_done_ce:
                                if self.ki_price_ce is None:
                                    if not np.isnan(self.current_price_ce):
                                        self.ki_price_ce = self.current_price_ce * (1+self.ki_val) 
                                else:
                                    if self.current_price_ce >= self.ki_price_ce:
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
                    if not self.triggered_pe:
                        if self.kiko_type == 0:
                            self.triggered_pe = True
                        elif self.kiko_type == 1:
                            self.current_price_pe = self.get_tick(self.now, self.symbol_pe)['c']
                            if self.ko_price_pe is None:
                                if not np.isnan(self.current_price_pe):
                                    self.ko_price_pe = self.current_price_pe * (1-self.ko_val) 
                            else:
                                if self.current_price_pe <= self.ko_price_pe:
                                    self.triggered_pe = True
                        elif self.kiko_type == 2:
                            self.current_price_pe = self.get_tick(self.now, self.symbol_pe)['c']
                            if self.ki_price_pe is None:
                                if not np.isnan(self.current_price_pe):
                                    self.ki_price_pe = self.current_price_pe * (1+self.ki_val) 
                            else:
                                if self.current_price_pe >= self.ki_price_pe:
                                    self.triggered_pe = True
                        elif self.kiko_type == 3:
                            self.current_price_pe = self.get_tick(self.now, self.symbol_pe)['c']
                            if self.ki_price_pe is None:
                                if not np.isnan(self.current_price_pe):
                                    self.ki_price_pe = self.current_price_pe * (1+self.ki_val) 
                            else:
                                if self.current_price_pe >= self.ki_price_pe:
                                    self.ki_done_pe = True
                            if self.ki_done_pe:
                                if self.ko_price_pe is None:
                                    if not np.isnan(self.current_price_pe):
                                        self.ko_price_pe = self.current_price_pe * (1-self.ko_val) 
                                else:
                                    if self.current_price_pe <= self.ko_price_pe:
                                        self.triggered_pe = True
                        elif self.kiko_type == 4:
                            self.current_price_pe = self.get_tick(self.now, self.symbol_pe)['c']
                            if self.ko_price_pe is None:
                                if not np.isnan(self.current_price_pe):
                                    self.ko_price_pe = self.current_price_pe * (1-self.ko_val) 
                            else:
                                if self.current_price_pe <= self.ko_price_pe:
                                    self.ko_done_pe = True
                            if self.ko_done_pe:
                                if self.ki_price_pe is None:
                                    if not np.isnan(self.current_price_pe):
                                        self.ki_price_pe = self.current_price_pe * (1+self.ki_val) 
                                else:
                                    if self.current_price_pe >= self.ki_price_pe:
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

