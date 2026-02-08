import datetime, time
import pandas as pd
import numpy as np
import pandas as pd, pandas_ta as ta

from utils.definitions import *
from utils.sessions import *
import direct_redis

if REDIS:
    from engine.ems import EventInterfacePositional, lot_size_dict, strike_diff_dict
else:
    from engine.ems_db import EventInterfacePositional, lot_size_dict, strike_diff_dict

class PBSARLEGACY(EventInterfacePositional):

    def __init__(self):
        super().__init__()
        self.strat_id = self.__class__.__name__.lower()
        self.signal = 0
        self.last_expiry = None

    def get_random_uid(self):
        # Select
        self.active_weekday = 99#np.random.choice(weekdays)
        self.session = 'x0'# np.random.choice(sessions)
        self.underlying = np.random.choice(underlyings)
        self.selector = 'PCT' # np.random.choice(selectors)
        if self.selector == 'M':
            self.selector_val = np.random.choice(moneynesses)
        elif self.selector == 'P':
            self.selector_val = np.random.choice(seek_prices)
        elif self.selector == 'PCT':
            self.selector_val = np.random.choice([0.0025, 0.0015, 0.00075, 0.0001, 0.0002, 0.00015, 0.0005, 0.0003, 0.00045])
        self.hedge_shift = np.random.choice([5, 7, 10])
        self.sl_pct = round(.05 * round(np.random.choice(np.random.rand(10)*0.5).round(2)/.05), 2)
        self.tgt_pct = round(.05 * round(np.random.choice(np.random.random(1)).round(2)/.05), 2) #np.random.choice(tgt_pcts)
        self.max_reset = np.random.choice([0, 1, 2, 3, 4, 5, 6])
        self.trail_on = np.random.choice([True, False])
        self.delay = np.random.choice(range(0, 30, 5))
        # ...
        self.af = np.random.choice([0.005, 0.01, 0.015, 0.02, 0.025])
        self.max_af = np.random.choice([0.005, 0.01, 0.015, 0.02, 0.025])
        timeframe = [3, 5, 10]
        self.timeframe = np.random.choice(timeframe)
        # ...
        return self.get_uid_from_params()

    def set_params_from_uid(self, uid):
        # print(uid)
        s = uid.split('_')
        try:
            assert s[0] == self.strat_id
        except AssertionError:
            raise ValueError(f'Invalid UID {uid} for strat ID {self.strat_id}')
        s = s[1:]
        self.active_weekday = int(s.pop(0))
        self.session = s.pop(0)
        self.delay = int(s.pop(0))#=='True'
        self.underlying = s.pop(0)
        self.selector = s.pop(0)
        if self.selector == 'P' or self.selector == 'M':
            self.selector_val = int(s.pop(0))
        elif self.selector == 'PCT':
            self.selector_val = float(s.pop(0))
        self.hedge_shift = int(s.pop(0))
        self.sl_pct = float(s.pop(0))
        self.tgt_pct = float(s.pop(0))
        self.trail_on = s.pop(0)=='True'
        self.max_reset = int(s.pop(0))
        # ...
        self.af = float(s.pop(0))
        self.max_af = float(s.pop(0))
        self.timeframe = int(s.pop(0))
        self.itmperc = float(s.pop(0))
        # self.indicator_thresh = int(s.pop(0))
        # CROSS CHECK
        assert len(s)==0
        self.gen_uid = self.get_uid_from_params()
        print(self.gen_uid)
        assert uid == self.gen_uid
        self.uid = uid
        # print(self.uid)
    
    def get_uid_from_params(self):
        return f"""
        {self.strat_id}_
        {self.active_weekday}_
        {self.session}_
        {self.delay}_
        {self.underlying}_
        {self.selector}_
        {self.selector_val}_
        {self.hedge_shift}_
        {self.sl_pct}_
        {self.tgt_pct}_
        {self.trail_on}_
        {self.max_reset}_
        {self.af}_
        {self.max_af}_
        {self.timeframe}_
        {self.itmperc}
        """.replace('\n', '').replace(' ', '').strip('_')
    
    def update_psar(self):

        resample_tf = str(self.timeframe) + 'min'
        temp_data = self.futprice.copy()
        temp_data = temp_data.resample(resample_tf ,on='ts').agg({'o':'first', 'h': 'max', 'l': 'min', 'c':'last'}).between_time("9:05", "15:30").dropna().reset_index()

        psar = ta.psar(temp_data['h'], temp_data['l'], temp_data['c'], af=self.af, max_af=self.max_af)

        self.psar_values_for_day[self.now] = psar.iloc[-1]

        return psar.iloc[-1]
    
    def calculate_psar(self):
        resample_tf = str(self.timeframe) + 'min'
        data = self.futprice.copy()
        data = data.resample(resample_tf ,on='ts').agg({'o':'first', 'h': 'max', 'l': 'min', 'c':'last'}).between_time("9:05", "15:29").dropna().reset_index()

        # max_acceleration = self.max_af
        # acceleration = self.af
        # psar = [None] * len(data)
        # # print(psar)
        # psar[0] = data.iloc[0]['h'] # Initial PSAR value

        # trend = 1  # 1 for uptrend, -1 for downtrend
        # acceleration_factor = acceleration
        # extreme_point = data.iloc[0]['h']

        # for i in range(1, len(data)):
        #     if trend == 1:
        #         if data.iloc[i - 1]['h'] > extreme_point:
        #             extreme_point = data.iloc[i - 1]['h']
        #             acceleration_factor = min(acceleration_factor + acceleration, max_acceleration)
        #     else:
        #         if data.iloc[i - 1]['l'] < extreme_point:
        #             extreme_point = data.iloc[i - 1]['l']
        #             acceleration_factor = min(acceleration_factor + acceleration, max_acceleration)

        #     psar[i] = psar[i - 1] + acceleration_factor * (extreme_point - psar[i - 1])

        #     if trend == 1:
        #         if data.iloc[i]['l'] < psar[i]:
        #             trend = -1
        #             psar[i] = extreme_point
        #             extreme_point = data.iloc[i]['l']
        #             acceleration_factor = acceleration
        #     else:
        #         if data.iloc[i]['h'] > psar[i]:
        #             trend = 1
        #             psar[i] = extreme_point
        #             extreme_point = data.iloc[i]['h']
        #             acceleration_factor = acceleration

        # psar_df = pd.DataFrame(psar, index=data.index)
        # return psar_df.iloc[-1].values[0]

        ## Optimized Code ##
        # Constants
        max_acceleration = self.max_af
        acceleration = self.af

        # Prepare arrays
        highs = data['h'].values
        lows = data['l'].values
        n = len(data)

        psar = np.full(n, np.nan)  # Pre-allocate PSAR array
        trend = np.ones(n, dtype=int)  # 1 for uptrend, -1 for downtrend
        extreme_point = np.zeros(n)
        acceleration_factor = np.full(n, acceleration)

        # Initial conditions
        psar[0] = highs[0]  # Start with the high as the initial PSAR
        extreme_point[0] = highs[0]

        # Iterative logic using a vectorized approach
        for i in range(1, n):
            # Calculate PSAR
            psar[i] = psar[i - 1] + acceleration_factor[i - 1] * (extreme_point[i - 1] - psar[i - 1])

            if trend[i - 1] == 1:  # Uptrend
                # Update extreme point for uptrend
                extreme_point[i] = max(extreme_point[i - 1], highs[i])
                # Check for trend reversal
                if lows[i] < psar[i]:
                    trend[i] = -1
                    psar[i] = extreme_point[i - 1]
                    extreme_point[i] = lows[i]
                    acceleration_factor[i] = acceleration
                else:
                    trend[i] = 1
                    acceleration_factor[i] = min(acceleration_factor[i - 1] + acceleration, max_acceleration)
            else:  # Downtrend
                # Update extreme point for downtrend
                extreme_point[i] = min(extreme_point[i - 1], lows[i])
                # Check for trend reversal
                if highs[i] > psar[i]:
                    trend[i] = 1
                    psar[i] = extreme_point[i - 1]
                    extreme_point[i] = highs[i]
                    acceleration_factor[i] = acceleration
                else:
                    trend[i] = -1
                    acceleration_factor[i] = min(acceleration_factor[i - 1] + acceleration, max_acceleration)

        # Convert PSAR to DataFrame and return the last value
        psar_df_opt = pd.DataFrame(psar, index=data.index)
        return psar_df_opt.iloc[-1].values[0]


    def on_new_day(self):
        # ...
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
        self.session_vals = sessions_dict[self.session]
        self.start_time = self.session_vals['start_time']# #datetime.time(9, 15)#
        self.stop_time = datetime.time(15,27)

        dtn = datetime.datetime.now()
        self.stop_time = (datetime.datetime.combine(dtn.date(), self.stop_time) - datetime.timedelta(minutes=self.delay)).time()
        self.futprice = self.get_all_ticks_by_symbol(self.mysymbol)
        self.futprice = self.futprice[self.futprice['ts'] < self.now]
        self.futprice = self.futprice.tail(30000).reset_index().drop(['index'], axis=1)

        self.premium_for_day_selection = True
        self.psar_values_for_day = []
        # self.signal = 0
        
    def on_event(self):
        pass

    def on_bar_complete(self):
        # print(self.now, self.symbol_ce, self.symbol_pe)
        if self.now.time() > datetime.time(15, 27):
            return
        
        if self.last_expiry is None:
            self.last_expiry = str(self.now.date())
        # ...
        if self.premium_for_day_selection:
            if self.symbol_ce is None and self.symbol_pe is None:
                atm_pe = self.find_symbol_by_moneyness(self.now, self.underlying, 0, 'PE', 0)
                if atm_pe != None:
                    self.dte = self.get_dte(self.now, atm_pe)
                    self.premium_for_day_selection = False

            elif self.symbol_ce is not None:
                self.dte = self.get_dte(self.now, self.symbol_ce)
                self.premium_for_day_selection = False
                
            elif self.symbol_pe is not None:
                self.dte = self.get_dte(self.now, self.symbol_pe)
                self.premium_for_day_selection = False


        self.current_tick = self.get_tick(self.now, self.mysymbol)
        if self.now.time() == datetime.time(9, 7) and self.mysymbol == 'SENSEXSPOT':
            self.current_tick = self.get_tick(self.now.replace(minute=9), self.mysymbol)
            
        self.underlying_close = self.current_tick['c']
        self.futprice.loc[len(self.futprice.index)] = ([self.current_tick['ts'], self.current_tick['o'], self.current_tick['h'], self.current_tick['l'], self.current_tick['c']])

        # ...
        if self.dte != self.active_weekday and self.active_weekday != 99:
            return
        
        # ROLLOVER LOGIC
        if self.dte == 0 and self.now.time() == datetime.time(14, 50):
            self.roll_over = True
            expiry_idx = 1
        elif self.dte == 0 and self.now.time() > datetime.time(14, 50):
            self.roll_over = False
            expiry_idx = 1
        else:
            self.roll_over = False
            expiry_idx = 0

        # INDICATOR CALC
        if ((self.now + datetime.timedelta(minutes=1)).time().minute % self.timeframe == 0 and self.now.time() < datetime.time(15, 27)) or (self.now.time() == datetime.time(9, 7)):
            
            self.signal_val = self.calculate_psar()
            if self.signal_val > self.underlying_close :
                self.signal = -1
            elif self.signal_val < self.underlying_close :
                self.signal = 1
            else:
                self.signal = 0

            # open("pbsardelta_itmpct_log.txt", "a").write(f"{self.af} | {self.now} | SIGNAL: {self.signal} | TREND: {self.trend} | PSAR: {self.signal_val} | UNDERLYING: {self.underlying_close} | resampled_h : {resampled_candle['h']} | resampled_l: {resampled_candle['l']} | resampled_c: {resampled_candle['c']}\n")

        # EXIT
        if (self.now.time() >= self.start_time and self.now.time() < self.stop_time ) :

            if self.position_ce == 1:
                self.current_price_ce = float(self.get_tick(self.now, self.symbol_ce)['c'])
                if self.trail_on:
                    self.new_sl_ce = self.current_price_ce * (1+self.sl_pct)
                    if self.new_sl_ce < self.sl_price_ce:
                        self.sl_price_ce = self.new_sl_ce
                self.to_exit = False
                # SL
                # if self.current_price_ce >= self.sl_price_ce:
                #     self.to_exit = True
                #     self.reason_ce = 'SL'
                # # TGT
                # if self.current_price_ce <= self.tgt_price_ce:
                #     self.to_exit = True
                #     self.reason_ce = 'TGT'
                # ROLL-OVER
                if self.roll_over:
                    self.to_exit = True
                    self.reason_ce = 'ROLL-OVER'
                
                if self.signal == 1:
                    self.to_exit = True
                    self.reason_ce = 'SIGNAL INVERSION'


                if self.to_exit:
                    self.success_ce, _ = self.place_trade(
                        self.now, 'BUY', self.lot_size, self.symbol_ce, note=self.reason_ce
                    )
                    if self.success_ce:
                        self.position_ce = -1
                        self.symbol_ce = None
            
                        
            if self.position_pe == 1:
                self.current_price_pe = float(self.get_tick(self.now, self.symbol_pe)['c'])
                if self.trail_on:
                    self.new_sl_pe = self.current_price_pe * (1+self.sl_pct)
                    if self.new_sl_pe < self.sl_price_pe:
                        self.sl_price_pe = self.new_sl_pe
                self.to_exit = False
                # SL
                # if self.current_price_pe >= self.sl_price_pe:
                #     self.to_exit = True
                #     self.reason_pe = 'SL'
                # # TGT
                # if self.current_price_pe <= self.tgt_price_pe:
                #     self.to_exit = True
                #     self.reason_pe = 'TGT'
                # # ROLL-OVER
                if self.roll_over:
                    self.to_exit = True
                    self.reason_pe = 'ROLL-OVER'
                
                if self.signal == -1:
                    self.to_exit = True
                    self.reason_pe = 'SIGNAL INVERSION'
                
                if self.to_exit:
                    self.success_pe, _ = self.place_trade(
                        self.now, 'BUY', self.lot_size, self.symbol_pe, note=self.reason_pe
                    )
                    if self.success_pe:
                        self.position_pe = -1
                        self.symbol_pe = None
                        

        # RE-ENTRY
        if self.position_ce == -1:
            #print(self.now, self.current_price_ce)
            if (self.reason_ce == 'SL' or self.reason_ce == 'TGT') and self.reset_count_ce < self.max_reset:
                self.position_ce = 0
                self.reset_count_ce += 1
                
            if self.reason_ce in ['ROLL-OVER', 'SIGNAL INVERSION']:
                self.position_ce = 0
                
        if self.position_pe == -1:    
            #print(self.now, self.current_price_pe)
            if (self.reason_pe == 'SL' or self.reason_pe == 'TGT') and self.reset_count_pe < self.max_reset:
                self.position_pe = 0
                self.reset_count_pe += 1

            if self.reason_pe in ['ROLL-OVER', 'SIGNAL INVERSION']:
                self.position_pe = 0

        # CHANGE ACTIVE DTE
        if self.roll_over :
            self.last_expiry = str(self.now.date())

        # ENTRY
        if self.now.time() >= self.start_time and self.now.time() < self.stop_time :
            if self.position_ce == 0 and self.signal == -1 :
                # select symbol CE
                if self.selector == 'P':
                    self.symbol_ce = self.find_symbol_by_premium(
                        self.now, self.underlying, expiry_idx, 'CE', self.selector_val
                    )
                elif self.selector == 'M':
                    self.symbol_ce = self.find_symbol_by_moneyness(
                        self.now, self.underlying, expiry_idx, 'CE', self.selector_val
                    )
                elif self.selector == 'PCT':
                    self.symbol_ce = self.find_symbol_by_premium(
                    self.now, self.underlying, expiry_idx, 'CE', self.selector_val*self.underlying_close
                    )

                if self.symbol_ce is not None:
                    self.success_ce, self.entry_price_ce = self.place_trade(
                        self.now, 'SELL', self.lot_size, self.symbol_ce, note='ENTRY'
                    )
                    if self.success_ce:
                        self.sl_price_ce = float(self.entry_price_ce)*(1+self.sl_pct)
                        # self.tgt_price_ce = float(self.entry_price_ce)*(1-self.tgt_pct)
                        self.position_ce = 1
                        

            if self.position_pe == 0 and self.signal == 1 :
                # select symbol P1
                if self.selector == 'P':
                    self.symbol_pe = self.find_symbol_by_premium(
                        self.now, self.underlying, expiry_idx, 'PE', self.selector_val
                    )
                elif self.selector == 'M':
                    self.symbol_pe = self.find_symbol_by_moneyness(
                        self.now, self.underlying, expiry_idx, 'PE', self.selector_val
                    )
                elif self.selector == 'PCT':
                    self.symbol_pe = self.find_symbol_by_premium(
                    self.now, self.underlying, expiry_idx, 'PE', self.selector_val*self.underlying_close
                    )

                if self.symbol_pe is not None:
                    self.success_pe, self.entry_price_pe = self.place_trade(
                        self.now, 'SELL', self.lot_size, self.symbol_pe, note='ENTRY'
                    )
                    if self.success_pe:
                        self.sl_price_pe = float(self.entry_price_pe)*(1+self.sl_pct)
                        # self.tgt_price_pe = float(self.entry_price_pe)*(1-self.tgt_pct)
                        self.position_pe = 1
            
                